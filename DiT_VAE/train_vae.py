import argparse
import math
import os

import sys

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
sys.path.append((os.path.join(father_path, 'Next3d')))

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch
import logging
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import inspect
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import dnnlib
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from vae.triplane_vae import AutoencoderKL, AutoencoderKLRollOut
from vae.data.dataset_online_vae import TriplaneDataset
from einops import rearrange
from vae.utils.common_utils import instantiate_from_config
from Next3d.training_avatar_texture.triplane_generation import TriPlaneGenerator
import Next3d.legacy as legacy

from torch_utils import misc
import datetime

logger = get_logger(__name__, log_level="INFO")


def collate_fn(data):
    model_names = [example["data_model_name"] for example in data]
    zs = torch.cat([example["data_z"] for example in data], dim=0)
    verts = torch.cat([example["data_vert"] for example in data], dim=0)

    return {
        'model_names': model_names,
        'zs': zs,
        'verts': verts
    }


def rollout_fn(triplane):
    triplane = rearrange(triplane, "b c f h w -> b f c h w")
    b, f, c, h, w = triplane.shape
    triplane = triplane.permute(0, 2, 3, 1, 4).reshape(-1, c, h, f * w)
    return triplane


def unrollout_fn(triplane):
    res = triplane.shape[-2]
    ch = triplane.shape[1]
    triplane = triplane.reshape(-1, ch // 3, res, 3, res).permute(0, 3, 1, 2, 4).reshape(-1, 3, ch, res, res)
    triplane = rearrange(triplane, "b f c h w -> b c f h w")
    return triplane


def triplane_generate(G_model, z, conditioning_params, std, mean, truncation_psi=0.7, truncation_cutoff=14):
    w = G_model.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    triplane = G_model.synthesis(w, noise_mode='const')
    triplane = (triplane - mean) / std
    return triplane


def gan_model(gan_models, device, gan_model_base_dir):
    gan_model_dict = gan_models
    gan_model_load = {}
    for model_name in gan_model_dict.keys():
        model_pkl = os.path.join(gan_model_base_dir, model_name + '.pkl')
        with dnnlib.util.open_url(model_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        gan_model_load[model_name] = G_new
    return gan_model_load


def main(vae_config: str,
         gan_model_config: str,
         output_dir: str,
         std_dir: str,
         mean_dir: str,
         conditioning_params_dir: str,
         gan_model_base_dir: str,
         train_data: Dict,
         train_batch_size: int = 2,
         max_train_steps: int = 500,
         learning_rate: float = 3e-5,
         scale_lr: bool = False,
         lr_scheduler: str = "constant",
         lr_warmup_steps: int = 0,
         adam_beta1: float = 0.5,
         adam_beta2: float = 0.9,
         adam_weight_decay: float = 1e-2,
         adam_epsilon: float = 1e-08,
         max_grad_norm: float = 1.0,
         gradient_accumulation_steps: int = 1,
         gradient_checkpointing: bool = True,
         checkpointing_steps: int = 500,
         pretrained_model_path_zero123: str = None,
         resume_from_checkpoint: Optional[str] = None,
         mixed_precision: Optional[str] = "fp16",
         use_8bit_adam: bool = False,
         rollout: bool = False,
         enable_xformers_memory_efficient_attention: bool = True,
         seed: Optional[int] = None, ):
    *_, config = inspect.getargvalues(inspect.currentframe())
    base_dir = output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    config_vae = OmegaConf.load(vae_config)

    if rollout:
        vae = AutoencoderKLRollOut(ddconfig=config_vae['ddconfig'], lossconfig=config_vae['lossconfig'], embed_dim=8)

    else:
        vae = AutoencoderKL(ddconfig=config_vae['ddconfig'], lossconfig=config_vae['lossconfig'], embed_dim=8)
    print(f"VAE total params = {len(list(vae.named_parameters()))} ")
    if 'perceptual_weight' in config_vae['lossconfig']['params'].keys():
        config_vae['lossconfig']['params']['device'] = str(accelerator.device)
    loss_fn = instantiate_from_config(config_vae['lossconfig'])
    conditioning_params = torch.load(conditioning_params_dir).to(str(accelerator.device))
    data_std = torch.load(std_dir).to(str(accelerator.device)).reshape(1, -1, 1, 1, 1)

    data_mean = torch.load(mean_dir).to(str(accelerator.device)).reshape(1, -1, 1, 1, 1)

    # define the gan model
    print("########## gan model load ##########")
    config_gan_model = OmegaConf.load(gan_model_config)
    gan_model_all = gan_model(config_gan_model['gan_models'], str(accelerator.device), gan_model_base_dir)
    print("########## gan model loaded ##########")
    if scale_lr:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        vae.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataset = TriplaneDataset(**train_data)

    # Preprocessing the dataset

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True, num_workers=2
    )

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32

    # Move text_encode and vae to gpu and cast to weight_dtype

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("trainvae", config=vars(args))

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        if resume_from_checkpoint != "latest":
            accelerator.load_state(resume_from_checkpoint)
        else:
            accelerator.load_state(os.path.join(output_dir, path))

        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch
    else:
        all_final_training_dirs = []
        dirs = os.listdir(base_dir)
        if len(dirs) != 0:
            dirs = [d for d in dirs if d.startswith("2024")]  # specific years
            if len(dirs) != 0:
                base_resume_paths = [os.path.join(base_dir, d) for d in dirs]
                for base_resume_path in base_resume_paths:
                    checkpoint_file_names = os.listdir(base_resume_path)
                    checkpoint_file_names = [d for d in checkpoint_file_names if d.startswith("checkpoint")]
                    if len(checkpoint_file_names) != 0:
                        for checkpoint_file_name in checkpoint_file_names:
                            final_training_dir = os.path.join(base_resume_path, checkpoint_file_name)
                            all_final_training_dirs.append(final_training_dir)
                if len(all_final_training_dirs) != 0:
                    sorted_all_final_training_dirs = sorted(all_final_training_dirs, key=lambda x: int(x.split("-")[1]))
                    latest_dir = sorted_all_final_training_dirs[-1]
                    path = os.path.basename( latest_dir)
                    accelerator.print(f"Resuming from checkpoint {path}")
                    accelerator.load_state(latest_dir)
                    global_step = int(path.split("-")[1])

                    first_epoch = global_step // num_update_steps_per_epoch
                    resume_step = global_step % num_update_steps_per_epoch
                else:
                    accelerator.print(f"Training from start")
            else:
                accelerator.print(f"Training from start")
        else:
            accelerator.print(f"Training from start")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        vae.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     print(epoch)
            #     print(first_epoch)
            #     print(step)
            #     if step % gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
            with accelerator.accumulate(vae):
                # Convert images to latent space
                z_values = batch["zs"].to(weight_dtype)
                model_names = batch["model_names"]

                triplane_values = []
                with torch.no_grad():
                    for z_id in range(z_values.shape[0]):
                        z_value = z_values[z_id].unsqueeze(0)
                        model_name = model_names[z_id]
                        triplane_value = triplane_generate(gan_model_all[model_name], z_value,
                                                           conditioning_params, data_std, data_mean)
                        triplane_values.append(triplane_value)
                triplane_values = torch.cat(triplane_values, dim=0)
                vert_values = batch["verts"].to(weight_dtype)
                triplane_values = rearrange(triplane_values, "b f c h w -> b c f h w")
                if rollout:
                    triplane_values_roll = rollout_fn(triplane_values.clone())
                    reconstructions, posterior = vae(triplane_values_roll)
                    reconstructions_unroll = unrollout_fn(reconstructions)
                    loss, log_dict_ae = loss_fn(triplane_values, reconstructions_unroll, posterior, vert_values,
                                                split="train")
                else:
                    reconstructions, posterior = vae(triplane_values)
                    loss, log_dict_ae = loss_fn(triplane_values, reconstructions, posterior, vert_values,
                                                split="train")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            logs = log_dict_ae
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/triplane_vae.yaml")
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))
