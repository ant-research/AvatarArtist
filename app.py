import os
import sys
import warnings
import logging
import argparse
import json
import random
from datetime import datetime

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from natsort import natsorted, ns
from einops import rearrange
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download
import spaces
import gradio as gr
import base64
import imageio_ffmpeg as ffmpeg
import subprocess
from different_domain_imge_gen.landmark_generation import generate_annotation

from transformers import (
    Dinov2Model, CLIPImageProcessor, CLIPVisionModelWithProjection, AutoImageProcessor
)
from Next3d.training_avatar_texture.camera_utils import LookAtPoseSampler, FOV_to_intrinsics

import recon.dnnlib as dnnlib
import recon.legacy as legacy

from DiT_VAE.diffusion.utils.misc import read_config
from DiT_VAE.vae.triplane_vae import AutoencoderKL as AutoencoderKLTriplane
from DiT_VAE.diffusion import IDDPM, DPMS
from DiT_VAE.diffusion.model.nets import TriDitCLIPDINO_XL_2
from DiT_VAE.diffusion.data.datasets import get_chunks

# Get the directory of the current script
father_path = os.path.dirname(os.path.abspath(__file__))

# Add necessary paths dynamically
sys.path.extend([
    os.path.join(father_path, 'recon'),
    os.path.join(father_path, 'Next3d'),
    os.path.join(father_path, 'data_process'),
    os.path.join(father_path, 'data_process/lib')

])

from lib.FaceVerse.renderer import Faceverse_manager
from data_process.input_img_align_extract_ldm_demo import Process
from lib.config.config_demo import cfg
import shutil

# Suppress warnings (especially for PyTorch)
warnings.filterwarnings("ignore")

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="4D Triplane Generation Arguments")

    # Configuration and model checkpoints
    parser.add_argument("--config", type=str, default="./configs/infer_config.py",
                        help="Path to the configuration file.")

    # Generation parameters
    parser.add_argument("--bs", type=int, default=1,
                        help="Batch size for processing.")
    parser.add_argument("--cfg_scale", type=float, default=4.5,
                        help="CFG scale parameter.")
    parser.add_argument("--sampling_algo", type=str, default="dpm-solver",
                        choices=["iddpm", "dpm-solver"],
                        help="Sampling algorithm to be used.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    # parser.add_argument("--select_img", type=str, default=None,
    #                     help="Optional: Select a specific image.")
    parser.add_argument('--step', default=-1, type=int)
    # parser.add_argument('--use_demo_cam', action='store_true', help="Enable predefined camera parameters")
    return parser.parse_args()


def set_env(seed=0):
    """Set random seed for reproducibility across multiple frameworks."""
    torch.manual_seed(seed)  # Set PyTorch seed
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    np.random.seed(seed)  # Set NumPy seed
    random.seed(seed)  # Set Python built-in random module seed
    torch.set_grad_enabled(False)  # Disable gradients for inference


def to_rgb_image(image: Image.Image):
    """Convert an image to RGB format if necessary."""
    if image.mode == 'RGB':
        return image
    elif image.mode == 'RGBA':
        img = Image.new("RGB", image.size, (127, 127, 127))
        img.paste(image, mask=image.getchannel('A'))
        return img
    else:
        raise ValueError(f"Unsupported image type: {image.mode}")


def image_process(image_path, clip_image_processor, dino_img_processor, device):
    """Preprocess an image for CLIP and DINO models."""
    image = to_rgb_image(Image.open(image_path))
    clip_image = clip_image_processor(images=image, return_tensors="pt").pixel_values.to(device)
    dino_image = dino_img_processor(images=image, return_tensors="pt").pixel_values.to(device)
    return dino_image, clip_image


# def video_gen(frames_dir, output_path, fps=30):
#     """Generate a video from image frames."""
#     frame_files = natsorted(os.listdir(frames_dir), alg=ns.PATH)
#     frames = [cv2.imread(os.path.join(frames_dir, f)) for f in frame_files]
#     H, W = frames[0].shape[:2]
#     video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (W, H))
#     for frame in frames:
#         video_writer.write(frame)
#     video_writer.release()


def trans(tensor_img):
    img = (tensor_img.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1) * 255.
    img = img.to(torch.uint8)
    img = img[0].detach().cpu().numpy()

    return img


def get_vert(vert_dir):
    uvcoords_image = np.load(os.path.join(vert_dir))[..., :3]
    uvcoords_image[..., -1][uvcoords_image[..., -1] < 0.5] = 0
    uvcoords_image[..., -1][uvcoords_image[..., -1] >= 0.5] = 1
    return torch.tensor(uvcoords_image.copy()).float().unsqueeze(0)


def generate_samples(DiT_model, cfg_scale, sample_steps, clip_feature, dino_feature, uncond_clip_feature,
                     uncond_dino_feature, device, latent_size, sampling_algo):
    """
    Generate latent samples using the specified diffusion model.

    Args:
        DiT_model (torch.nn.Module): The diffusion model.
        cfg_scale (float): The classifier-free guidance scale.
        sample_steps (int): Number of sampling steps.
        clip_feature (torch.Tensor): CLIP feature tensor.
        dino_feature (torch.Tensor): DINO feature tensor.
        uncond_clip_feature (torch.Tensor): Unconditional CLIP feature tensor.
        uncond_dino_feature (torch.Tensor): Unconditional DINO feature tensor.
        device (str): Device for computation.
        latent_size (tuple): The latent space size.
        sampling_algo (str): The sampling algorithm ('iddpm' or 'dpm-solver').

    Returns:
        torch.Tensor: The generated samples.
    """
    n = 1  # Batch size
    z = torch.randn(n, 8, latent_size[0], latent_size[1], device=device)

    if sampling_algo == 'iddpm':
        z = z.repeat(2, 1, 1, 1)  # Duplicate for classifier-free guidance
        model_kwargs = dict(y=torch.cat([clip_feature, uncond_clip_feature]),
                            img_feature=torch.cat([dino_feature, dino_feature]),
                            cfg_scale=cfg_scale)
        diffusion = IDDPM(str(sample_steps))
        samples = diffusion.p_sample_loop(DiT_model.forward_with_cfg, z.shape, z, clip_denoised=False,
                                          model_kwargs=model_kwargs, progress=True, device=device)
        samples, _ = samples.chunk(2, dim=0)  # Remove unconditional samples

    elif sampling_algo == 'dpm-solver':
        dpm_solver = DPMS(DiT_model.forward_with_dpmsolver,
                          condition=[clip_feature, dino_feature],
                          uncondition=[uncond_clip_feature, dino_feature],
                          cfg_scale=cfg_scale)
        samples = dpm_solver.sample(z, steps=sample_steps, order=2, skip_type="time_uniform", method="multistep")
    else:
        raise ValueError(f"Invalid sampling_algo '{sampling_algo}'. Choose either 'iddpm' or 'dpm-solver'.")

    return samples


def load_motion_aware_render_model(ckpt_path, device):
    """Load the motion-aware render model from a checkpoint."""
    logging.info("Loading motion-aware render model...")
    with dnnlib.util.open_url(ckpt_path, 'rb') as f:
        network = legacy.load_network_pkl(f)  # type: ignore
    logging.info("Motion-aware render model loaded.")
    return network['G_ema'].to(device)


def load_diffusion_model(ckpt_path, latent_size, device):
    """Load the diffusion model (DiT)."""
    logging.info("Loading diffusion model (DiT)...")

    DiT_model = TriDitCLIPDINO_XL_2(input_size=latent_size).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Remove keys that can cause mismatches
    for key in ['pos_embed', 'base_model.pos_embed', 'model.pos_embed']:
        ckpt['state_dict'].pop(key, None)
        ckpt.get('state_dict_ema', {}).pop(key, None)

    state_dict = ckpt.get('state_dict_ema', ckpt)
    DiT_model.load_state_dict(state_dict, strict=False)
    DiT_model.eval()
    logging.info("Diffusion model (DiT) loaded.")
    return DiT_model


def load_vae_clip_dino(config, device):
    """Load VAE, CLIP, and DINO models."""
    logging.info("Loading VAE, CLIP, and DINO models...")

    # Load CLIP image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path)
    image_encoder.requires_grad_(False)
    image_encoder.to(device)

    # Load VAE
    config_vae = OmegaConf.load(config.vae_triplane_config_path)
    vae_triplane = AutoencoderKLTriplane(ddconfig=config_vae['ddconfig'], lossconfig=None, embed_dim=8)
    vae_triplane.to(device)

    vae_ckpt_path = os.path.join(config.vae_pretrained, 'pytorch_model.bin')
    if not os.path.isfile(vae_ckpt_path):
        raise RuntimeError(f"VAE checkpoint not found at {vae_ckpt_path}")

    vae_triplane.load_state_dict(torch.load(vae_ckpt_path, map_location="cpu"))
    vae_triplane.requires_grad_(False)

    # Load DINO model
    dinov2 = Dinov2Model.from_pretrained(config.dino_pretrained)
    dinov2.requires_grad_(False)
    dinov2.to(device)

    # Load image processors
    dino_img_processor = AutoImageProcessor.from_pretrained(config.dino_pretrained)
    clip_image_processor = CLIPImageProcessor()

    logging.info("VAE, CLIP, and DINO models loaded.")
    return vae_triplane, image_encoder, dinov2, dino_img_processor, clip_image_processor


def prepare_working_dir(dir, style):
    print('stylestylestylestylestylestylestyle',style)
    if style:
        return dir
    else:
        import tempfile
        working_dir = tempfile.TemporaryDirectory()
        return working_dir.name


def launch_pretrained():
    from huggingface_hub import hf_hub_download, snapshot_download
    hf_hub_download(repo_id="KumaPower/AvatarArtist", repo_type='model', local_dir="./pretrained_model")


def prepare_image_list(img_dir, selected_img):
    """Prepare the list of image paths for processing."""
    if selected_img and selected_img in os.listdir(img_dir):
        return [os.path.join(img_dir, selected_img)]

    return sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])


def images_to_video(image_folder, output_video, fps=30):
    # Get all image files and ensure correct order
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images = natsorted(images)  # Sort filenames naturally to preserve frame order

    if not images:
        print("❌ No images found in the directory!")
        return

    # Get the path to the FFmpeg executable
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    print(f"Using FFmpeg from: {ffmpeg_exe}")

    # Define input image pattern (expects images named like "%04d.png")
    image_pattern = os.path.join(image_folder, "%04d.png")

    # FFmpeg command to encode video
    command = [
        ffmpeg_exe, '-framerate', str(fps), '-i', image_pattern,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',  # High-quality H.264 encoding
        '-pix_fmt', 'yuv420p', '-b:v', '5000k',  # Ensure compatibility & increase bitrate
        output_video
    ]

    # Run FFmpeg command
    subprocess.run(command, check=True)

    print(f"✅ High-quality MP4 video has been generated: {output_video}")


def model_define():
    args = get_args()
    set_env(args.seed)
    input_process_model = Process(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float32
    logging.info(f"Running inference with {weight_dtype}")

    # Load configuration
    default_config = read_config(args.config)

    # Ensure valid sampling algorithm
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']
    # Load motion-aware render model
    motion_aware_render_model = load_motion_aware_render_model(default_config.motion_aware_render_model_ckpt, device)

    # Load diffusion model (DiT)
    triplane_size = (256 * 4, 256)
    latent_size = (triplane_size[0] // 8, triplane_size[1] // 8)
    sample_steps = args.step if args.step != -1 else {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}[
        args.sampling_algo]
    DiT_model = load_diffusion_model(default_config.DiT_model_ckpt, latent_size, device)

    # Load VAE, CLIP, and DINO
    vae_triplane, image_encoder, dinov2, dino_img_processor, clip_image_processor = load_vae_clip_dino(default_config,
                                                                                                       device)

    # Load normalization parameters
    triplane_std = torch.load(default_config.std_dir).to(device).reshape(1, -1, 1, 1, 1)
    triplane_mean = torch.load(default_config.mean_dir).to(device).reshape(1, -1, 1, 1, 1)

    # Load average latent vector
    ws_avg = torch.load(default_config.ws_avg_pkl).to(device)[0]

    # Set up face verse for amimation
    base_coff = np.load(
        'pretrained_model/temp.npy').astype(
        np.float32)
    base_coff = torch.from_numpy(base_coff).float()
    Faceverse = Faceverse_manager(device=device, base_coeff=base_coff)

    return motion_aware_render_model, sample_steps, DiT_model, \
        vae_triplane, image_encoder, dinov2, dino_img_processor, clip_image_processor, triplane_std, triplane_mean, ws_avg, Faceverse, device, input_process_model


def duplicate_batch(tensor, batch_size=2):
    if tensor is None:
        return None  # 如果是 None，则直接返回
    return tensor.repeat(batch_size, *([1] * (tensor.dim() - 1)))  # 复制 batch 维度


@torch.inference_mode()
@spaces.GPU(duration=200)
def avatar_generation(items, save_path_base, video_path_input, source_type, is_styled, styled_img):
    """
    Generate avatars from input images.

    Args:
        items (list): List of image paths.
        bs (int): Batch size.
        sample_steps (int): Number of sampling steps.
        cfg_scale (float): Classifier-free guidance scale.
        save_path_base (str): Base directory for saving results.
        DiT_model (torch.nn.Module): The diffusion model.
        render_model (torch.nn.Module): The rendering model.
        std (torch.Tensor): Standard deviation normalization tensor.
        mean (torch.Tensor): Mean normalization tensor.
        ws_avg (torch.Tensor): Latent average tensor.
    """
    if is_styled:
        items = [styled_img]
    else:
        items = [items]
    video_folder = "./demo_data/target_video"
    video_name = os.path.basename(video_path_input).split(".")[0]
    target_path = os.path.join(video_folder, 'data_' + video_name)
    exp_base_dir = os.path.join(target_path, 'coeffs')
    exp_img_base_dir = os.path.join(target_path, 'images512x512')
    motion_base_dir = os.path.join(target_path, 'motions')
    label_file_test = os.path.join(target_path, 'images512x512/dataset_realcam.json')

    if source_type == 'example':
        input_img_fvid = './demo_data/source_img/img_generate_different_domain/coeffs/trained_input_imgs'
        input_img_motion = './demo_data/source_img/img_generate_different_domain/motions/trained_input_imgs'
    elif source_type == 'custom':
        input_img_fvid = os.path.join(save_path_base, 'processed_img/dataset/coeffs/input_image')
        input_img_motion = os.path.join(save_path_base, 'processed_img/dataset/motions/input_image')
    else:
        raise ValueError("Wrong type")
    bs = 1
    sample_steps = 20
    cfg_scale = 4.5
    pitch_range = 0.25
    yaw_range = 0.35
    triplane_size = (256 * 4, 256)
    latent_size = (triplane_size[0] // 8, triplane_size[1] // 8)
    for chunk in tqdm(list(get_chunks(items, 1)), unit='batch'):
        if bs != 1:
            raise ValueError("Batch size > 1 not implemented")

        image_dir = chunk[0]

        image_name = os.path.splitext(os.path.basename(image_dir))[0]
        dino_img, clip_image = image_process(image_dir, clip_image_processor, dino_img_processor, device)

        clip_feature = image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        uncond_clip_feature = image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[
            -2]
        dino_feature = dinov2(dino_img).last_hidden_state
        uncond_dino_feature = dinov2(torch.zeros_like(dino_img)).last_hidden_state

        samples = generate_samples(DiT_model, cfg_scale, sample_steps, clip_feature, dino_feature,
                                   uncond_clip_feature, uncond_dino_feature, device, latent_size,
                                   'dpm-solver')

        samples = (samples / 0.3994218)
        samples = rearrange(samples, "b c (f h) w -> b c f h w", f=4)
        samples = vae_triplane.decode(samples)
        samples = rearrange(samples, "b c f h w -> b f c h w")
        samples = samples * std + mean
        torch.cuda.empty_cache()

        save_frames_path_out = os.path.join(save_path_base, image_name, 'out')
        save_frames_path_outshow = os.path.join(save_path_base, image_name, 'out_show')
        save_frames_path_depth = os.path.join(save_path_base, image_name, 'depth')

        os.makedirs(save_frames_path_out, exist_ok=True)
        os.makedirs(save_frames_path_outshow, exist_ok=True)
        os.makedirs(save_frames_path_depth, exist_ok=True)

        img_ref = np.array(Image.open(image_dir))
        img_ref_out = img_ref.copy()
        img_ref = torch.from_numpy(img_ref.astype(np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(device)

        motion_app_dir = os.path.join(input_img_motion, image_name + '.npy')
        motion_app = torch.tensor(np.load(motion_app_dir), dtype=torch.float32).unsqueeze(0).to(device)

        id_motions = os.path.join(input_img_fvid, image_name + '.npy')

        all_pose = json.loads(open(label_file_test).read())['labels']
        all_pose = dict(all_pose)
        if os.path.exists(id_motions):
            coeff = np.load(id_motions).astype(np.float32)
            coeff = torch.from_numpy(coeff).to(device).float().unsqueeze(0)
            Faceverse.id_coeff = Faceverse.recon_model.split_coeffs(coeff)[0]
        motion_dir = os.path.join(motion_base_dir, video_name)
        exp_dir = os.path.join(exp_base_dir, video_name)
        for frame_index, motion_name in enumerate(
                tqdm(natsorted(os.listdir(motion_dir), alg=ns.PATH), desc="Processing Frames")):
            exp_each_dir_img = os.path.join(exp_img_base_dir, video_name, motion_name.replace('.npy', '.png'))
            exp_each_dir = os.path.join(exp_dir, motion_name)
            motion_each_dir = os.path.join(motion_dir, motion_name)

            # Load pose data
            pose_key = os.path.join(video_name, motion_name.replace('.npy', '.png'))

            cam2world_pose = LookAtPoseSampler.sample(
                3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_index / len(os.listdir(motion_dir))),
                3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_index / len(os.listdir(motion_dir))),
                torch.tensor([0, 0, 0], device=device), radius=2.7, device=device)
            pose_show = torch.cat([cam2world_pose.reshape(-1, 16),
                                   FOV_to_intrinsics(fov_degrees=18.837, device=device).reshape(-1, 9)], 1).to(device)

            pose = torch.tensor(np.array(all_pose[pose_key]).astype(np.float32)).float().unsqueeze(0).to(device)

            # Load and resize expression image
            exp_img = np.array(Image.open(exp_each_dir_img).resize((512, 512)))

            # Load expression coefficients
            exp_coeff = torch.from_numpy(np.load(exp_each_dir).astype(np.float32)).to(device).float().unsqueeze(0)
            exp_target = Faceverse.make_driven_rendering(exp_coeff, res=256)

            # Load motion data
            motion = torch.tensor(np.load(motion_each_dir)).float().unsqueeze(0).to(device)

            img_ref_double = duplicate_batch(img_ref, batch_size=2)
            motion_app_double = duplicate_batch(motion_app, batch_size=2)
            motion_double = duplicate_batch(motion, batch_size=2)
            pose_double = torch.cat([pose_show, pose], dim=0)
            exp_target_double = duplicate_batch(exp_target, batch_size=2)
            samples_double = duplicate_batch(samples, batch_size=2)
            # Select refine_net processing method
            final_out = render_model(
                img_ref_double, None, motion_app_double, motion_double, c=pose_double, mesh=exp_target_double,
                triplane_recon=samples_double,
                ws_avg=ws_avg, motion_scale=1.
            )

            # Process output image
            final_out_show = trans(final_out['image_sr'][0].unsqueeze(0))
            final_out_notshow = trans(final_out['image_sr'][1].unsqueeze(0))
            depth = final_out['image_depth'][0].unsqueeze(0)
            depth = -depth
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 2 - 1
            depth = trans(depth)

            depth = np.repeat(depth[:, :, :], 3, axis=2)
            # Save output images
            frame_name = f'{str(frame_index).zfill(4)}.png'
            Image.fromarray(depth, 'RGB').save(os.path.join(save_frames_path_depth, frame_name))
            Image.fromarray(final_out_notshow, 'RGB').save(os.path.join(save_frames_path_out, frame_name))

            Image.fromarray(final_out_show, 'RGB').save(os.path.join(save_frames_path_outshow, frame_name))

        # Generate videos
        images_to_video(save_frames_path_out, os.path.join(save_path_base, image_name + '_out.mp4'))
        images_to_video(save_frames_path_outshow, os.path.join(save_path_base, image_name + '_outshow.mp4'))
        images_to_video(save_frames_path_depth, os.path.join(save_path_base, image_name + '_depth.mp4'))

        logging.info(f"✅ Video generation completed successfully!")
        return os.path.join(save_path_base, image_name + '_out.mp4'), os.path.join(save_path_base,
                                                                                   image_name + '_outshow.mp4'),  os.path.join(save_path_base, image_name + '_depth.mp4')


def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"


def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")


def process_image(input_image, source_type, is_style, save_dir):
    """ 🎯 处理 input_image，根据是否是示例图片执行不同逻辑 """
    process_img_input_dir = os.path.join(save_dir, 'input_image')
    process_img_save_dir = os.path.join(save_dir, 'processed_img')
    os.makedirs(process_img_save_dir, exist_ok=True)
    os.makedirs(process_img_input_dir, exist_ok=True)
    if source_type == "example":
        return input_image, source_type
    else:
        # input_process_model.inference(input_image, process_img_save_dir)
        shutil.copy(input_image, process_img_input_dir)
        input_process_model.inference(process_img_input_dir, process_img_save_dir, is_img=True, is_video=False)
        img_name = os.path.basename(input_image)
        imge_dir = os.path.join(save_dir, 'processed_img/dataset/images512x512/input_image', img_name)
        return imge_dir, source_type  # 这里替换成 处理用户上传图片的逻辑


def style_transfer(processed_image, style_prompt, cfg, strength, save_base):
    """
    🎭 这个函数用于风格转换
    ✅ 你可以在这里填入你的风格化代码
    """
    src_img_pil = Image.open(processed_image)
    img_name = os.path.basename(processed_image)
    save_dir = os.path.join(save_base, 'style_img')
    os.makedirs(save_dir, exist_ok=True)
    control_image = generate_annotation(src_img_pil, max_faces=1)
    trg_img_pil = pipeline_sd(
        prompt=style_prompt,
        image=src_img_pil,
        strength=strength,
        control_image=Image.fromarray(control_image),
        guidance_scale=cfg,
        negative_prompt='worst quality, normal quality, low quality, low res, blurry',
        num_inference_steps=30,
        controlnet_conditioning_scale=1.5
    )['images'][0]
    trg_img_pil.save(os.path.join(save_dir, img_name))
    return os.path.join(save_dir, img_name) # 🚨 这里需要替换成你的风格转换逻辑


def reset_flag():
    return False
css = """
/* ✅ 让所有 Image 居中 + 自适应宽度 */
.gr-image img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: 100%;
    height: auto;
}

/* ✅ 让所有 Video 居中 + 自适应宽度 */
.gr-video video {
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: 100%;
    height: auto;
}

/* ✅ 可选：让按钮和 markdown 居中 */
#generate_block {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 1rem;
}


/* 可选：让整个容器宽一点 */
#main_container {
    max-width: 1280px;   /* ✅ 例如限制在 1280px 内 */
    margin-left: auto;   /* ✅ 水平居中 */
    margin-right: auto;
    padding-left: 1rem;
    padding-right: 1rem;
}

"""

def launch_gradio_app():
    styles = {
        "Ghibli": "Ghibli style avatar, anime style",
        "Pixar": "a 3D render of a face in Pixar style",
        "Lego": "a 3D render of a head of a lego man 3D model",
        "Greek Statue": "a FHD photo of a white Greek statue",
        "Elf": "a FHD photo of a face of a beautiful elf with silver hair in live action movie",
        "Zombie": "a FHD photo of a face of a zombie",
        "Tekken": "a 3D render of a Tekken game character",
        "Devil": "a FHD photo of a face of a devil in fantasy movie",
        "Steampunk": "Steampunk style portrait, mechanical, brass and copper tones",
        "Mario": "a 3D render of a face of Super Mario",
        "Orc": "a FHD photo of a face of an orc in fantasy movie",
        "Masque": "a FHD photo of a face of a person in masquerade",
        "Skeleton": "a FHD photo of a face of a skeleton in fantasy movie",
        "Peking Opera": "a FHD photo of face of character in Peking opera with heavy make-up",
        "Yoda": "a FHD photo of a face of Yoda in Star Wars",
        "Hobbit": "a FHD photo of a face of Hobbit in Lord of the Rings",
        "Stained Glass": "Stained glass style, portrait, beautiful, translucent",
        "Graffiti": "Graffiti style portrait, street art, vibrant, urban, detailed, tag",
        "Pixel-art": "pixel art style portrait, low res, blocky, pixel art style",
        "Retro": "Retro game art style portrait, vibrant colors",
        "Ink": "a portrait in ink style, black and white image",
    }

    with gr.Blocks(analytics_enabled=False, delete_cache=[3600, 3600], css=css, elem_id="main_container") as demo:
        logo_url = "./docs/AvatarArtist.png"
        logo_base64 = get_image_base64(logo_url)
        # 🚀 让 Logo 居中 & 标题对齐
        gr.HTML(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; margin-bottom: 20px;">
                <img src="{logo_base64}" style="height:50px; margin-right: 15px; display: block;" onerror="this.style.display='none'"/>
                <h1 style="font-size: 32px; font-weight: bold;">AvatarArtist: Open-Domain 4D Avatarization</h1>
            </div>
            """
        )

        # 🚀 让按钮在一行对齐
        gr.HTML(
            """
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
                <a title="Website" href="https://kumapowerliu.github.io/AvatarArtist/" target="_blank" rel="noopener noreferrer">
                    <img src="https://img.shields.io/badge/Website-Visit-blue?style=for-the-badge&logo=GoogleChrome">
                </a>
                <a title="arXiv" href="https://arxiv.org/abs/2503.19906" target="_blank" rel="noopener noreferrer">
                    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=for-the-badge&logo=arXiv">
                </a>
                <a title="Github" href="https://github.com/ant-research/AvatarArtist" target="_blank" rel="noopener noreferrer">
                    <img src="https://img.shields.io/github/stars/ant-research/AvatarArtist?style=for-the-badge&logo=github&logoColor=white&color=orange">
                </a>
            </div>
            """
        )
        gr.HTML(
            """
            <div style="text-align: left; font-size: 16px; line-height: 1.6; margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                <strong>🧑‍🎨 How to use this demo:</strong>
                <ol style="margin-top: 10px; padding-left: 20px;">
                    <li><strong>Select or upload a source image</strong> – this will be the avatar's face.</li>
                    <li><strong>Select or upload a target video</strong> – the avatar will mimic this motion.</li>
                    <li><strong>Click the <em>Process Image</em> button</strong> – this prepares the source image to meet our model's input requirements.</li>
                    <li><strong>(Optional)</strong> Click <em>Apply Style</em> to change the appearance of the processed image – we offer a variety of fun styles to choose from!</li>
                    <li><strong>Click <em>Generate Avatar</em></strong> to create the final animated result driven by the target video.</li>
                </ol>
                <p style="margin-top: 10px;"><strong>🎨 Tip:</strong> Try different styles to get various artistic effects for your avatar!</p>
            </div>
            """
        )
        # 🚀 添加重要提示框
        gr.HTML(
            """
            <div style="background-color: #FFDDDD; padding: 15px; border-radius: 10px; border: 2px solid red; text-align: center; margin-top: 20px;">
                <h4 style="color: red; font-size: 18px;">
                    🚨 <strong>Important Notes:</strong> Please try to provide a <u>front-facing</u> or <u>full-face</u> image without obstructions.
                </h4>
                <p style="color: black; font-size: 16px;">
                    ❌ Our demo does <strong>not</strong> support uploading videos with specific motions because processing requires time.<br>
                    ✅ Feel free to check out our <a href="https://github.com/ant-research/AvatarArtist" target="_blank" style="color: red; font-weight: bold;">GitHub repository</a> to drive portraits using your desired motions.
                </p>
            </div>
            """
        )
        # DISPLAY
        image_folder = "./demo_data/source_img/img_generate_different_domain/images512x512/trained_input_imgs"
        video_folder = "./demo_data/target_video"

        examples_images = sorted(
            [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
             f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        examples_videos = sorted(
            [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]
        )
        print(examples_videos)
        source_type = gr.State("example")
        is_from_example = gr.State(value=True)
        is_styled = gr.State(value=False)
        working_dir = gr.State()

        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="input_image"):
                    with gr.TabItem('🎨 Upload Image'):
                        input_image = gr.Image(
                            label="Upload Source Image",
                            value=os.path.join(image_folder, '02057_(2).png'),
                            image_mode="RGB", height=512, container=True,
                            sources="upload", type="filepath"
                        )

                        def mark_as_example(example_image):
                            print("✅ mark_as_example called")
                            return "example", True, False

                        def mark_as_custom(user_image, is_from_example_flag):
                            print("✅ mark_as_custom called")
                            if is_from_example_flag:
                                print("⚠️ Ignored mark_as_custom triggered by example")
                                return "example", False, False
                            return "custom", False, False

                        input_image.change(
                            mark_as_custom,
                            inputs=[input_image, is_from_example],
                            outputs=[source_type, is_from_example, is_styled]  # ✅ 只返回 source_type，不要输出 input_image
                        )

                # ✅ 让 `Examples` 组件单独占一行，并绑定点击事件
                with gr.Row():
                    example_component = gr.Examples(
                        examples=examples_images,
                        inputs=[input_image],
                        examples_per_page=10,
                    )
                    # ✅ 监听 `Examples` 的 `click` 事件
                    example_component.dataset.click(
                        fn=mark_as_example,
                        inputs=[input_image],
                        outputs=[source_type, is_from_example, is_styled]
                    )

            with gr.Column(variant='panel' ):
                with gr.Tabs(elem_id="input_video"):
                    with gr.TabItem('🎬 Target Video'):
                        video_input = gr.Video(
                            label="Select Target Motion",
                            height=512, container=True,interactive=False, format="mp4",
                            value=examples_videos[0]
                        )

                with gr.Row():
                    gr.Examples(
                        examples=examples_videos,
                        inputs=[video_input],
                        examples_per_page=10,
                    )
            with gr.Column(variant='panel' ):
                with gr.Tabs(elem_id="processed_image"):
                    with gr.TabItem('🖼️ Processed Image'):
                        processed_image = gr.Image(
                            label="Processed Image",
                            image_mode="RGB", type="filepath",
                            elem_id="processed_image",
                            height=512,            container=True,
                            interactive=False
                        )
                        processed_image_button = gr.Button("🔧 Process Image", variant="primary")
            with gr.Column(variant='panel' ):
                with gr.Tabs(elem_id="style_transfer"):
                    with gr.TabItem('🎭 Style Transfer'):
                        style_image = gr.Image(
                            label="Style Image",
                            image_mode="RGB", type="filepath",
                            elem_id="style_image",
                            height=512,          container=True,
                            interactive=False
                        )
                        style_choice = gr.Dropdown(
                            choices=list(styles.keys()),
                            label="Choose Style",
                            value="Pixar"
                        )
                        cfg_slider = gr.Slider(
                            minimum=3.0, maximum=10.0, value=7.5, step=0.1,
                            label="CFG Scale"
                        )
                        strength_slider = gr.Slider(
                            minimum=0.4, maximum=0.85, value=0.65, step=0.05,
                            label="SDEdit Strength"
                        )
                        style_button = gr.Button("🎨 Apply Style", interactive=False)
                        gr.Markdown(
                            "⬅️ Please click **Process Image** first. "
                            "**Apply Style** will transform the image in the **Processed Image** panel "
                            "according to the selected style."
                        )


        with gr.Row():
            with gr.Tabs(elem_id="render_output"):
                with gr.TabItem('🎥 Animation Results'):
                    # ✅ 让 `Generate Avatar` 按钮单独占一行
                    with gr.Row():
                        with gr.Column(scale=1, elem_id="generate_block", min_width=200):
                            submit = gr.Button('🚀 Generate Avatar', elem_id="avatarartist_generate", variant='primary',
                                               interactive=False)
                            gr.Markdown("⬇️ Please click **Process Image** first before generating.",
                                        elem_id="generate_tip")

                    # ✅ 让两个 `Animation Results` 窗口并排
                    with gr.Row():
                        output_video = gr.Video(
                            label="Generated Animation Input Video View",
                            format="mp4", height=512, width=512,
                            autoplay=True
                        )

                        output_video_2 = gr.Video(
                            label="Generated Animation Rotate View",
                            format="mp4", height=512, width=512,
                            autoplay=True
                        )

                        output_video_3 = gr.Video(
                            label="Generated Animation Rotate View Depth",
                            format="mp4", height=512, width=512,
                            autoplay=True
                        )
        def apply_style_and_mark(processed_image, style_choice, cfg, strength, working_dir):
            styled = style_transfer(processed_image, styles[style_choice], cfg, strength, working_dir)
            return styled, True

        def process_image_and_enable_style(input_image, source_type, is_styled, wd):
            processed_result, updated_source_type = process_image(input_image, source_type, is_styled, wd)
            return processed_result, updated_source_type, gr.update(interactive=True), gr.update(interactive=True)
        processed_image_button.click(
            fn=prepare_working_dir,
            inputs=[working_dir, is_styled],
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=process_image_and_enable_style,
            inputs=[input_image, source_type, is_styled, working_dir],
            outputs=[processed_image, source_type, style_button, submit],
            queue=True
        )
        style_button.click(
            fn=apply_style_and_mark,
            inputs=[processed_image, style_choice, cfg_slider, strength_slider, working_dir],
            outputs=[style_image, is_styled]
        )
        submit.click(
            fn=avatar_generation,
            inputs=[processed_image, working_dir, video_input, source_type, is_styled, style_image],
            outputs=[output_video, output_video_2, output_video_3],  # ⏳ 稍后展示视频
            queue=True
        )


        demo.queue()
        demo.launch(server_name="0.0.0.0")


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    image_folder = "./demo_data/source_img/img_generate_different_domain/images512x512/trained_input_imgs"
    example_img_names = os.listdir(image_folder)
    render_model, sample_steps, DiT_model, \
        vae_triplane, image_encoder, dinov2, dino_img_processor, clip_image_processor, std, mean, ws_avg, Faceverse, device, input_process_model = model_define()
    controlnet_path = '/nas8/liuhongyu/model/ControlNetMediaPipeFaceold'
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16
    )
    sd_path =  '/nas8/liuhongyu/model/stable-diffusion-2-1-base'
    pipeline_sd = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        sd_path, torch_dtype=torch.float16,
        use_safetensors=True, controlnet=controlnet, variant="fp16"
    ).to(device)
    demo_cam = False
    launch_gradio_app()
