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

from transformers import (
    Dinov2Model, CLIPImageProcessor, CLIPVisionModelWithProjection, AutoImageProcessor
)
from Next3d.training_avatar_texture.camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from data_process.lib.FaceVerse.renderer import Faceverse_manager
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
    os.path.join(father_path, 'Next3d')
])

# Suppress warnings (especially for PyTorch)
warnings.filterwarnings("ignore")

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="4D Triplane Generation Arguments")

    # Configuration and model checkpoints
    parser.add_argument("--config", type=str, default="./configs/infer_config.py",
                        help="Path to the configuration file.")

    # Input data paths
    parser.add_argument("--target_path", type=str, required=True, default='./demo_data/target_video/data_obama',
                        help="Base path of the dataset.")
    parser.add_argument("--img_file", type=str, required=True, default='./demo_data/source_img/img_generate_different_domain/images512x512/demo_imgs',
                        help="Directory containing input images.")
    parser.add_argument("--input_img_motion", type=str,
                        default="./demo_data/source_img/img_generate_different_domain/motions/demo_imgs",
                        help="Directory containing motion features.")
    parser.add_argument("--video_name", type=str, required=True, default='Obama',
                        help="Name of the video.")
    parser.add_argument("--input_img_fvid", type=str,
                        default="./demo_data/source_img/img_generate_different_domain/coeffs/demo_imgs",
                        help="Path to input image coefficients.")

    # Output settings
    parser.add_argument("--output_basedir", type=str, default="./output",
                        help="Base directory for saving output results.")

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
    parser.add_argument("--select_img", type=str, default=None,
                        help="Optional: Select a specific image.")
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--use_demo_cam', action='store_true', help="Enable predefined camera parameters")
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


def image_process(image_path):
    """Preprocess an image for CLIP and DINO models."""
    image = to_rgb_image(Image.open(image_path))
    clip_image = clip_image_processor(images=image, return_tensors="pt").pixel_values.to(device)
    dino_image = dino_img_processor(images=image, return_tensors="pt").pixel_values.to(device)
    return dino_image, clip_image


def video_gen(frames_dir, output_path, fps=30):
    """Generate a video from image frames."""
    frame_files = natsorted(os.listdir(frames_dir), alg=ns.PATH)
    frames = [cv2.imread(os.path.join(frames_dir, f)) for f in frame_files]
    H, W = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (W, H))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


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

def images_to_video(image_folder, output_video, fps=30):
    # 获取所有图片文件，并确保顺序正确
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images = natsorted(images)  # 按文件名排序，确保帧顺序

    if not images:
        print("❌ 目录中没有找到图片！")
        return

    # 获取 FFmpeg 可执行文件路径
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    print(f"Using FFmpeg from: {ffmpeg_exe}")

    # 生成输入文件路径（确保格式为 "%04d.png"）
    image_pattern = os.path.join(image_folder, "%04d.png")

    # FFmpeg 命令
    command = [
        ffmpeg_exe, '-framerate', str(fps), '-i', image_pattern,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',  # 高质量 H.264 编码
        '-pix_fmt', 'yuv420p', '-b:v', '5000k',  # 改善颜色兼容性，增加比特率
        output_video
    ]

    # 运行 FFmpeg
    subprocess.run(command, check=True)

    print(f"✅ 高质量 MP4 视频已生成: {output_video}")
@torch.inference_mode()
def avatar_generation(items, bs, sample_steps, cfg_scale, save_path_base, DiT_model, render_model, std, mean, ws_avg,
                      Faceverse, pitch_range=0.25, yaw_range=0.35, demo_cam=False):
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
    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):
        if bs != 1:
            raise ValueError("Batch size > 1 not implemented")

        image_dir = chunk[0]
        image_name = os.path.splitext(os.path.basename(image_dir))[0]
        dino_img, clip_image = image_process(image_dir)

        clip_feature = image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        uncond_clip_feature = image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[
            -2]
        dino_feature = dinov2(dino_img).last_hidden_state
        uncond_dino_feature = dinov2(torch.zeros_like(dino_img)).last_hidden_state

        samples = generate_samples(DiT_model, cfg_scale, sample_steps, clip_feature, dino_feature,
                                   uncond_clip_feature, uncond_dino_feature, device, latent_size,
                                   args.sampling_algo)

        samples = (samples / default_config.scale_factor)
        samples = rearrange(samples, "b c (f h) w -> b c f h w", f=4)
        samples = vae_triplane.decode(samples)
        samples = rearrange(samples, "b c f h w -> b f c h w")
        samples = samples * std + mean
        torch.cuda.empty_cache()

        save_frames_path_combine = os.path.join(save_path_base, image_name, 'combine')
        save_frames_path_out = os.path.join(save_path_base, image_name, 'out')
        os.makedirs(save_frames_path_combine, exist_ok=True)
        os.makedirs(save_frames_path_out, exist_ok=True)

        img_ref = np.array(Image.open(image_dir))
        img_ref_out = img_ref.copy()
        img_ref = torch.from_numpy(img_ref.astype(np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(device)

        motion_app_dir = os.path.join(args.input_img_motion, image_name + '.npy')
        motion_app = torch.tensor(np.load(motion_app_dir), dtype=torch.float32).unsqueeze(0).to(device)

        id_motions = os.path.join(args.input_img_fvid, image_name + '.npy')

        all_pose = json.loads(open(label_file_test).read())['labels']
        all_pose = dict(all_pose)
        if os.path.exists(id_motions):
            coeff = np.load(id_motions).astype(np.float32)
            coeff = torch.from_numpy(coeff).to(device).float().unsqueeze(0)
            Faceverse.id_coeff = Faceverse.recon_model.split_coeffs(coeff)[0]
        motion_dir = os.path.join(motion_base_dir, args.video_name)
        exp_dir = os.path.join(exp_base_dir, args.video_name)
        for frame_index, motion_name in enumerate(
                tqdm(natsorted(os.listdir(motion_dir), alg=ns.PATH), desc="Processing Frames")):
            exp_each_dir_img = os.path.join(exp_img_base_dir, args.video_name, motion_name.replace('.npy', '.png'))
            exp_each_dir = os.path.join(exp_dir, motion_name)
            motion_each_dir = os.path.join(motion_dir, motion_name)

            # Load pose data
            pose_key = os.path.join(args.video_name, motion_name.replace('.npy', '.png'))
            if demo_cam:
                cam2world_pose = LookAtPoseSampler.sample(
                    3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_index / len(os.listdir(motion_dir))),
                    3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_index / len(os.listdir(motion_dir))),
                    torch.tensor([0, 0, 0], device=device), radius=2.7, device=device)
                pose = torch.cat([cam2world_pose.reshape(-1, 16),
                                  FOV_to_intrinsics(fov_degrees=18.837, device=device).reshape(-1, 9)], 1).to(device)
            else:
                pose = torch.tensor(np.array(all_pose[pose_key]).astype(np.float32)).float().unsqueeze(0).to(device)

            # Load and resize expression image
            exp_img = np.array(Image.open(exp_each_dir_img).resize((512, 512)))

            # Load expression coefficients
            exp_coeff = torch.from_numpy(np.load(exp_each_dir).astype(np.float32)).to(device).float().unsqueeze(0)
            exp_target = Faceverse.make_driven_rendering(exp_coeff, res=256)

            # Load motion data
            motion = torch.tensor(np.load(motion_each_dir)).float().unsqueeze(0).to(device)

            # Select refine_net processing method
            final_out = render_model(
                img_ref, None, motion_app, motion, c=pose, mesh=exp_target, triplane_recon=samples,
                ws_avg=ws_avg, motion_scale=1.
            )

            # Process output image
            final_out = trans(final_out['image_sr'])
            output_img_combine = np.hstack((img_ref_out, exp_img, final_out))

            # Save output images
            frame_name = f'{str(frame_index).zfill(5)}.png'
            Image.fromarray(output_img_combine, 'RGB').save(os.path.join(save_frames_path_combine, frame_name))
            Image.fromarray(final_out, 'RGB').save(os.path.join(save_frames_path_out, frame_name))

        # Generate videos
        images_to_video(save_frames_path_combine, os.path.join(save_path_base, image_name + '_combine.mp4'))
        images_to_video(save_frames_path_out, os.path.join(save_path_base, image_name + '_out.mp4'))
        logging.info(f"✅ Video generation completed successfully!")
        logging.info(f"📂 Combined video saved at: {os.path.join(save_path_base, image_name + '_combine.mp4')}")
        logging.info(f"📂 Output video saved at: {os.path.join(save_path_base, image_name + '_out.mp4')}")


def load_motion_aware_render_model(ckpt_path):
    """Load the motion-aware render model from a checkpoint."""
    logging.info("Loading motion-aware render model...")
    with dnnlib.util.open_url(ckpt_path, 'rb') as f:
        network = legacy.load_network_pkl(f)  # type: ignore
    logging.info("Motion-aware render model loaded.")
    return network['G_ema'].to(device)


def load_diffusion_model(ckpt_path, latent_size):
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


def prepare_image_list(img_dir, selected_img):
    """Prepare the list of image paths for processing."""
    if selected_img and selected_img in os.listdir(img_dir):
        return [os.path.join(img_dir, selected_img)]

    return sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])


if __name__ == '__main__':

    model_path = "./pretrained_model"
    if not os.path.exists(model_path):
        logging.info("📥 Model not found. Downloading from Hugging Face...")
        snapshot_download(repo_id="KumaPower/AvatarArtist", local_dir=model_path)
        logging.info("✅ Model downloaded successfully!")
    else:
        logging.info("🎉 Pretrained model already exists. Skipping download.")

    args = get_args()
    exp_base_dir = os.path.join(args.target_path, 'coeffs')
    exp_img_base_dir = os.path.join(args.target_path, 'images512x512')
    motion_base_dir = os.path.join(args.target_path, 'motions')
    label_file_test = os.path.join(args.target_path, 'images512x512/dataset_realcam.json')
    set_env(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float32
    logging.info(f"Running inference with {weight_dtype}")

    # Load configuration
    default_config = read_config(args.config)

    # Ensure valid sampling algorithm
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # Prepare image list
    items = prepare_image_list(args.img_file, args.select_img)
    logging.info(f"Input images: {items}")

    # Load motion-aware render model
    motion_aware_render_model = load_motion_aware_render_model(default_config.motion_aware_render_model_ckpt)

    # Load diffusion model (DiT)
    triplane_size = (256 * 4, 256)
    latent_size = (triplane_size[0] // 8, triplane_size[1] // 8)
    sample_steps = args.step if args.step != -1 else {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}[
        args.sampling_algo]
    DiT_model = load_diffusion_model(default_config.DiT_model_ckpt, latent_size)

    # Load VAE, CLIP, and DINO
    vae_triplane, image_encoder, dinov2, dino_img_processor, clip_image_processor = load_vae_clip_dino(default_config,
                                                                                                       device)

    # Load normalization parameters
    triplane_std = torch.load(default_config.std_dir).to(device).reshape(1, -1, 1, 1, 1)
    triplane_mean = torch.load(default_config.mean_dir).to(device).reshape(1, -1, 1, 1, 1)

    # Load average latent vector
    ws_avg = torch.load(default_config.ws_avg_pkl).to(device)[0]

    # Set up save directory
    save_root = os.path.join(args.output_basedir, f'{datetime.now().date()}', args.video_name)
    os.makedirs(save_root, exist_ok=True)

    # Set up face verse for amimation
    base_coff = np.load(
        'pretrained_model/temp.npy').astype(
        np.float32)
    base_coff = torch.from_numpy(base_coff).float()
    Faceverse = Faceverse_manager(device=device, base_coeff=base_coff)

    # Run avatar generation
    avatar_generation(items, args.bs, sample_steps, args.cfg_scale, save_root, DiT_model, motion_aware_render_model,
                      triplane_std, triplane_mean, ws_avg, Faceverse, demo_cam=args.use_demo_cam)
