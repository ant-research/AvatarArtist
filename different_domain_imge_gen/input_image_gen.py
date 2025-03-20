import sys
import os
import json
import zipfile
import argparse

sys.path.append(os.getcwd())

from os.path import join as opj

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision.transforms import ToPILImage

ImageFile.LOAD_TRUNCATED_IMAGES = True

import transformers
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)

from landmark_generation import generate_annotation
from natsort import ns, natsorted


class DomainImageGeneration:
    def __init__(
            self, src_path, args, model_name,
            device="cuda:0", clip_skip=2,
            use_anime_vae=False, save_path='',
            only_lora=False, model_base_path='./diffusion_model'
    ):
        """Initialize DomainImageGeneration class"""
        self.model_base_path = model_base_path
        self.device = device
        self.args = args
        self.src_path = src_path
        self.use_model = model_name
        self.only_lora = only_lora

        self.out_path_base = os.path.join(save_path, model_name)
        os.makedirs(self.out_path_base, exist_ok=True)

        self.diffusion_checkpoint_path = self._get_model_path(model_name)
        self.pipe = self._load_pipeline(model_name, use_anime_vae, clip_skip)
        print("All models loaded successfully")

    def _get_model_path(self, model_name):
        """Retrieve the model checkpoint path based on the model name"""
        base_path = self.model_base_path
        if model_name == "stable-diffusion-2-1-base":
            return os.path.join(base_path, "stable-diffusion-2-1-base")
        elif self.only_lora:
            return os.path.join(base_path, "stable-diffusion-v1-5")
        else:
            return os.path.join(base_path, model_name)

    def _load_controlnet(self, model_name):
        """Load the ControlNet model"""
        controlnet_path = os.path.join(self.model_base_path, 'ControlNetMediaPipeFace')
        if model_name == "stable-diffusion-2-1-base":
            controlnet_path += "old"

        return ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=torch.float16
        )

    def _load_pipeline(self, model_name, use_anime_vae, clip_skip):
        """Load the Stable Diffusion ControlNet Img2Img Pipeline"""
        controlnet = self._load_controlnet(model_name)

        if use_anime_vae:
            print("Using Anime VAE")
            anime_vae = AutoencoderKL.from_pretrained(
                "/nas8/liuhongyu/model/kl-f8-anime2", torch_dtype=torch.float16
            )
            pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.diffusion_checkpoint_path, torch_dtype=torch.float16, safety_checker=None,
                vae=anime_vae, controlnet=controlnet
            ).to(self.device)

            self._load_lora(pipeline, "detail-tweaker-lora/add_detail.safetensors")

        elif model_name == "stable-diffusion-2-1-base":
            pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.diffusion_checkpoint_path, torch_dtype=torch.float16,
                use_safetensors=True, controlnet=controlnet, variant="fp16"
            ).to(self.device)

        else:
            text_encoder = transformers.CLIPTextModel.from_pretrained(
                self.diffusion_checkpoint_path,
                subfolder="text_encoder",
                num_hidden_layers=12 - (clip_skip - 1),
                torch_dtype=torch.float16
            )
            pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.diffusion_checkpoint_path, torch_dtype=torch.float16,
                use_safetensors=True, text_encoder=text_encoder,
                controlnet=controlnet, variant="fp16"
            ).to(self.device)

        self._apply_negative_embedding(pipeline, model_name)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config, use_karras_sigmas=True
        )

        print("Target diffusion model loaded")
        return pipeline

    def _load_lora(self, pipeline, lora_name):
        """Load LoRA weights into the model"""
        lora_path = f"/nas8/liuhongyu/model/{lora_name}"
        state_dict, network_alphas = pipeline.lora_state_dict(lora_path)
        pipeline.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=pipeline.unet)

    def _apply_negative_embedding(self, pipeline, model_name):
        """Apply negative embedding (textual inversion)"""
        if model_name not in ["stable-diffusion-xl-base-1.0", "stable-diffusion-2-1-base"]:
            if self.only_lora:
                self._load_lora(pipeline, model_name)
                pipeline.safety_checker = lambda images, clip_input: (images, None)
            else:
                pipeline.load_textual_inversion(
                    "/nas8/liuhongyu/lora_model",
                    weight_name="EasyNegativeV2.safetensors",
                    token="EasyNegative"
                )

    def image_generation(self, prompt,  strength=0.7,
                         guidance_scale=7.5, num_inference_steps=30):
        """Generate images using the diffusion model"""
        out_path = os.path.join(self.out_path_base, prompt.replace(" ", "_"))
        os.makedirs(out_path, exist_ok=True)
        src_img_list = natsorted(
            [f for f in os.listdir(self.src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))],
            alg=ns.PATH
        )

        all_gen_nums = 0  # Counter for generated images

        for img_name in src_img_list:
            src_img_pil = Image.open(os.path.join(self.src_path, img_name))
            control_image = generate_annotation(src_img_pil, max_faces=1)

            if control_image is not None:
                prompt_input = prompt

                # Apply different generation methods based on the model type
                if self.use_model in ['stable-diffusion-xl-base-1.0', 'stable-diffusion-2-1-base']:
                    trg_img_pil = self.pipe(
                        prompt=prompt_input,
                        image=src_img_pil,
                        strength=strength,
                        control_image=Image.fromarray(control_image),
                        guidance_scale=guidance_scale,
                        negative_prompt='worst quality, normal quality, low quality, low res, blurry',
                        num_inference_steps=num_inference_steps,
                        controlnet_conditioning_scale=1.5
                    )['images'][0]
                else:
                    trg_img_pil = self.pipe(
                        prompt=prompt_input,
                        image=src_img_pil,
                        control_image=Image.fromarray(control_image),
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        controlnet_conditioning_scale=1.5,
                        negative_prompt='EasyNegative, worst quality, normal quality, low quality, low res, blurry'
                    )['images'][0]

                # Save the generated image if valid
                if np.array(trg_img_pil).max() > 0:
                    trg_img_pil.save(opj(out_path, img_name))
                    all_gen_nums += 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Domain Image Generation")

    parser.add_argument(
        "--src_img_path",
        type=str,
        default="demo_input",
        help="Path to the source image directory"
    )
    parser.add_argument("--strength", type=float, default=0.6, help="Strength of the sdeedit")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for image generation")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for Stable Diffusion")
    parser.add_argument("--sd_model_id", type=str, default="stable-diffusion-2-1-base", help="Stable Diffusion model ID")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--save_base", type=str, default="./output", help="Output directory for generated images")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on (e.g., 'cuda:0')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_anime_vae", action="store_true", help="Enable Anime VAE for image generation")
    parser.add_argument("--model_base_path", type=str, default="./diffusion_model", help="Output directory for generated images")

    return parser.parse_args()


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main function to execute the image generation process."""
    args = parse_args()

    # Set random seed to ensure reproducibility
    set_random_seed(args.seed)

    # Check if the source image path exists
    if not os.path.exists(args.src_img_path):
        raise FileNotFoundError(f"❌ Source image path does not exist: {args.src_img_path}")

    # Ensure the output directory exists
    os.makedirs(args.save_base, exist_ok=True)

    # Initialize the DomainImageGeneration class and generate images
    data_generation = DomainImageGeneration(
        src_path=args.src_img_path,
        args=args,
        model_name=args.sd_model_id,
        save_path=args.save_base,
        device=args.device,
        use_anime_vae=args.use_anime_vae,
        model_base_path = args.model_base_path
    )

    # Start image generation
    data_generation.image_generation(prompt=args.prompt, strength=args.strength, guidance_scale=args.guidance_scale,num_inference_steps=args.num_inference_steps )


if __name__ == "__main__":
    main()