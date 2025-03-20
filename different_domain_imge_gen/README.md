This script can help you transfer the realistic domain's portrait to the other domain.

 
## ⚙️ Requirements and Installation

We test with the diffusers==0.20.1

 

### Download Weight

You should download the [mediapipe controlnet](CrucibleAI/ControlNetMediaPipeFace) and [SD21](stabilityai/stable-diffusion-2-1-base).

You can also download some third-party models form huggingface, for example [3Danimation](https://huggingface.co/Yntec/3Danimation/tree/main)

```bash
python3 input_image_gen.py --model_base_path 'the dir of your download models' --prompt "a 3D render of a face in Pixar style"
```
## Domain Types and Corresponding Prompts

As mentioned in our main paper, we used 28 domain images during training, including the original realistic domain. We categorize our domains into two types.

### Type 1: Stable Diffusion 2.1 Generated Domains

The first type uses the official Stable Diffusion 2.1 model as the generative model. For this type, the text prompts used are shown in the table below. We generate images in 20 different domain styles, with 6,000 images per domain.

#### Table 1: Full-text prompts corresponding to each domain

| **Concise Name of Domain** | **Full text prompt** |
|----------------------------|----------------------|
| Pixar                      | a 3D render of a face in Pixar style |
| Lego                       | a 3D render of a head of a Lego man 3D model |
| Greek statue               | a FHD photo of a white Greek statue |
| Elf                        | a FHD photo of a face of a beautiful elf with silver hair in live action movie |
| Zombie                     | a FHD photo of a face of a zombie |
| Tekken                     | a 3D render of a Tekken game character |
| Devil                      | a FHD photo of a face of a devil in fantasy movie |
| Steampunk                  | Steampunk style portrait, mechanical, brass and copper tones |
| Mario                      | a 3D render of a face of Super Mario |
| Orc                        | a FHD photo of a face of an orc in fantasy movie |
| Masque                     | a FHD photo of a face of a person in masquerade |
| Skeleton                   | a FHD photo of a face of a skeleton in fantasy movie |
| Peking Opera               | a FHD photo of a face of a character in Peking opera with heavy make-up |
| Yoda                       | a FHD photo of a face of Yoda in Star Wars |
| Hobbit                     | a FHD photo of a face of a Hobbit in Lord of the Rings |
| Stained glass              | Stained glass style, portrait, beautiful, translucent |
| Graffiti                   | Graffiti style portrait, street art, vibrant, urban, detailed, tag |
| Pixel-art                  | Pixel art style portrait, low res, blocky, pixel art style |
| Retro                      | Retro game art style portrait, vibrant colors |
| Ink                        | A portrait in ink style, black and white image |

### Type 2: Civitai Third-Party Models

The second type utilizes third-party models from Civitai as the generative models, where each model corresponds to a specific style. For these models, the same text prompt is used across all models, and we set the prompt as:

> **"masterpieces, portrait, high-quality"**

#### Table 2: Models used for each domain

| **Concise Name of Domain** | **Model Name** |
|----------------------------|---------------|
| 3D-Animation               | 3D Animation Diffusion-V1.0 |
| Toon                       | ToonYou-Beta6 |
| AAM                        | AAM Anime Mix |
| Counterfeit                | Counterfeit-V3.0 |
| Pencil                     | Pencil Sketch |
| Lyriel                     | Lyriel-V1.6 |
| XXM                        | XXMix9realistic |

All models were sourced from **Civitai**, an AI-Generated Content (AIGC) social platform.