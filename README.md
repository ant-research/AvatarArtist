<div align="center">
    <img src="docs/AvatarArtist.ico" width="250px">
</div>

<h2 align="center">
    <a href="https://arxiv.org/abs/2503.19906">
        [CVPR 2025] AvatarArtist: Open-Domain 4D Avatarization
    </a>
</h2>

<p align="center">
<img alt="avatarrtist" src="docs/teaser.gif" width="80%">
</p>

[//]: # (<div align="center">)

[//]: # (    <img src="docs/teaser.gif" width="350px">)

[//]: # (</div>)

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update. </h5>

<h5 align="center">
    <a href='https://kumapowerliu.github.io/AvatarArtist'>
        <img src='https://img.shields.io/badge/Project-Page-green'>
    </a> 
    <a href='https://arxiv.org/abs/2503.19906'>
        <img src='https://img.shields.io/badge/Technique-Report-red'>
    </a> 
    <a href='https://huggingface.co/KUMAPOWER/AvatarArtist'>
        <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'>
    </a>
    <a href='https://huggingface.co/spaces/KumaPower/AvatarArtist' target="_blank">
    <img src='https://img.shields.io/badge/🚀 Try%20on%20HuggingFace-blue?logo=huggingface&logoColor=white' alt='Hugging Face Demo'>
    </a>
    <a href="https://github.com/ant-research/AvatarArtist">
    <img src="https://img.shields.io/github/stars/ant-research/AvatarArtist?style=social" alt="GitHub stars">
    </a>
</h5>

<div align="center">
    This repository contains the official implementation of AvatarArtist, a method for generating 4D avatars from a single image.
</div>

<br>

<details open>
    <summary>💡 We also have other avatar projects that may interest you ✨.</summary>
    <p>

> **[HeadArtist: Text-conditioned 3D Head Generation with Self Score Distillation, SIGGRAPH 2024](https://arxiv.org/abs/2312.07539)**  
> Hongyu Liu, Xuan Wang, Ziyu Wan, etc.  
> <span>
> <a href='https://github.com/ant-research/HeadArtist'><img src='https://img.shields.io/badge/-Github-black?logo=github'></a>
> <a href='https://kumapowerliu.github.io/HeadArtist'><img src='https://img.shields.io/badge/Project-Page-green'></a>
> <a href='https://arxiv.org/abs/2312.07539'><img src='https://img.shields.io/badge/Arxiv-2312.07539-b31b1b.svg?logo=arXiv'></a>
> </span>

> **[Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation, SIGGRAPH Asia 2024](https://arxiv.org/abs/2406.01900)**  
> Yue Ma, Hongyu Liu, Hongfa Wang, etc.  
> <span><a href='https://github.com/mayuelala/FollowYourEmoji'><img src='https://img.shields.io/badge/-Github-black?logo=github'></a>
> <a href='https://follow-your-emoji.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
> <a href='https://arxiv.org/abs/2406.01900'><img src='https://img.shields.io/badge/Arxiv-2406.01900-b31b1b.svg?logo=arXiv'></a></span>

</details>

## 🚨 News
- [04/02/2025] online demo released! [Try it!!!!](https://huggingface.co/spaces/KumaPower/AvatarArtist)🔥🔥🔥🔥
- [03/30/2025] Gradio demo released!
- [03/26/2025] Inference Code and pretrained models released!

## ⚙️ Setup

### Environment

```bash
git clone --depth=1 https://github.com/ant-research/AvatarArtist 
cd AvatarArtist
conda create -n avatarartist python=3.9.0
conda activate avatarartist
pip install -r requirements.txt
```

### Download Weights

The weights are available at [🤗HuggingFace](https://huggingface.co/KumaPower/AvatarArtist), you can download it with the following commands. Please move the required files into the `pretrained_model` directory:

```bash
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type model \
KUMAPOWER/AvatarArtist \
--local-dir pretrained_model
```

## 🚀 Demo

Welcome to our demo! 🎉  
This project provides an easy-to-use interface to explore and experience our core features.

### 🔧 Quick Start

After installing the necessary dependencies, simply run the following command to launch the demo:

```python
python3 app.py
```

## 🤗 Usage

### Inference
<div align="center">
  <img src="docs/avatarartist_infer_overview.gif"  width="850px" />
  <p>
    <em>
    Our approach consists of two steps during the inference process. First, the DiT model generates a 4D representation based on the input image. Then, our Motion-Aware Cross-Domain Renderer takes this 4D representation as input and, guided by both the input image and driving signals, renders it into the final target image.
    </em>
  </p>
</div>
 
This is an example of inference using the demo data. The images used in this example are sourced from https://civitai.com/.  
```python
python3 inference.py \
    --img_file './demo_data/source_img/img_from_web/images512x512/final_ipimgs' \
    --input_img_fvid './demo_data/source_img/img_from_web/coeffs/final_ipimgs' \
    --input_img_motion './demo_data/source_img/img_from_web/motions/final_ipimgs' \
    --video_name 'Obama' \
    --target_path './demo_data/target_video/data_obama'
    # --use_demo_cam (create a video like the teaser using predefined camera parameters)
```

This is an example of performing inference using the model. The images used in this example are diverse-domain images generated by a diffusion model, as described in our paper. You can use the --select_img option to specify a particular input image.
```python
python3 inference.py \
    --img_file './demo_data/source_img/img_generate_different_domain/images512x512/demo_imgs' \
    --input_img_fvid './demo_data/img_generate_different_domain/coeffs/demo_imgs' \
    --input_img_motion './demo_data/source_img/img_generate_different_domain/motions/demo_imgs' \
    --video_name "Obama" \
    --target_path './demo_data/target_video/data_obama' \
    --select_img 'your_selected_image.png in img_file'
``` 



### Custom Data Processing

We provide a set of scripts to process input images and videos for use with our model. These scripts ensure that the data is properly formatted and preprocessed, making it compatible with our inference pipeline. You can use them to prepare your own custom data for generating results with our model.  

Please refer to [this guide](https://github.com/ant-research/AvatarArtist/tree/main/data_process) to learn how to obtain the inference data. You can also check the [demo data](https://github.com/ant-research/HeadArtist/tree/main/demo_data) for reference. The data structure is shown below.  

The files in the `"dataset"` folder serve as the final input to the model, while the other files are intermediate outputs from the data processing pipeline：

```
📦 datasets/
├── 📂 dataset/
│   ├── 📂 coeffs/
│   ├── 📂 images512x512/
│   ├── 📂 uvRender256x256/
│   ├── 📂 orthRender256x256_face_eye/
│   ├── 📂 motions/
├── 📂 crop_fv_tracking/
├── 📂 realign_detections/
├── 📂 realign_detections/
├── 📂 realign/
├── 📂 raw_detection/
├── 📂 align_3d_landmark/
├── 📂 raw_frames/
```

### Different domain's input images generation

We provide a set of scripts to transfer the realistic domain's portrait to the other domain. Please refer to [this guide](https://github.com/ant-research/AvatarArtist/tree/main/different_domain_imge_gen). 



## **📋 To-Do List**
### **Pending Tasks**
- [ ] Release training code

---

### **✅ Completed Tasks**
- [x] Gradio demo
- [x] Release inference code  
- [x] Release data processing tools  
- [x] Release the pipeline to generate input for different domains

## 👍 Credits  

We sincerely appreciate the contributions of the following open-source projects, which have significantly influenced our work:  

- **DiT** builds upon [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha).  
- **VAE** is based on [LVDM](https://github.com/YingqingHe/LVDM).  
- **Motion-aware rendering** is inspired by [Portrait4D](https://github.com/YuDeng/Portrait-4D).  
- **4D representation** in our paper is proposed in [Next3D](https://github.com/MrTornado24/Next3D) and [Next3D++](https://github.com/XChenZ/invertAvatar).  
- We referenced [DATID3D](https://github.com/gwang-kim/DATID-3D) for domain-specific prompts.  

## 🔒 License

* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## ✏️ Citation
If you make use of our work, please cite our paper.
```bibtex
@article{liu2025avatarartist,
  title={AvatarArtist: Open-Domain 4D Avatarization},
  author={Hongyu Liu, Xuan Wang, Ziyu Wan, Yue Ma, Jingye Chen, Yanbo Fan, Yujun Shen, Yibing Song, Qifeng Chen },
  booktitle={CVPR},
  year={2025}
}
```

