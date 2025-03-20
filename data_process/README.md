# <u>Data Preprocessing Pipeline</u> by *AvatarArtist* 
This repo describes how to process your own data for using our model.

## 🎉 Overview

<div align=center>
<img src="data_process_pipe.png">
</div>

## ⚙️ Requirements and Installation

We recommend the requirements as follows.

### Environment

```bash
git clone --depth=1 https://github.com/ant-research/AvatarArtist 
cd AvatarArtist
conda create -n consisid_data python=3.9.0
conda activate avatarartist
pip install -r requirements.txt
```

### Download Weight

The weights are available at [🤗HuggingFace](https://huggingface.co/BestWishYsh/ConsisID-preview), you can download it with the following commands.

```bash
# way 1
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
cd util
python download_weights.py

# way 2
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type model \
BestWishYsh/ConsisID-preview \
--local-dir ckpts

# way 3
modelscope download --model \
BestWishYSH/ConsisID-preview \
--local-dir pretr

# way 4
git lfs install
git clone https://www.wisemodel.cn/SHYuanBest/ConsisID-Preview.git
```

Once ready, the weights will be organized in this format:

```
📦 ConsisiID/
├── 📂 pretrained_model/
│   ├── 📂 data_process_model/
│       ├── 📂 pdfgc
│       ├── 📂 landmark3d
│       ├── 📂 hrnet_w18_wflw
│       ├── 📂 facerecon
│       ├── 📂 bfm2flame_mapper
│       ├── 📂 facedetect
```

## 🗝️ Usage


Process the target video to obtain the target pose motion and mesh.

```bash
python3 input_img_align_extract_ldm.py --input_dir ./demo_data/hongyu_2.mp4 --is_video --save_dir ./demo_data/data_process_out
```

Process the image to extract the source image.


```bash
python3 input_img_align_extract_ldm.py --input_dir ./demo_data/ip_imgs --is_img --save_dir ./demo_data/data_process_out 
```
Our code supports step-by-step data processing. For example, if your images are already aligned, you can proceed directly to the next step.

```bash
python3 input_img_align_extract_ldm.py --input_dir ./demo_data/ip_imgs --is_img --save_dir ./demo_data/data_process_out  --already_align
```
## 👍 Credits

- This code builds on [Portrait4D](https://github.com/YuDeng/Portrait-4D) and [InvertAvatar](https://github.com/XChenZ/invertAvatar). We have integrated and organized their data processing code. Thanks for open-sourcing!
 