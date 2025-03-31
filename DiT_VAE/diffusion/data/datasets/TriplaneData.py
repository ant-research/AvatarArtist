import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor
from DiT_VAE.diffusion.data.builder import DATASETS
from omegaconf import OmegaConf
from torchvision import transforms
from transformers import CLIPImageProcessor
import io
import zipfile
import numpy
import json


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


@DATASETS.register_module()
class TriplaneData(Dataset):
    def __init__(self,
                 data_base_dir,
                 model_names,
                 data_json_file,
                 dino_path,
                 i_drop_rate=0.1,
                 image_size=256,
                 **kwargs):
        self.dict_data_image = json.load(open(data_json_file))  # {'image_name': pose}
        self.data_base_dir = data_base_dir
        self.dino_img_processor = AutoImageProcessor.from_pretrained(dino_path)
        self.size = image_size
        self.data_list = list(self.dict_data_image.keys())
        self.zip_file_dict = {}
        config_gan_model = OmegaConf.load(model_names)
        all_models = config_gan_model['gan_models'].keys()
        for model_name in all_models:
            zipfile_path = os.path.join(self.data_base_dir, model_name + '.zip')
            zipfile_load = zipfile.ZipFile(zipfile_path)
            self.zip_file_dict[model_name] = zipfile_load
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        self.i_drop_rate = i_drop_rate


    def getdata(self, idx):

        data_name = self.data_list[idx]
        data_model_name = self.dict_data_image[data_name]['model_name']
        zipfile_loaded = self.zip_file_dict[data_model_name]
        # zipfile_path = os.path.join(self.data_base_dir, data_model_name)
        # zipfile_loaded = zipfile.ZipFile(zipfile_path)
        with zipfile_loaded.open(self.dict_data_image[data_name]['z_dir'], 'r') as f:
            buffer = io.BytesIO(f.read())
            data_z = torch.load(buffer)

        with zipfile_loaded.open(self.dict_data_image[data_name]['vert_dir'], 'r') as f:
            buffer = io.BytesIO(f.read())
            data_vert = torch.load(buffer)

        with zipfile_loaded.open(self.dict_data_image[data_name]['img_dir'], 'r') as f:
            raw_image = to_rgb_image(Image.open(f))
            dino_img = self.dino_img_processor(images=raw_image, return_tensors="pt").pixel_values
            image = self.transform(raw_image.convert("RGB"))
            clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        return {
            "raw_image": raw_image,
            "dino_img": dino_img,
            "image": image,
            "clip_image": clip_image.clone(),
            "data_z": data_z,
            "data_vert": data_vert,
            "data_model_name": data_model_name,
            "drop_image_embed": drop_image_embed,

        }

        #
        # img_path = self.img_samples[index]
        # npz_path = self.txt_feat_samples[index]
        # npy_path = self.vae_feat_samples[index]
        # prompt = self.prompt_samples[index]
        # data_info = {
        #     'img_hw': torch.tensor([torch.tensor(self.resolution), torch.tensor(self.resolution)], dtype=torch.float32),
        #     'aspect_ratio': torch.tensor(1.)
        # }
        #
        # img = self.loader(npy_path) if self.load_vae_feat else self.loader(img_path)
        # txt_info = np.load(npz_path)
        # txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
        # attention_mask = torch.ones(1, 1, txt_fea.shape[1])     # 1x1xT
        # if 'attention_mask' in txt_info.keys():
        #     attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
        # if txt_fea.shape[1] != self.max_lenth:
        #     txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_lenth-txt_fea.shape[1], 1)], dim=1)
        #     attention_mask = torch.cat([attention_mask, torch.zeros(1, 1, self.max_lenth-attention_mask.shape[-1])], dim=-1)
        #
        # if self.transform:
        #     img = self.transform(img)
        #
        # data_info['prompt'] = prompt
        # return img, txt_fea, attention_mask, data_info

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')


    def __len__(self):
        return len(self.data_list)

    def __getattr__(self, name):
        if name == "set_epoch":
            return lambda epoch: None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
