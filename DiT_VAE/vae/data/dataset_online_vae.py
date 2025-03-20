import os

import numpy
import json
import zipfile
import torch
from PIL import Image
# from transformers import CLIPImageProcessor
from torch.utils.data import Dataset
import io
from omegaconf import OmegaConf
import numpy as np
# from torchvision import transforms
# from einops import rearrange
# import random
# import os
# from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
# import time
# import io
# import array
# import numpy as np
#
# from training.triplane import TriPlaneGenerator


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



# image(contain style),z,pose,text
class TriplaneDataset(Dataset):
    # image, triplane, ref_feature
    def __init__(self, json_file, data_base_dir, model_names):
        super().__init__()
        self.dict_data_image = json.load(open(json_file))  # {'image_name': pose}
        self.data_base_dir = data_base_dir
        self.data_list = list(self.dict_data_image.keys())
        self.zip_file_dict = {}
        config_gan_model = OmegaConf.load(model_names)
        all_models = config_gan_model['gan_models'].keys()
        for model_name in all_models:
            zipfile_path = os.path.join(self.data_base_dir, model_name+'.zip')
            zipfile_load = zipfile.ZipFile(zipfile_path)
            self.zip_file_dict[model_name] = zipfile_load

    def getdata(self, idx):
        # need z and expression and model name
        # image:"seed0035.png"
        # data_each_dict = {
        #     'vert_dir': vert_dir,
        #     'z_dir': z_dir,
        #     'pose_dir': pose_dir,
        #     'img_dir': img_dir,
        #     'model_name': model_name
        # }
        data_name = self.data_list[idx]
        data_model_name = self.dict_data_image[data_name]['model_name']
        zipfile_loaded = self.zip_file_dict[data_model_name]
        # zipfile_path = os.path.join(self.data_base_dir, data_model_name)
        # zipfile_loaded = zipfile.ZipFile(zipfile_path)
        with zipfile_loaded.open(self.dict_data_image[data_name]['z_dir'], 'r') as f:
            buffer = io.BytesIO(f.read())
            data_z = torch.load(buffer)
        buffer.close()
        f.close()
        with zipfile_loaded.open(self.dict_data_image[data_name]['vert_dir'], 'r') as ff:
            buffer_v = io.BytesIO(ff.read())
            data_vert = torch.load(buffer_v)
        buffer_v.close()
        ff.close()
        #     raw_image = to_rgb_image(Image.open(f))
        #
        # data_model_name = self.dict_data_image[data_name]['model_name']
        # data_z_dir = os.path.join(self.data_base_dir, data_model_name, self.dict_data_image[data_name]['z_dir'])
        # data_vert_dir = os.path.join(self.data_base_dir, data_model_name, self.dict_data_image[data_name]['vert_dir'])
        # data_z = torch.load(data_z_dir)
        # data_vert = torch.load(data_vert_dir)

        return {
            "data_z": data_z,
            "data_vert": data_vert,
            "data_model_name": data_model_name
        }

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

# for zip files
