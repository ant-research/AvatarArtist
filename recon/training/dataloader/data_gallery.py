# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
from unittest import skip
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2

try:
    import pyspng
except ImportError:
    pyspng = None

mouth_idx = list(range(22, 52))


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,  # Name of the dataset.
                 raw_shape,  # Shape of the raw image data (NCHW).
                 max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=True,  # Enable conditioning labels? False = label dimension is zero.
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 load_obj=True,
                 return_name=False,
                 random_seed=0,  # Random seed to use when applying max_size.
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self.load_obj = load_obj
        self.return_name = return_name

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_verts_ply(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_geo(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
        # assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8

        label_cam = self.get_label(idx)
        mesh_cond = self.get_vert(self._raw_idx[idx])
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
            if self._use_labels:
                label_1 = label_cam[0:25]
                label_2 = label_cam[25:]
                assert label_1.shape == (25,)
                assert label_2.shape == (25,)

                label_1[[1, 2, 3, 4, 8]] *= -1
                label_2[[1, 2, 3, 4, 8]] *= -1
                label_cam = np.concatenate([label_1, label_2], axis=-1)
        if self.return_name:
            return self._image_fnames[self._raw_idx[idx]], image.copy(), label_cam, mesh_cond
        else:
            return image.copy(), label_cam, mesh_cond

    def load_random_data(self):
        gen_cond_sample_idx = [np.random.randint(self.__len__()) for _ in range(self.random_sample_num)]
        all_gen_c = np.stack([self.get_label(i) for i in gen_cond_sample_idx])
        all_gen_v = [self.get_vert(i) for i in gen_cond_sample_idx]
        all_gt_img = np.stack([self.get_image(i).astype(np.float32) / 127.5 - 1 for i in gen_cond_sample_idx])

        return all_gen_c, all_gen_v, all_gt_img

    def get_by_name(self, name):
        raw_idx = self._image_fnames.index(name)
        image = self._load_raw_image(raw_idx, resolution=self.resolution)
        mesh_cond = self.get_vert(raw_idx)
        label = self._get_raw_labels()[raw_idx]
        cam = self._raw_cams[raw_idx]
        label_cam = np.concatenate([label, cam], axis=-1)
        return image.copy(), label_cam, mesh_cond

    def get_label(self, idx):
        raise NotImplementedError

    def get_vert(self, idx):
        raise NotImplementedError

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    # @property
    # def gen_label_dim(self):
    #         return 25 # 25 for camera params only

    @property
    def has_labels(self):
        return any(x != 0 for x in [25])

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 data_label_path,
                 label_file_vfhq,
                 label_file_ffhq,
                 mesh_path_ffhq,
                 motion_path_ffhq,
                 mesh_path_vfhq,
                 motion_path_vfhq,
                 mesh_path_ffhq_label,
                 mesh_path_vfhq_label,
                 resolution=512,
                 static=False,
                 **super_kwargs,
                 ):
        self._path = path
        self._mesh_ffhq = mesh_path_ffhq
        self._motion_ffhq = np.load(motion_path_ffhq)
        # self._label_ffhq = np.load(label_file_ffhq)
        self._mesh_vfhq = mesh_path_vfhq
        self._motion_vfhq = np.load(motion_path_vfhq, allow_pickle=True)
        # self._label_vfhq = np.load(label_file_vfhq)

        self._data_label_path = data_label_path
        self.data_json = json.loads(open(data_label_path).read())
        PIL.Image.init()
        self._raw_cams_ffhq = json.loads(open(label_file_ffhq).read())['labels']
        self.mesh_path_ffhq_label = json.loads(open(mesh_path_ffhq_label).read())
        self.mesh_path_vfhq_label = json.loads(open(mesh_path_vfhq_label).read())
        # self._image_fnames = list(dict(json.loads(open(self._raw_cams_ffhq).read())['labels']).keys())
        # self._raw_cams_vfhq = self._load_raw_label(self._mesh_path_vfhq, 'labels')

        self.all_input_ids = list(self.data_json.keys())

        name = os.path.splitext(os.path.basename(self._path))[0]
        # raw_shape = [len(self._image_fnames)] + [3, resolution, resolution]
        raw_shape = [len(self.all_input_ids)] + list([3, 512, 512])
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return len(self.all_input_ids)

    # phase_real_z, phase_real_latent, phase_real_c_1_d, phase_real_c_2_d, phase_real_c_3_d, phase_real_v_1_d, phase_real_v_2_d, phase_real_v_s, motion_1, motion_2, motion_ffhq, model_list

    def __getitem__(self, idx):

        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')


    def getdata(self, idx):
        base_dir = self.all_input_ids[idx]
        model_name = self.data_json[base_dir]
        latent_dir = os.path.join(self._path, base_dir, '0.pt')
        latent_dit_dir = os.path.join(self._path, base_dir, '0_dit.pt')
        phase_real_z = torch.load(latent_dir).float()
        phase_real_latent = torch.load(latent_dit_dir).float()

        phase_real_c_1_d, phase_real_c_2_d, phase_real_c_3_d = self.get_label(idx)
        motion_ffhq, phase_real_v_s  = self.get_ffhq_motion()
        motion_1, motion_2, phase_real_v_1_d, phase_real_v_2_d = self.get_vfhq_motion()


        return {
            "model_name": model_name,
            "phase_real_z": phase_real_z,
            "phase_real_latent": phase_real_latent,
            "phase_real_c_1_d": phase_real_c_1_d.unsqueeze(0),
            "phase_real_c_2_d": phase_real_c_2_d.unsqueeze(0),
            "phase_real_c_3_d": phase_real_c_3_d.unsqueeze(0),
            "phase_real_v_s": phase_real_v_s.unsqueeze(0),
            "motion_ffhq": motion_ffhq.unsqueeze(0),
            "motion_1": motion_1.unsqueeze(0),
            "motion_2": motion_2.unsqueeze(0),
            "phase_real_v_1_d": phase_real_v_1_d.unsqueeze(0),
            "phase_real_v_2_d": phase_real_v_2_d.unsqueeze(0)
        }
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname, path=None):
        if not path:
            path = self._path
        if self._type == 'dir':
            return open(os.path.join(path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_label(self, json_path, sub_key=None):
        with open(json_path, 'rb') as f:
            labels = json.load(f)
        if sub_key is not None: labels = labels[sub_key]
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        return np.array(labels).astype(np.float32)

    def _load_raw_image_core(self, fname, path=None, resolution=None):
        with self._open_file(fname, path) as f:
            image = PIL.Image.open(f)
            if resolution:
                image = image.resize((resolution, resolution))
            image = np.array(image)  # .astype(np.float32)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_motion(self, raw_idx, resolution=None):
        fname = self._image_fnames[raw_idx]
        image = self._load_raw_image_core(fname, resolution=resolution)  # [C, H, W]
        return image

    def _load_vfhq_raw_labels(self):
        labels = self._load_raw_label(os.path.join(self._path, self.label_file), 'labels')
        return labels

    def _load_ffhq_raw_labels(self):
        labels = self._load_raw_label(os.path.join(self._path, self.label_file), 'labels')
        return labels

    def get_vert(self, vert_dir):
        uvcoords_image = np.load(os.path.join(vert_dir))[...,
                         :3]  # [HW3] 前两维date range(-1, 1)，第三维是face_mask，最后一维是render_mask
        uvcoords_image[..., -1][uvcoords_image[..., -1] < 0.5] = 0;
        uvcoords_image[..., -1][uvcoords_image[..., -1] >= 0.5] = 1
        # out = {'uvcoords_image': torch.tensor(uvcoords_image.copy()).float()}
        return torch.tensor(uvcoords_image.copy()).float()

    def load_random_data(self):
        gen_cond_sample_idx = [np.random.randint(self.__len__()) for _ in range(self.random_sample_num)]
        all_gen_c = np.stack([self.get_label(i) for i in gen_cond_sample_idx])
        all_gen_v = [self.get_vert(i) for i in gen_cond_sample_idx]
        all_gt_img = np.stack([self.get_image(i).astype(np.float32) / 127.5 - 1 for i in gen_cond_sample_idx])

        return all_gen_c, all_gen_v, all_gt_img

    def get_label(self, idx):
        # all_nums = self._raw_cams_ffhq
        gen_cond_sample_idx = [np.random.randint(len(self._raw_cams_ffhq)) for _ in range(3)]
        cam = [self._raw_cams_ffhq[i][1] for i in gen_cond_sample_idx]
        return torch.tensor(np.array(cam[0]).astype(np.float32)).float(), torch.tensor(np.array(cam[1]).astype(np.float32)).float(), torch.tensor(np.array(cam[2]).astype(np.float32)).float()

    def get_ffhq_motion(self):
        assert len(self.mesh_path_ffhq_label) == self._motion_ffhq.shape[0]
        gen_cond_sample_idx = np.random.randint(self._motion_ffhq.shape[0])
        motion = self._motion_ffhq[gen_cond_sample_idx]
        vert_dir = os.path.join(self._mesh_ffhq, self.mesh_path_ffhq_label[gen_cond_sample_idx])
        vert = self.get_vert(vert_dir)


        return torch.tensor(motion).float(), vert

    def get_vfhq_motion(self):
        assert len(self.mesh_path_vfhq_label) == self._motion_vfhq.shape[0]
        gen_cond_sample_idx_row = np.random.randint(self._motion_vfhq.shape[0])
        motions = self._motion_vfhq[gen_cond_sample_idx_row]
        verts = self.mesh_path_vfhq_label[gen_cond_sample_idx_row]
        assert motions.shape[0] == len(verts)
        # print('motions.shape', motions.shape)
        # print('motions.shape[0]', motions.shape[0])

        gen_cond_sample_idx_col = np.random.randint(motions.shape[0], size=2)
        motions_1 = motions[gen_cond_sample_idx_col[0]]
        motions_2 = motions[gen_cond_sample_idx_col[1]]
        verts_1_dir = os.path.join(self._mesh_vfhq, verts[gen_cond_sample_idx_col[0]])
        verts_2_dir = os.path.join(self._mesh_vfhq, verts[gen_cond_sample_idx_col[1]])
        verts_1 = self.get_vert(verts_1_dir)
        verts_2 = self.get_vert(verts_2_dir)
        return torch.tensor(motions_1).float(), torch.tensor(motions_2).float(), verts_1, verts_2
