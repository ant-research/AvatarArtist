# Main training loop of Portrait4D, modified from EG3D: https://github.com/NVlabs/eg3d

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import random
import torch
import torch.nn as nn
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from omegaconf import OmegaConf
from rendering import RenderingClass
from einops import rearrange

import legacy
from recon.models.lpips.lpips import LPIPS
from recon.models.id.id_loss import IDLoss
from recon.training.generator.triplane_v20_original import TriPlaneGenerator
from headGallery_model.models.triplane_vae import AutoencoderKL as AutoencoderKLTriplane
# from training.triplane import PartTriPlaneGeneratorDeform

# ----------------------------------------------------------------------------

# def setup_snapshot_image_grid(all_shape_params, all_exp_params, all_pose_params, all_eye_pose_params, all_c,
#                               static_dataset=False, random_seed=1):
#     gw = 7
#     gh = 4
#
#     grid_indices = np.random.RandomState(random_seed).randint(0, len(all_shape_params), size=(gw * gh))
#
#     shape_params = all_shape_params[grid_indices]
#     shape_params = np.tile(np.expand_dims(shape_params, 1), (1, 3, 1)).reshape(gw * gh, 3, -1)
#
#     grid_indices2 = np.random.RandomState(random_seed + 1).randint(0, len(all_exp_params), size=(gw * gh))
#     mot_indices = np.random.RandomState(random_seed + 2).randint(0, len(all_exp_params[0]), size=(gw * gh, 2))
#
#     exp_params = all_exp_params[grid_indices2]
#     exp_params = np.stack([exp_params[i, mot_indices[i]] for i in range(len(mot_indices))])  # (gw * gh, 2, dim)
#
#     pose_params = all_pose_params[grid_indices2]
#     pose_params = np.stack([pose_params[i, mot_indices[i]] for i in range(len(mot_indices))])  # (gw * gh, 2, dim)
#
#     eye_pose_params = all_eye_pose_params[grid_indices2]
#     eye_pose_params = np.stack(
#         [eye_pose_params[i, mot_indices[i]] for i in range(len(mot_indices))])  # (gw * gh, 2, dim)
#
#     if not static_dataset:
#         # for dynamic
#         exp_params = np.concatenate([exp_params, exp_params[:, -1:]], axis=1).reshape(gw * gh, 3,
#                                                                                       -1)  # (gw * gh, 3, dim)
#         pose_params = np.concatenate([pose_params, pose_params[:, -1:]], axis=1).reshape(gw * gh, 3, -1)
#         eye_pose_params = np.concatenate([eye_pose_params, eye_pose_params[:, -1:]], axis=1).reshape(gw * gh, 3, -1)
#     else:
#         # for static
#         exp_params = np.concatenate([exp_params[:, :1], exp_params[:, :1], exp_params[:, :1]], axis=1).reshape(gw * gh,
#                                                                                                                3,
#                                                                                                                -1)  # (gw * gh, 3, dim)
#         pose_params = np.concatenate([pose_params[:, :1], pose_params[:, :1], pose_params[:, :1]], axis=1).reshape(
#             gw * gh, 3, -1)
#         eye_pose_params = np.concatenate([eye_pose_params[:, :1], eye_pose_params[:, :1], eye_pose_params[:, :1]],
#                                          axis=1).reshape(gw * gh, 3, -1)
#
#     grid_indices3 = np.random.randint(0, len(all_c), size=(gw * gh * 3))
#     c = all_c[grid_indices3].reshape(gw * gh, 3, -1)
#
#     return (gw, gh), shape_params, exp_params, pose_params, eye_pose_params, c


# def setup_snapshot_image_grid(all_shape_params, all_exp_params, all_pose_params, all_eye_pose_params, all_c,
#                               static_dataset=False, random_seed=1):
#     gw = 7
#     gh = 4
#
#     grid_indices = np.random.RandomState(random_seed).randint(0, len(all_shape_params), size=(gw * gh))
#
#     shape_params = all_shape_params[grid_indices]
#     shape_params = np.tile(np.expand_dims(shape_params, 1), (1, 3, 1)).reshape(gw * gh, 3, -1)
#
#     grid_indices2 = np.random.RandomState(random_seed + 1).randint(0, len(all_exp_params), size=(gw * gh))
#     mot_indices = np.random.RandomState(random_seed + 2).randint(0, len(all_exp_params[0]), size=(gw * gh, 2))
#
#     exp_params = all_exp_params[grid_indices2]
#     exp_params = np.stack([exp_params[i, mot_indices[i]] for i in range(len(mot_indices))])  # (gw * gh, 2, dim)
#
#     pose_params = all_pose_params[grid_indices2]
#     pose_params = np.stack([pose_params[i, mot_indices[i]] for i in range(len(mot_indices))])  # (gw * gh, 2, dim)
#
#     eye_pose_params = all_eye_pose_params[grid_indices2]
#     eye_pose_params = np.stack(
#         [eye_pose_params[i, mot_indices[i]] for i in range(len(mot_indices))])  # (gw * gh, 2, dim)
#
#     if not static_dataset:
#         # for dynamic
#         exp_params = np.concatenate([exp_params, exp_params[:, -1:]], axis=1).reshape(gw * gh, 3,
#                                                                                       -1)  # (gw * gh, 3, dim)
#         pose_params = np.concatenate([pose_params, pose_params[:, -1:]], axis=1).reshape(gw * gh, 3, -1)
#         eye_pose_params = np.concatenate([eye_pose_params, eye_pose_params[:, -1:]], axis=1).reshape(gw * gh, 3, -1)
#     else:
#         # for static
#         exp_params = np.concatenate([exp_params[:, :1], exp_params[:, :1], exp_params[:, :1]], axis=1).reshape(gw * gh,
#                                                                                                                3,
#                                                                                                                -1)  # (gw * gh, 3, dim)
#         pose_params = np.concatenate([pose_params[:, :1], pose_params[:, :1], pose_params[:, :1]], axis=1).reshape(
#             gw * gh, 3, -1)
#         eye_pose_params = np.concatenate([eye_pose_params[:, :1], eye_pose_params[:, :1], eye_pose_params[:, :1]],
#                                          axis=1).reshape(gw * gh, 3, -1)
#
#     grid_indices3 = np.random.randint(0, len(all_c), size=(gw * gh * 3))
#     c = all_c[grid_indices3].reshape(gw * gh, 3, -1)
#
#     return (gw, gh), shape_params, exp_params, pose_params, eye_pose_params, c

@torch.no_grad()
def setup_snapshot_image_grid_gallery(val_set, vae_triplane, vae_std, vae_mean, render, device):
    gw = 2
    gh = 2
    phase_real_z_val, phase_real_latent_val, phase_real_c_1_d_val, phase_real_c_2_d_val, phase_real_c_3_d_val, phase_real_v_1_d_val, phase_real_v_2_d_val, phase_real_v_s_val, motion_1_val, motion_2_val, motion_ffhq_val, model_list_val = next(
        val_set)
    phase_real_z_val, phase_real_latent_val, phase_real_c_1_d_val, phase_real_c_2_d_val, phase_real_c_3_d_val, phase_real_v_1_d_val, phase_real_v_2_d_val, phase_real_v_s_val, motion_1_val, motion_2_val, motion_ffhq_val, model_list_val = \
    phase_real_z_val.to(device), phase_real_latent_val.to(device), phase_real_c_1_d_val.to(device), phase_real_c_2_d_val.to(device), phase_real_c_3_d_val.to(device), phase_real_v_1_d_val.to(device), phase_real_v_2_d_val.to(device), \
        phase_real_v_s_val.to(device), motion_1_val.to(device), motion_2_val.to(device), motion_ffhq_val.to(device), model_list_val
    batchsize = phase_real_z_val.shape[0]
    cur_z = phase_real_z_val.unsqueeze(1).repeat(1, 3, 1)  # [b,1,512]
    # cur_latent = phase_real_latent.unsqueeze(1).repeat(1, 1, 1, 1, 1)  # [b,3,c, h, w]
    c_1 = phase_real_c_1_d_val.unsqueeze(1)  # input
    c_2 = phase_real_c_2_d_val.unsqueeze(1)  # motion
    c_3 = phase_real_c_3_d_val.unsqueeze(1)  # target
    cur_c = torch.cat([c_1, c_2, c_3], 1)  # from ffhq
    exp_d_1 = phase_real_v_1_d_val.unsqueeze(1)  # motion_1 from vfhq
    exp_d_2 = phase_real_v_2_d_val.unsqueeze(1)  # motion_2 from vfhq

    exp_s = phase_real_v_s_val.unsqueeze(1).repeat(1, 3, 1, 1, 1)  # motion_3 from ffhq for static similar to Portrait4D
    cur_exp_params = torch.cat([exp_d_1, exp_d_2, exp_d_2], dim=1)


    cur_exp_out = torch.cat([cur_exp_params[:batchsize // 2], exp_s[batchsize // 2:]], dim=0)

    model_list_out = [val for val in model_list_val for i in range(3)]
    model_list_out = listfunc(model_list_out, 3)
    assert phase_real_latent_val.shape[0] == batchsize

    cano_tri = vae_triplane.decode(phase_real_latent_val.to(torch.float16))
    cano_tri = cano_tri.float()
    cano_tri = rearrange(cano_tri, "b c f h w -> b f c h w")
    ref_tri = cano_tri * vae_std + vae_mean
    exp_target = cur_exp_out[:, 2]
    # ref_tri_out = render.gen_triplane(ref_tri, exp_target)
    motion_1_out = torch.cat([motion_1_val[:batchsize // 2], motion_ffhq_val[batchsize // 2:]], dim=0)
    motion_2_out = torch.cat([motion_2_val[:batchsize // 2], motion_ffhq_val[batchsize // 2:]], dim=0)
    # ref_img = render.get_img_with_tri(ref_tri, c_3)

    # always half static
    return (gw, gh), cur_z, cur_c, cur_exp_out, motion_1_out, motion_2_out, model_list_out, ref_tri


# ----------------------------------------------------------------------------

def save_image_grid_all(img_app, img_mot, img_recon, img_ref, fname, drange, grid_size):
    lo, hi = drange
    img_app = np.asarray(img_app, dtype=np.float32)
    img_app = (img_app - lo) * (255 / (hi - lo))
    img_app = np.rint(img_app).clip(0, 255).astype(np.uint8)

    img_mot = np.asarray(img_mot, dtype=np.float32)
    img_mot = (img_mot - lo) * (255 / (hi - lo))
    img_mot = np.rint(img_mot).clip(0, 255).astype(np.uint8)

    img_recon = np.asarray(img_recon, dtype=np.float32)
    img_recon = (img_recon - lo) * (255 / (hi - lo))
    img_recon = np.rint(img_recon).clip(0, 255).astype(np.uint8)

    img_ref = np.asarray(img_ref, dtype=np.float32)
    img_ref = (img_ref - lo) * (255 / (hi - lo))
    img_ref = np.rint(img_ref).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img_app.shape

    img = np.concatenate([img_app, img_mot, img_recon, img_ref], -1)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def split_gen(gen, batch_gpu, batch_size, device):
    assert type(gen) == list
    if type(gen[0]) == np.ndarray:
        all_gen = torch.from_numpy(np.stack(gen)).pin_memory().to(device).float()
        all_gen = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen.split(batch_size)]
    elif type(gen[0]) == dict:
        all_gen = [[{} for _ in range(batch_size // batch_gpu)] for _ in range(len(gen) // batch_size)]
        for key in gen[0].keys():
            key_value = torch.from_numpy(np.stack([sub[key] for sub in gen])).pin_memory().to(device).float()
            key_value_split = [phase_gen_c.split(batch_gpu) for phase_gen_c in key_value.split(batch_size)]
            for i in range(len(key_value_split)):
                for j in range(len(key_value_split[i])):
                    all_gen[i][j][key] = key_value_split[i][j]
    else:
        raise NotImplementedError
    return all_gen


def split_gen_new(gen, batch_gpu, batch_size, device):
    if type(gen) == torch.Tensor:
        all_gen = gen.view((gen.shape[0] * gen.shape[1],) + (gen.shape[2:])).pin_memory().to(device)
        all_gen = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen.split(batch_size)]
    elif type(gen[0]) == dict:
        all_gen = [[{} for _ in range(batch_size // batch_gpu)] for _ in
                   range(int(len(gen) * list(gen[0].values())[0].shape[0] // batch_size))]
        for key in gen[0].keys():
            key_value = torch.cat([sub[key] for sub in gen], dim=0).pin_memory().to(device)
            key_value_split = [phase_gen_c.split(batch_gpu) for phase_gen_c in key_value.split(batch_size)]
            for i in range(len(key_value_split)):
                for j in range(len(key_value_split[i])):
                    all_gen[i][j][key] = key_value_split[i][j]
    else:
        raise NotImplementedError
    return all_gen

@torch.no_grad()
# similar to the fetch_random_params
def fetch_dataset(phase_real_z, phase_real_latent, phase_real_c_1_d, phase_real_c_2_d, phase_real_c_3_d,
                  phase_real_v_1_d, phase_real_v_2_d, phase_real_v_s, motion_1, motion_2, motion_ffhq, model_list,
                  vae_triplane, vae_std, vae_mean, render):
    batchsize = phase_real_z.shape[0]
    cur_z = phase_real_z.unsqueeze(1).repeat(1, 3, 1)  # [b,1,512]
    # cur_latent = phase_real_latent.unsqueeze(1).repeat(1, 1, 1, 1, 1)  # [b,3,c, h, w]
    c_1 = phase_real_c_1_d.unsqueeze(1)
    c_2 = phase_real_c_2_d.unsqueeze(1)
    c_3 = phase_real_c_3_d.unsqueeze(1)
    cur_c = torch.cat([c_1, c_2, c_3], 1)  # from ffhq
    exp_d_1 = phase_real_v_1_d.unsqueeze(1)  # motion_1 from vfhq
    exp_d_2 = phase_real_v_2_d.unsqueeze(1)  # motion_2 from vfhq


    exp_s = phase_real_v_s.unsqueeze(1).repeat(1, 3, 1, 1, 1)  # motion_3 from ffhq for static similar to Portrait4D
    cur_exp_params = torch.cat([exp_d_1, exp_d_2, exp_d_2], dim=1)
    cur_exp_out = torch.cat([cur_exp_params[:batchsize // 2], exp_s[batchsize // 2:]], dim=0)
    model_list_out = [val for val in model_list for i in range(3)]
    model_list_out = listfunc(model_list_out, 3)
    assert phase_real_latent.shape[0] == batchsize
    cano_tri = vae_triplane.decode(phase_real_latent.to(torch.float16))
    cano_tri = cano_tri.float()
    cano_tri = rearrange(cano_tri, "b c f h w -> b f c h w")
    ref_tri = cano_tri * vae_std + vae_mean
    exp_target = cur_exp_out[:, 2]
    # ref_tri_out = render.gen_triplane(ref_tri, exp_target)
    motion_1_out = torch.cat([motion_1[:batchsize // 2], motion_ffhq[batchsize // 2:]], dim=0)
    motion_2_out = torch.cat([motion_2[:batchsize // 2], motion_ffhq[batchsize // 2:]], dim=0)
    # always half static
    return cur_z, cur_c, cur_exp_out, motion_1_out, motion_2_out, model_list_out, cano_tri, ref_tri, exp_target


# choose random FLAME parameters for online data synthesis (torch version)


def gan_model(gan_models, device, gan_model_base_dir):
    gan_model_dict = gan_models["gan_models"]
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


def listfunc(listTemp, n):
    out_list = []
    for i in range(0, len(listTemp), n):
        each = listTemp[i:i + n]
        out_list.append(each)
    return out_list





def collate_fn(data):
    model_list = [example["model_name"] for example in data]
    phase_real_z = torch.cat([example["phase_real_z"] for example in data], dim=0)
    phase_real_latent = torch.cat([example["phase_real_latent"] for example in data], dim=0)
    phase_real_c_1_d = torch.cat([example["phase_real_c_1_d"] for example in data], dim=0)
    phase_real_c_2_d = torch.cat([example["phase_real_c_2_d"] for example in data], dim=0)
    phase_real_c_3_d = torch.cat([example["phase_real_c_3_d"] for example in data], dim=0)
    phase_real_v_s = torch.cat([example["phase_real_v_s"] for example in data], dim=0)
    motion_ffhq = torch.cat([example["motion_ffhq"] for example in data], dim=0)
    motion_1 = torch.cat([example["motion_1"] for example in data], dim=0)
    motion_2 = torch.cat([example["motion_2"] for example in data], dim=0)
    phase_real_v_1_d = torch.cat([example["phase_real_v_1_d"] for example in data], dim=0)
    phase_real_v_2_d = torch.cat([example["phase_real_v_2_d"] for example in data], dim=0)

    return phase_real_z, phase_real_latent, phase_real_c_1_d, phase_real_c_2_d, phase_real_c_3_d, phase_real_v_1_d, phase_real_v_2_d, phase_real_v_s, motion_1, motion_2, motion_ffhq, model_list
# ----------------------------------------------------------------------------

def training_loop(
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        D_patch_kwargs={},  # Options for patch discriminator (deprecated).
        G_opt_kwargs={},  # Options for generator optimizer.
        D_opt_kwargs={},  # Options for discriminator optimizer.
        D_patch_opt_kwargs={},  # Options for patch discriminator optimizer (deprecated).
        augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
        loss_kwargs={},  # Options for loss function.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu=4,  # Number of samples processed at a time by one GPU.
        ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup=0.05,  # EMA ramp-up coefficient. None = no rampup.
        G_reg_interval=None,  # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
        D_patch_reg_interval=16,  # How often to perform regularization for D patch (deprecated)
        augment_p=0,  # Initial value of augmentation probability.
        ada_target=None,  # ADA target value. None = fixed p.
        ada_interval=4,  # How often to perform ADA adjustment?
        ada_kimg=500,
        # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        kimg_per_tick=4,  # Progress snapshot interval.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
        resume_pkl=None,  # Network pickle to resume training from.
        resume_kimg=0,  # First kimg to report when resuming training.
        cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
        abort_fn=None,
        # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
        motion_scale=1.0,  # Scale of the motion-related cross-attention outputs.
        swapping_prob=0.5,  # Probability to set dynamic data as static data.
        half_static=True,  # Whether or not to set the second half of the batchsize as static data.
        resume_pkl_G_syn=None,  # Checkpoint of pre-trained GenHead generator for training data synthesis.
        truncation_psi=0.7,  # Truncation rate of GenHead for training data synthesis.
        cross_lr_scale=1.0,  # Learning rate scale of the motion-related cross attentions.
        gan_model_base_dir=None,
        vae_pretrained=None,
        render_pretrain=None,
        vae_triplane_config=None,
        pretrain_portrait_4D = None,
        load_tri_pretrain = True,

):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False  # Avoids errors with the augmentation pipe.

    # Load training set.
    batch_size_dataset = batch_size // num_gpus

    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(
        **training_set_kwargs)  # subclass of training.dataset.Dataset
    # Construct networks.
    if rank == 0:
        print('Constructing networks...')

    common_kwargs = dict(c_dim=25, img_resolution=512, img_channels=3)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module

    for m in G.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    D_img_channel = training_set.num_channels * 3 if G_kwargs.rendering_kwargs.get(
        'gen_lms_cond') else training_set.num_channels * 2
    # if G_kwargs.rendering_kwargs.get('gen_mask_cond'): D_img_channel += 1
    D = dnnlib.util.construct_class_by_name(c_dim=25, img_resolution=512,
                                            img_channels=D_img_channel, **D_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    # Load pre-trained GenHead model
    if rank == 0:
        print(f'Resuming GenHead from "{resume_pkl_G_syn}"')
    config_gan_model = OmegaConf.load(resume_pkl_G_syn)

    G_syn_dict = gan_model(config_gan_model, device, gan_model_base_dir)
    # with dnnlib.util.open_url(resume_pkl_G_syn) as f:
    #     G_syn = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)

    # G_syn = PartTriPlaneGeneratorDeform(*G_syn_meta.init_args, **G_syn_meta.init_kwargs).eval().requires_grad_(False).to(device)
    # misc.copy_params_and_buffers(G_syn_meta, G_syn, require_all=False)
    # G_syn.neural_rendering_resolution = G_syn_meta.neural_rendering_resolution
    # G_syn.rendering_kwargs = G_syn_meta.rendering_kwargs

    # For VAE decoder
    config_vae_triplane = OmegaConf.load(vae_triplane_config)
    vae_triplane = AutoencoderKLTriplane(ddconfig=config_vae_triplane['ddconfig'], lossconfig=None,
                                         embed_dim=8)
    vae_triplane_model_file = os.path.join(vae_pretrained, 'pytorch_model.bin')
    if not os.path.isfile(vae_triplane_model_file):
        raise RuntimeError(f"{vae_triplane_model_file} does not exist")
    vae_triplane_state_dict = torch.load(vae_triplane_model_file, map_location="cpu")
    vae_triplane.load_state_dict(vae_triplane_state_dict)
    vae_triplane.requires_grad_(False)
    vae_triplane = vae_triplane.to(device, dtype=torch.float16)

    # For rendering
    Rendering = RenderingClass(device, config_vae_triplane['render_network_pkl'],
                               ws_avg_pkl=config_vae_triplane['ws_avg'])
    data_std = torch.load(config_vae_triplane['std_dir']).to(device).reshape(1, -1, 1, 1, 1)
    data_mean = torch.load(config_vae_triplane['mean_dir']).to(device).reshape(1, -1, 1, 1, 1)

    # For LPIPS loss computation
    lpips = LPIPS(vgg_path=config_vae_triplane['vgg_path'], net='vgg', model_path=config_vae_triplane['vgg']).to(device)
    set_requires_grad(lpips, requires_grad=False)

    # For ID loss computation
    idloss = IDLoss(config_vae_triplane['ir_se50'])
    idloss = idloss.eval().to(device)
    set_requires_grad(idloss, requires_grad=False)

    # For PD-FGC motion embedding extraction
    # pd_fgc = FanEncoder()
    # weight_dict = torch.load(motion_pretrained)
    # pd_fgc.load_state_dict(weight_dict, strict=False)
    # pd_fgc = pd_fgc.eval().to(device)
    # set_requires_grad(pd_fgc, requires_grad=False)

    # set D_patch for 3D-to-2D imitation (deprecated), see Mimic3D for details: https://github.com/SeanChenxy/Mimic3D  check this
    D_patch = None
    # if loss_kwargs.patch_scale < 1:
    #     img_resolution = loss_kwargs.neural_rendering_resolution_initial if loss_kwargs.neural_rendering_resolution_final is None else loss_kwargs.neural_rendering_resolution_final
    #     common_patch_kwargs = dict(c_dim=0, img_resolution=img_resolution, img_channels=3)
    #     D_patch = dnnlib.util.construct_class_by_name(**D_patch_kwargs, **common_patch_kwargs).train().requires_grad_(
    #         False).to(device)  # subclass of torch.nn.Module

    # if pretrain_portrait_4D is not  None and resume_pkl is None:
    #     print(f'Resuming encoders from "{pretrain_portrait_4D}"')
    #     with dnnlib.util.open_url(pretrain_portrait_4D) as f:
    #         print("Reloading Modules!")
    #         load_model = legacy.load_network_pkl(f)
    #         G_encoder = load_model['G_ema']
    #     misc.copy_params_and_buffers(G_encoder.encoder_global, G.encoder_global )
    #     misc.copy_params_and_buffers(G_encoder.encoder_detail, G.encoder_detail )
    #     del G_encoder
    if (resume_pkl is None) and (rank == 0):
        print(f'Resuming rendering and super and D from "{render_pretrain}"')
        with dnnlib.util.open_url(render_pretrain) as f:
            print("Reloading Modules!")
            load_model = legacy.load_network_pkl(f)
            G_deco = load_model['G_ema']
            D_deco = load_model['D']
        misc.copy_params_and_buffers(G_deco.decoder, G.decoder, require_all=True)
        misc.copy_params_and_buffers(G_deco.renderer, G.renderer, require_all=True)
        misc.copy_params_and_buffers(G_deco.ray_sampler, G.ray_sampler, require_all=True)
        misc.copy_params_and_buffers(G_deco.superresolution, G.superresolution, require_all=True)
        misc.copy_params_and_buffers(D_deco, D, require_all=True)
    # if load_tri_pretrain:
    #     misc.copy_params_and_buffers(Rendering.Render.face_backbone, G.face_backbone, require_all=True)
    #     misc.copy_params_and_buffers(Rendering.Render.triplnae_encoder, G.triplnae_encoder, require_all=True)
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        load_model = [('G', G), ('G_ema', G_ema)]
        if D is not None:
            load_model.append(('D', D))
        if D_patch is not None:
            load_model.append(('D_patch', D_patch))
        for name, module in load_model:
            if name in resume_data and resume_data[name] is not None:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
            else:
                print(f'resume_data do not have {name}')
        if 'augment_pipe' in resume_data and resume_data['augment_pipe'] is not None:
            augment_p = resume_data['augment_pipe'].p



    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(
            device)  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe, lpips, D_patch]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    # conditioning_params = torch.load(config_vae_triplane['conditioning_params_dir']).to(device)

    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, G_syn=G_syn_dict, D_patch=D_patch,
                                               augment_pipe=augment_pipe, lpips=lpips, id_loss=idloss,
                                               conditioning_params=config_vae_triplane['conditioning_params_dir'], w_avg=config_vae_triplane['ws_avg'],
                                               **loss_kwargs)  # subclass of training.loss.Loss
    phases = []
    phases_asserts = [('G', G, G_opt_kwargs, G_reg_interval), ]
    if D is not None:
        phases_asserts.append(('D', D, D_opt_kwargs, D_reg_interval))
    # no d_patch
    if D_patch is not None:
        phases_asserts.append(('D_patch', D_patch, D_patch_opt_kwargs, D_patch_reg_interval))
    for name, module, opt_kwargs, reg_interval in phases_asserts:

        # if G_update_all is False:
        #     parameter_names = [n for (n, p) in module.named_parameters() if 'superresolution' not in n and not ('decoder' in n and 'encoder_global' not in n)  and 'bn' not in n] # do not update mlp and super-resolution following Real-Time Radiance Fields for Single-Image Portrait View Synthesis
        # else:
        parameter_names = [n for (n, p) in module.named_parameters() if 'bn' not in n]

        if name == 'G':
            parameters_group = []
            parameters_cross_names = [n for n in parameter_names if 'encoder_canonical' in n and (
                    'maps' in n or 'maps_neutral' in n or 'proj_y' in n or 'proj_y_neutral' in n or 'norm2' in n or 'attn2' in n)]
            parameters_base_names = [n for n in parameter_names if not n in parameters_cross_names]
            parameters_cross = [p for (n, p) in module.named_parameters() if n in parameters_cross_names]
            parameters_base = [p for (n, p) in module.named_parameters() if n in parameters_base_names]
            parameters_group.append({'params': parameters_cross, 'name': 'G_cross'})
            parameters_group.append({'params': parameters_base, 'name': 'G_base'})
            parameters = parameters_group
        else:
            parameters = [p for (n, p) in module.named_parameters() if n in parameter_names]

        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(parameters, **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(parameters, **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]

        if name == 'G':
            for param_group in opt.param_groups:
                if param_group['name'] == 'G_cross':
                    param_group['lr'] = param_group['lr'] * cross_lr_scale

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None



    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus,
                                                seed=random_seed)
    training_set_iterator = iter(
        torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, collate_fn=collate_fn,
                                    batch_size=batch_size // num_gpus, **data_loader_kwargs))

    val_set_iterator = iter(
        torch.utils.data.DataLoader(dataset=training_set ,collate_fn=collate_fn, shuffle=True,
                                    batch_size=4, num_workers=1))
    if rank == 0:
        print('Exporting sample images...')
        batch_gpu_val = 4
        grid_size, cur_z_vals, cur_c_vals, cur_exp_out_vals, motion_1_out_vals, motion_2_out_vals, model_list_out_vals, ref_tri_vals = setup_snapshot_image_grid_gallery(
            val_set_iterator, vae_triplane, data_std, data_mean, Rendering, device)
        grid_size = (2, 2)
        cur_z_vals = cur_z_vals.split(batch_gpu_val)
        cur_c_vals = cur_c_vals.split(batch_gpu_val)
        cur_exp_out_vals = cur_exp_out_vals.split(batch_gpu_val)
        model_list_out_vals = listfunc(model_list_out_vals, batch_gpu_val)
        motion_1_out_vals = motion_1_out_vals.split(batch_gpu_val)
        motion_2_out_vals = motion_2_out_vals.split(batch_gpu_val)
        ref_tri_vals = ref_tri_vals.split(batch_gpu_val)
        out = []
        ref_out = []
        grid_c = []
        grid_c_recon = []
        kkkk = 0
        with torch.no_grad():
            for cur_z_val, cur_exp_out_val, cur_c_val, model_list_out_val, ref_tri_val in \
                    zip(cur_z_vals, cur_exp_out_vals, cur_c_vals, model_list_out_vals, ref_tri_vals):
                kkkk = kkkk + 1
                # print(' model_list_out_val',  model_list_out_val)
                syn_out = loss.gen_data_by_G_next3D(cur_z_val, cur_exp_out_val, cur_c_val, model_list_out_val)
                ref_tri_tri = Rendering.gen_triplane(ref_tri_val, cur_exp_out_val[:,2])
                ref_imgs_out = Rendering.get_img_with_tri(ref_tri_tri, syn_out['c'][:,2])
                grid_c_recon.append(syn_out['c'][:,2])
                # torch.save(syn_out['c'][:,2].detach().cpu(), f'/nas5/liuhongyu/dataset_video/dataset/val/c_val_{kkkk}.pt')
                out.append(syn_out)
                ref_out.append(ref_imgs_out)

        grid_c_recon.append(torch.load(config_vae_triplane['syn_out_c_path']).to(device))
        images_all = torch.cat([o['image_sr'] for o in out], dim=0)
        # torch.save(images_all.detach().cpu(), '/nas5/liuhongyu/dataset_video/dataset/val/images_all.pt')
        # images_all = torch.load(config_vae_triplane['images_all_path']).to(device)
        ref_imgs_all_val = torch.cat(ref_out, dim=0)
        # torch.save(ref_imgs_all_val.detach().cpu(), '/nas5/liuhongyu/dataset_video/dataset/val/ref_imgs_all_val.pt')
        # ref_imgs_all_val = torch.load(config_vae_triplane['ref_imgs_all_val_path']).to(device)
        images_all = images_all.reshape(-1, 3, *images_all.shape[-3:])
        images_app_val = images_all[:, 0]
        images_mot_val = images_all[:, 1]
        images_recon_val = images_all[:, 2]

        save_image_grid_all(127.5 * (images_app_val.cpu().numpy() + 1), 127.5 * (images_mot_val.cpu().numpy() + 1),
                            127.5 * (images_recon_val.cpu().numpy() + 1), 127.5 * (ref_imgs_all_val.cpu().numpy() + 1),
                            os.path.join(run_dir, 'reals.png'),
                            drange=[0, 255], grid_size=grid_size)
        # grid_c = torch.cat(grid_c)

        images_app_val = images_app_val.split(batch_gpu_val)
        images_mot_val = images_mot_val.split(batch_gpu_val)
        # grid_c = grid_c.split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    batch_num = batch_size // num_gpus

    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_z, phase_real_latent, phase_real_c_1_d, phase_real_c_2_d, phase_real_c_3_d, phase_real_v_1_d, phase_real_v_2_d, phase_real_v_s, motion_1, motion_2, motion_ffhq, model_list = next(
                training_set_iterator)

            phase_real_z, phase_real_c, phase_real_exp_params, motion_1, motion_2, model_list, phase_real_cano_tri, phase_real_ref_tri, phase_real_exp_params_target = fetch_dataset(
                phase_real_z.to(device), phase_real_latent.to(device), phase_real_c_1_d.to(device), phase_real_c_2_d.to(device), phase_real_c_3_d.to(device), phase_real_v_1_d.to(device),
                phase_real_v_2_d.to(device), phase_real_v_s.to(device), motion_1.to(device), motion_2.to(device), motion_ffhq.to(device), model_list, vae_triplane, data_std,
                data_mean, Rendering)
            phase_real_z = phase_real_z.split(batch_gpu)
            # phase_real_latent = phase_real_latent.split(batch_gpu).to(device)
            phase_real_c = phase_real_c.split(batch_gpu)
            phase_real_exp_params = phase_real_exp_params.split(batch_gpu)
            phase_real_motions_app = motion_1.split(batch_gpu)
            phase_real_motions = motion_2.split(batch_gpu)
            phase_real_cano_tri = phase_real_cano_tri.split(batch_gpu)
            phase_real_ref_tri = phase_real_ref_tri.split(batch_gpu)
            phase_real_model_list = listfunc(model_list, batch_gpu)
            phase_real_exp_params_target = phase_real_exp_params_target.split(batch_gpu)

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Online data generation. For efficiency, use same generated data for different phases
        phase_real_img_app = []
        phase_real_img_mot = []
        phase_real_img_recon = []

        phase_real_depth_recon = []
        phase_real_feature_recon = []
        phase_real_triplane_recon = []
        phase_real_c_recon = []
        # phase_real_motions_app = []
        # phase_real_motions = []
        # phase_real_triplane_latent_recon = []

        with torch.no_grad():
            for real_z,   real_pose_params, real_exp_params, real_models in \
                    zip(phase_real_z,   phase_real_c, phase_real_exp_params, phase_real_model_list):
                syn_out = loss.gen_data_by_G_next3D(real_z,   real_exp_params, real_pose_params, real_models)

                # Multiview images
                real_img = syn_out['image_sr']
                real_img = real_img.reshape(-1, 3, *real_img.shape[1:])
                real_img_app = real_img[:, 0]
                real_img_mot = real_img[:, 1]
                real_img_recon = real_img[:, 2]

                # Segmentation masks

                # Camera poses
                real_c_recon = syn_out['c']
                # real_c_recon = real_c_recon.reshape(-1, 3, *real_c_recon.shape[1:])
                real_c_recon = real_c_recon[:, 2]

                # Depth images
                real_depth_recon = syn_out['image_depth']
                real_depth_recon = real_depth_recon.reshape(-1, 3, *real_depth_recon.shape[1:])
                real_depth_recon = real_depth_recon[:, 2]

                # Feature maps before super-resolution module
                real_feature_recon = syn_out['image_feature']
                real_feature_recon = real_feature_recon.reshape(-1, 3, *real_feature_recon.shape[1:])
                real_feature_recon = real_feature_recon[:, 2]

                # Sampled tri-plane features
                real_triplane_recon = syn_out['triplane']
                real_triplane_recon = real_triplane_recon.reshape(-1, 3, *real_triplane_recon.shape[1:])
                real_triplane_recon = real_triplane_recon[:, 2]

                # Sampled latent recon
                # real_triplane_latent_recon = syn_out['vae_out_tri']

                phase_real_img_app.append(real_img_app)
                phase_real_img_mot.append(real_img_mot)
                phase_real_img_recon.append(real_img_recon)
                phase_real_depth_recon.append(real_depth_recon)
                phase_real_feature_recon.append(real_feature_recon)
                phase_real_triplane_recon.append(real_triplane_recon)
                phase_real_c_recon.append(real_c_recon)
                # phase_real_motions_app.append(real_motions_app)
                # phase_real_motions.append(real_motions)
                # phase_real_triplane_latent_recon.append(real_triplane_latent_recon)

        # Execute training phases.
        for phase in phases:
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img_app, real_img_mot, real_img_recon, real_depth_recon, real_feature_recon, real_triplane_recon, real_c_recon, real_motions_app, real_motions, real_cano_tri, real_ref_tri, real_exp_target in \
                    zip(phase_real_img_app, phase_real_img_mot, phase_real_img_recon, phase_real_depth_recon,
                        phase_real_feature_recon, phase_real_triplane_recon, phase_real_c_recon, phase_real_motions_app,
                        phase_real_motions, phase_real_cano_tri, phase_real_ref_tri, phase_real_exp_params_target):
                loss.accumulate_gradients(phase=phase.name, real_img_app=real_img_app, real_img_mot=real_img_mot,
                                          real_img_recon=real_img_recon,
                                          real_depth_recon=real_depth_recon, real_feature_recon=real_feature_recon,
                                          real_triplane_recon=real_triplane_recon,
                                          real_c_recon=real_c_recon,   mesh= real_exp_target, motions_app=real_motions_app,
                                          motions=real_motions,
                                          real_cano_tri=real_cano_tri, real_ref_tri=real_ref_tri,
                                          gain=phase.interval, cur_nimg=cur_nimg,
                                          motion_scale=motion_scale, swapping_prob=swapping_prob,
                                          half_static=half_static)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                # Do not update mlp decoder and super-resolution module at the warm-up stage following Live3dportrait: https://arxiv.org/abs/2305.02310 
                if cur_nimg <= loss.discrimination_kimg * 1e3 and phase.name == 'G':
                    sub_params = [p for (n, p) in phase.module.named_parameters() if
                                  'superresolution' in n or ('decoder' in n and 'encoder_global' not in n)]
                    for param in sub_params:
                        if param.grad is not None:
                            param.grad.zero_()

                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (
                    ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2 ** 30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [
            f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            print('Saving images...')
            out = []
            for image_app, image_mot, motion_app, motion, ref_tri, c, exp_vals in zip(
                    images_app_val, images_mot_val,
                    motion_1_out_vals, motion_2_out_vals, ref_tri_vals, grid_c_recon, cur_exp_out_vals):

                with torch.no_grad():
                    out.append(
                        G_ema(image_app, image_mot, motion_app, motion, c=c, mesh= exp_vals[:, 2],  triplane_recon=ref_tri, ws_avg=Rendering.ws_avg,
                               motion_scale=motion_scale))
            if 'image' in out[0]:
                images = torch.cat([o['image'].cpu() for o in out]).numpy()
                print(111111111111111111111111111111)
                print(images.shape)
                print(images.max())
                print(images.min())
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1],
                                grid_size=grid_size)
            if 'image_depth' in out[0]:
                images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
                save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_depth.png'),
                                drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
            if 'image_sr' in out[0] and out[0]['image_sr'] is not None:
                images_sr = torch.cat([o['image_sr'].cpu() for o in out]).numpy()
                save_image_grid(images_sr, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_sr.png'), drange=[-1, 1],
                                grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('D_patch', D_patch),
                                 ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module  # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

# ----------------------------------------------------------------------------
