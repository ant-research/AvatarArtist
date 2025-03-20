# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import torch
import torchvision
from torch_utils import persistence
from training_avatar_texture.networks_stylegan2_new import Generator as StyleGAN2Backbone_cond
from training_avatar_texture.volumetric_rendering.renderer import fill_mouth


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 use_tanh=False,
                 use_two_rgb=False,
                 use_norefine_rgb = False,
                 topology_path=None,  #
                 sr_num_fp16_res=0,
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 rendering_kwargs={},
                 sr_kwargs={},
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.texture_backbone = StyleGAN2Backbone_cond(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32,
                                                       mapping_kwargs=mapping_kwargs, use_tanh=use_tanh,
                                                       **synthesis_kwargs)  # render neural texture

        self.backbone = StyleGAN2Backbone_cond(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32 * 3,
                                               mapping_ws=self.texture_backbone.num_ws, use_tanh=use_tanh,
                                               mapping_kwargs=mapping_kwargs, **synthesis_kwargs)


        self.neural_rendering_resolution = 128
        self.rendering_kwargs = rendering_kwargs
        self.fill_mouth = True
        self.use_two_rgb = use_two_rgb
        self.use_norefine_rgb = use_norefine_rgb
        # print(self.use_two_rgb)

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        c = c[:, :self.c_dim]  # remove expression labels
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def visualize_mesh_condition(self, mesh_condition, to_imgs=False):
        uvcoords_image = mesh_condition['uvcoords_image'].clone().permute(0, 3, 1, 2)  # [B, C, H, W]
        ori_alpha_image = uvcoords_image[:, 2:].clone()
        full_alpha_image, mouth_masks = fill_mouth(ori_alpha_image, blur_mouth_edge=False)
        # upper_mouth_mask = mouth_masks.clone()
        # upper_mouth_mask[:, :, :87] = 0
        # alpha_image = torch.clamp(ori_alpha_image + upper_mouth_mask, min=0, max=1)

        if to_imgs:
            uvcoords_image[full_alpha_image.expand(-1, 3, -1, -1) == 0] = -1
            uvcoords_image = ((uvcoords_image + 1) * 127.5).to(dtype=torch.uint8).cpu()
            vis_images = []
            for vis_uvcoords in uvcoords_image:
                vis_images.append(torchvision.transforms.ToPILImage()(vis_uvcoords))
            return vis_images
        else:
            return uvcoords_image

    def synthesis(self, ws,   neural_rendering_resolution=None, update_emas=False,
                  cache_backbone=False, use_cached_backbone=False,
                  return_featmap=False, evaluation=False, **synthesis_kwargs):

        # Create a batch of rays for volume rendering

        texture_feat = self.texture_backbone.synthesis(ws, cond_list=None, return_list=False, update_emas=update_emas,
                                                       **synthesis_kwargs)
        static_feat = self.backbone.synthesis(ws, cond_list=None, return_list=False, update_emas=update_emas,
                                              **synthesis_kwargs)

        static_plane = static_feat
        static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        texture_feat_out = texture_feat.unsqueeze(1)
        out_triplane = torch.cat([texture_feat_out, static_plane], 1)

        return out_triplane


    def forward(self, z, c, v, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None,
                update_emas=False, cache_backbone=False,
                use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        return self.synthesis(ws, c, v, update_emas=update_emas,
                              neural_rendering_resolution=neural_rendering_resolution,
                              cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone,
                              **synthesis_kwargs)




