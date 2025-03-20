# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from os import device_encoding
from turtle import update
import math
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision
from torch_utils import persistence
from training_avatar_texture.networks_stylegan2_new import Generator as StyleGAN2Backbone_cond
from training_avatar_texture.volumetric_rendering.renderer import ImportanceRenderer, ImportanceRenderer_bsMotion
from training_avatar_texture.volumetric_rendering.ray_sampler import RaySampler, RaySampler_zxc
import dnnlib
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
        self.renderer = ImportanceRenderer_bsMotion()
        self.ray_sampler = RaySampler_zxc()
        # print(111111111111111111, use_tanh)
        self.texture_backbone = StyleGAN2Backbone_cond(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32,
                                                       mapping_kwargs=mapping_kwargs, use_tanh=use_tanh,
                                                       **synthesis_kwargs)  # render neural texture
        self.face_backbone = StyleGAN2Backbone_cond(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32,
                                                    mapping_kwargs=mapping_kwargs, use_tanh=use_tanh,
                                                    **synthesis_kwargs)
        self.backbone = StyleGAN2Backbone_cond(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32 * 3,
                                               mapping_ws=self.texture_backbone.num_ws, use_tanh=use_tanh,
                                               mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs['superresolution_module'], channels=32,
            img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res,
            sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                                       'decoder_output_dim': 32})
        self.neural_rendering_resolution = 128
        self.rendering_kwargs = rendering_kwargs
        self.fill_mouth = True
        self.triplnae_encoder = EncoderTriplane()
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

    def synthesis(self, ws, c, mesh_condition, neural_rendering_resolution=None, update_emas=False,
                  cache_backbone=False, use_cached_backbone=False,
                  return_featmap=False, evaluation=False, **synthesis_kwargs):
        batch_size = ws.shape[0]
        cam = c[:, -25:]
        cam2world_matrix = cam[:, :16].view(-1, 4, 4)
        intrinsics = cam[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        texture_feat = self.texture_backbone.synthesis(ws, cond_list=None, return_list=False, update_emas=update_emas,
                                                       **synthesis_kwargs)
        static_feat = self.backbone.synthesis(ws, cond_list=None, return_list=False, update_emas=update_emas,
                                              **synthesis_kwargs)

        static_plane = static_feat
        static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        static_plane_face = static_plane[:, 0]
        # texture_feats = self.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=update_emas,
        #                                                 **synthesis_kwargs)
        # static_feats = self.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=update_emas,
        #                                        **synthesis_kwargs)
        # static_plane = static_feats[-1]
        # static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        # static_feats[0] = static_feats[0].view(len(static_plane), 3, 32, static_feats[0].shape[-2],
        #                                        static_feats[0].shape[-1])[:, 0]
        # static_feats[-1] = static_plane[:, 0]
        # assert len(static_feats) == len(texture_feats)
        bbox_256 = [57, 185, 64, 192]  # the face region is the center-crop result from the frontal triplane.

        # rendering_images, full_alpha_image, mouth_masks, mask_images = self.rasterize(texture_feats,
        #                                                                                           mesh_condition[
        #                                                                                               'uvcoords_image'],
        #                                                                                           static_feats,
        #                                                                                           bbox_256)
        texture_feat_out = texture_feat.unsqueeze(1)
        out_triplane = torch.cat([texture_feat_out, static_plane], 1)
        rendering_image, full_alpha_image, rendering_image_only_img, mask_images = self.rasterize_sinle_input(
            texture_feat,
            mesh_condition ,
            static_plane_face,
            bbox_256
             )
        if self.use_norefine_rgb:
            rendering_stitch = rendering_image_only_img
        else:

            rendering_images_no_masks = self.triplnae_encoder(rendering_image)
            rendering_images = []
            for index, rendering_image_no_mask in enumerate(rendering_images_no_masks):
                rendering_images_each = torch.cat([rendering_image_no_mask, mask_images[index]], dim=1)
                rendering_images.append(rendering_images_each)
            rendering_images.append(rendering_image)
            rendering_stitch = self.face_backbone.synthesis(ws, rendering_images, return_list=False,
                                                            update_emas=update_emas, **synthesis_kwargs)



        rendering_stitch_, full_alpha_image_ = torch.zeros_like(rendering_stitch), torch.zeros_like(full_alpha_image)

        rendering_stitch_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(rendering_stitch,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(full_alpha_image,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image, rendering_stitch = full_alpha_image_, rendering_stitch_

        # blend features of neural texture and tri-plane
        full_alpha_image = torch.cat(
            (full_alpha_image, torch.zeros_like(full_alpha_image), torch.zeros_like(full_alpha_image)), 1).unsqueeze(2)
        rendering_stitch = torch.cat(
            (rendering_stitch, torch.zeros_like(rendering_stitch), torch.zeros_like(rendering_stitch)), 1)
        rendering_stitch = rendering_stitch.view(*static_plane.shape)
        blended_planes = rendering_stitch * full_alpha_image + static_plane * (1 - full_alpha_image)

        # Perform volume rendering
        if evaluation:
            assert 'noise_mode' in synthesis_kwargs.keys() and synthesis_kwargs['noise_mode'] == 'const', \
                ('noise_mode' in synthesis_kwargs.keys(), synthesis_kwargs['noise_mode'] == 'const')
        feature_samples, depth_samples, weights_samples = self.renderer(blended_planes, self.decoder, ray_origins,
                                                                        ray_directions,
                                                                        self.rendering_kwargs, evaluation=evaluation)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]

        if self.use_two_rgb:
            rendering_stitch_low_detail_ = torch.zeros_like(rendering_image_only_img)
            rendering_stitch_low_detail_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(
                rendering_image_only_img,
                size=(128, 128),
                mode='bilinear',
                antialias=True)
            rendering_stitch_low_detail = rendering_stitch_low_detail_
            rendering_stitch_low_detail = torch.cat(
                (rendering_stitch_low_detail, torch.zeros_like(rendering_stitch_low_detail),
                 torch.zeros_like(rendering_stitch_low_detail)), 1)
            rendering_stitch_low_detail = rendering_stitch_low_detail.view(*static_plane.shape)
            blended_planes_low_detail = rendering_stitch_low_detail * full_alpha_image + static_plane * (
                        1 - full_alpha_image)
            feature_samples_low_detail, _, _ = self.renderer(blended_planes_low_detail, self.decoder, ray_origins,
                                                                            ray_directions,
                                                                            self.rendering_kwargs,
                                                                            evaluation=evaluation)
            feature_samples_low_detail = feature_samples_low_detail.permute(0, 2, 1).reshape(N, feature_samples_low_detail.shape[-1], H, W).contiguous()

            rgb_image = feature_samples_low_detail[:, :3]

        sr_image = self.superresolution(rgb_image, feature_image, ws,
                                        noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                                        **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if
                                           k != 'noise_mode'})
        if return_featmap:
            return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,
                    'image_feature': feature_image, 'triplane': blended_planes,
                    }  # static_plane, 'texture_map': texture_feats[-2]}
        else:
            return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, "out_triplane":out_triplane}

    def synthesis_withTexture(self, ws, texture_feats, c, mesh_condition, static_feats=None,
                              neural_rendering_resolution=None, update_emas=False,
                              cache_backbone=False, use_cached_backbone=False, evaluation=False, **synthesis_kwargs):
        bs = ws.shape[0]
        # eg3d_ws, texture_ws = ws[:, :self.texture_backbone.num_ws], ws[:, self.texture_backbone.num_ws:]
        # cam = c[:, :25]
        cam = c[:, -25:]
        cam2world_matrix = cam[:, :16].view(-1, 4, 4)
        intrinsics = cam[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        if static_feats is None:
            static_feats = self.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=update_emas,
                                                   **synthesis_kwargs)

        static_plane = static_feats[-1].view(bs, 3, 32, static_feats[-1].shape[-2], static_feats[-1].shape[-1])
        assert len(static_feats) == len(texture_feats), (len(static_feats), len(texture_feats))
        bbox_256 = [57, 185, 64, 192]

        rendering_images, full_alpha_image, mouth_masks = self.rasterize(texture_feats,
                                                                         mesh_condition['uvcoords_image'],
                                                                         bbox_256=bbox_256,
                                                                         static_feats=[static_feats[0].view(bs, 3, 32,
                                                                                                            static_feats[
                                                                                                                0].shape[
                                                                                                                -2],
                                                                                                            static_feats[
                                                                                                                0].shape[
                                                                                                                -1])[:,
                                                                                       0]] +
                                                                                      static_feats[1:-1] + [
                                                                                          static_plane[:, 0]])
        rendering_stitch = self.face_backbone.synthesis(ws, rendering_images, return_list=False,
                                                        update_emas=update_emas, **synthesis_kwargs)

        # upper_mouth_mask = mouth_masks.clone()
        # upper_mouth_mask[:, :, :87] = 0
        # rendering_stitch = F.interpolate(static_plane[:, 0, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]], size=(256, 256), mode='bilinear',
        #                                  antialias=True) * upper_mouth_mask + rendering_stitch * (1 - upper_mouth_mask)

        rendering_stitch_, full_alpha_image_ = torch.zeros_like(rendering_stitch), torch.zeros_like(full_alpha_image)
        rendering_stitch_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(rendering_stitch,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(full_alpha_image,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image, rendering_stitch = full_alpha_image_, rendering_stitch_

        # blend features of neural texture and tri-plane
        full_alpha_image = torch.cat(
            (full_alpha_image, torch.zeros_like(full_alpha_image), torch.zeros_like(full_alpha_image)), 1).unsqueeze(2)
        rendering_stitch = torch.cat(
            (rendering_stitch, torch.zeros_like(rendering_stitch), torch.zeros_like(rendering_stitch)), 1)
        rendering_stitch = rendering_stitch.view(*static_plane.shape)
        blended_planes = rendering_stitch * full_alpha_image + static_plane * (1 - full_alpha_image)

        # if flag is not False:
        #     import cv2
        #     with torch.no_grad():
        #         if not hasattr(self, 'weight'):
        #             self.weight = torch.nn.Conv2d(32, 3, 1).weight.cuda()
        #         weight = self.weight
        #         vis = torch.nn.functional.conv2d((rendering_stitch * full_alpha_image)[:, 0, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]], weight)
        #         max_ = [torch.max(torch.abs(vis[:, i])) for i in range(3)]
        #         for i in range(3): vis[:, i] /= max_[i]
        #         print('rendering_stitch', vis.max().item(), vis.min().item())
        #         vis = torch.cat([vis[i] for i in range(blended_planes.shape[0])], dim=-1)
        #         vis = (vis.permute(1, 2, 0).clamp(min=-1.0, max=1.0) + 1.) * 127.5
        #         cv2.imwrite('vis_%s_rendering_stitch.png' % flag, vis.cpu().numpy().astype(np.uint8)[..., ::-1])
        #         vis = torch.nn.functional.conv2d((static_plane * (1 - full_alpha_image))[:, 0], weight)
        #         for i in range(3): vis[:, i] /= max_[i]
        #         print('static_plane', vis.max().item(), vis.min().item())
        #         vis = torch.cat([vis[i] for i in range(blended_planes.shape[0])], dim=-1)
        #         vis = (vis.permute(1, 2, 0).clamp(min=-1.0, max=1.0) + 1.) * 127.5
        #         cv2.imwrite('vis_%s_static_plane.png' % flag, vis.cpu().numpy().astype(np.uint8)[..., ::-1])
        #         vis = torch.nn.functional.conv2d(blended_planes[:, 0], weight)
        #         for i in range(3): vis[:, i] /= max_[i]
        #         print('blended_planes', vis.max().item(), vis.min().item())
        #         vis = torch.cat([vis[i] for i in range(blended_planes.shape[0])], dim=-1)
        #         vis = (vis.permute(1, 2, 0).clamp(min=-1.0, max=1.0) + 1.) * 127.5
        #         cv2.imwrite('vis_%s_blended_planes.png' % flag, vis.cpu().numpy().astype(np.uint8)[..., ::-1])

        # Perform volume rendering
        if evaluation:
            assert 'noise_mode' in synthesis_kwargs.keys() and synthesis_kwargs['noise_mode'] == 'const', \
                ('noise_mode' in synthesis_kwargs.keys(), synthesis_kwargs['noise_mode'] == 'const')
        feature_samples, depth_samples, weights_samples = self.renderer(blended_planes, self.decoder, ray_origins,
                                                                        ray_directions,
                                                                        self.rendering_kwargs, evaluation=evaluation)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws,
                                        noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                                        **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if
                                           k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,
                'feature_image': feature_image,
                'triplane': blended_planes}  # static_plane, 'texture_map': texture_feats[-2]}

    def synthesis_withCondition(self, ws, c, mesh_condition, gt_texture_feats=None, gt_static_feats=None,
                                texture_feats_conditions=None,
                                static_feats_conditions=None, neural_rendering_resolution=None, update_emas=False,
                                cache_backbone=False,
                                use_cached_backbone=False, only_image=False, return_feats=False, **synthesis_kwargs):
        bs = ws.shape[0]
        cam = c[:, -25:]
        cam2world_matrix = cam[:, :16].view(-1, 4, 4)
        intrinsics = cam[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        if gt_texture_feats is None:
            texture_feats = self.texture_backbone.synthesis(ws, cond_list=None, return_list=True,
                                                            feat_conditions=texture_feats_conditions,
                                                            update_emas=update_emas, **synthesis_kwargs)

        if gt_static_feats is None:
            static_feats = self.backbone.synthesis(ws, cond_list=None, return_list=True,
                                                   feat_conditions=static_feats_conditions,
                                                   update_emas=update_emas, **synthesis_kwargs)

        static_plane = static_feats[-1].view(bs, 3, 32, static_feats[-1].shape[-2], static_feats[-1].shape[-1])
        assert len(static_feats) == len(texture_feats)
        bbox_256 = [57, 185, 64, 192]

        rendering_images, full_alpha_image, mouth_masks = self.rasterize(texture_feats,
                                                                         mesh_condition['uvcoords_image'],
                                                                         bbox_256=bbox_256,
                                                                         static_feats=[static_feats[0].view(bs, 3, 32,
                                                                                                            static_feats[
                                                                                                                0].shape[
                                                                                                                -2],
                                                                                                            static_feats[
                                                                                                                0].shape[
                                                                                                                -1])[:,
                                                                                       0]] +
                                                                                      static_feats[1:-1] + [
                                                                                          static_plane[:, 0]])
        rendering_stitch = self.face_backbone.synthesis(ws, rendering_images, return_list=False,
                                                        update_emas=update_emas, **synthesis_kwargs)

        rendering_stitch_, full_alpha_image_ = torch.zeros_like(rendering_stitch), torch.zeros_like(full_alpha_image)
        rendering_stitch_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(rendering_stitch,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(full_alpha_image,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image, rendering_stitch = full_alpha_image_, rendering_stitch_

        # blend features of neural texture and tri-plane
        full_alpha_image = torch.cat(
            (full_alpha_image, torch.zeros_like(full_alpha_image), torch.zeros_like(full_alpha_image)), 1).unsqueeze(2)
        rendering_stitch = torch.cat(
            (rendering_stitch, torch.zeros_like(rendering_stitch), torch.zeros_like(rendering_stitch)), 1)
        rendering_stitch = rendering_stitch.view(*static_plane.shape)
        blended_planes = rendering_stitch * full_alpha_image + static_plane * (1 - full_alpha_image)

        # Perform volume rendering
        evaluation = 'noise_mode' in synthesis_kwargs.keys() and synthesis_kwargs['noise_mode'] == 'const'
        feature_samples, depth_samples, weights_samples = self.renderer(blended_planes, self.decoder, ray_origins,
                                                                        ray_directions,
                                                                        self.rendering_kwargs, evaluation=evaluation)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws,
                                        noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                                        **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if
                                           k != 'noise_mode'})

        if only_image:
            return {'image': sr_image}
        out = {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_feature': feature_image,
               'triplane': blended_planes}
        if return_feats:
            out['static'] = static_feats
            out['texture'] = texture_feats
        return out

    def rasterize_sinle_input(self, texture_feat_input, uvcoords_image, static_feat_input, bbox_256,
                              res_list=[32, 32, 64, 128, 256]):
        '''
        uvcoords_image [B, H, W, C]
        '''
        if not uvcoords_image.dtype == torch.float32: uvcoords_image = uvcoords_image.float()
        grid, alpha_image = uvcoords_image[..., :2], uvcoords_image[..., 2:].permute(0, 3, 1, 2)
        full_alpha_image, mouth_masks = fill_mouth(alpha_image.clone(), blur_mouth_edge=False)
        upper_mouth_mask = mouth_masks.clone()
        upper_mouth_mask[:, :, :87] = 0
        upper_mouth_alpha_image = torch.clamp(alpha_image + upper_mouth_mask, min=0, max=1)
        res = texture_feat_input.shape[2]
        bbox = [round(i * res / 256) for i in bbox_256]
        rendering_image = F.grid_sample(texture_feat_input, grid, align_corners=False)
        rendering_feat = F.interpolate(rendering_image, size=(res, res), mode='bilinear', antialias=True)
        alpha_image_ = F.interpolate(alpha_image, size=(res, res), mode='bilinear', antialias=True)
        static_feat = F.interpolate(static_feat_input[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]], size=(res, res),
                                    mode='bilinear', antialias=True)
        condition_mask_list = []
        rendering_img_nomask = rendering_feat * alpha_image_ + static_feat * (1 - alpha_image_)
        rendering_image = torch.cat([
            rendering_img_nomask,
            F.interpolate(upper_mouth_alpha_image, size=(res, res), mode='bilinear', antialias=True)], dim=1)
        for res_mask in res_list:
            condition_mask = F.interpolate(upper_mouth_alpha_image, size=(res_mask, res_mask), mode='bilinear',
                                           antialias=True)
            condition_mask_list.append(condition_mask)
            # print('rendering_images', grid.shape, rendering_images[-1].shape)
        return rendering_image, full_alpha_image, rendering_img_nomask, condition_mask_list

    def rasterize(self, texture_feats, uvcoords_image, static_feats, bbox_256):
        '''
        uvcoords_image [B, H, W, C]
        '''
        if not uvcoords_image.dtype == torch.float32: uvcoords_image = uvcoords_image.float()
        grid, alpha_image = uvcoords_image[..., :2], uvcoords_image[..., 2:].permute(0, 3, 1, 2)
        full_alpha_image, mouth_masks = fill_mouth(alpha_image.clone(), blur_mouth_edge=False)
        upper_mouth_mask = mouth_masks.clone()
        upper_mouth_mask[:, :, :87] = 0
        upper_mouth_alpha_image = torch.clamp(alpha_image + upper_mouth_mask, min=0, max=1)
        rendering_images = []
        rendering_images_nomask = []
        for idx, texture in enumerate(texture_feats):
            res = texture.shape[2]
            bbox = [round(i * res / 256) for i in bbox_256]
            rendering_image = F.grid_sample(texture, grid, align_corners=False)
            rendering_feat = F.interpolate(rendering_image, size=(res, res), mode='bilinear', antialias=True)
            alpha_image_ = F.interpolate(alpha_image, size=(res, res), mode='bilinear', antialias=True)
            static_feat = F.interpolate(static_feats[idx][:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]], size=(res, res),
                                        mode='bilinear', antialias=True)
            rendering_images.append(torch.cat([
                rendering_feat * alpha_image_ + static_feat * (1 - alpha_image_),
                F.interpolate(upper_mouth_alpha_image, size=(res, res), mode='bilinear', antialias=True)], dim=1))
            rendering_images_nomask.append(rendering_feat * alpha_image_ + static_feat * (1 - alpha_image_))
            # print('rendering_images', grid.shape, rendering_images[-1].shape)
        return rendering_images, full_alpha_image, mouth_masks, rendering_images_nomask

    def sample(self, coordinates, directions, z, c, mesh_condition, truncation_psi=1, truncation_cutoff=None,
               update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        batch_size = ws.shape[0]
        texture_feat = self.texture_backbone.synthesis(ws, cond_list=None, return_list=False, update_emas=update_emas,
                                                       **synthesis_kwargs)
        static_feat = self.backbone.synthesis(ws, cond_list=None, return_list=False, update_emas=update_emas,
                                              **synthesis_kwargs)

        static_plane = static_feat
        static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        static_plane_face = static_plane[:, 0]

        # texture_feats = self.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=update_emas,
        #                                                 **synthesis_kwargs)
        #
        # static_feats = self.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=update_emas,
        #                                        **synthesis_kwargs)
        # static_plane = static_feats[-1]
        # static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        # static_feats[0] = static_feats[0].view(len(static_plane), 3, 32, static_feats[0].shape[-2],
        #                                        static_feats[0].shape[-1])[:, 0]
        # static_feats[-1] = static_plane[:, 0]
        # assert len(static_feats) == len(texture_feats)
        bbox_256 = [57, 185, 64, 192]
        # rendering_images, full_alpha_image, mouth_masks, rendering_images_nomask = self.rasterize(texture_feats,
        #                                                                                           mesh_condition[
        #                                                                                               'uvcoords_image'],
        #                                                                                           static_feats,
        #                                                                                           bbox_256)
        rendering_image, full_alpha_image,  rendering_image_only_img, mask_images = self.rasterize_sinle_input(texture_feat,
                                                                                                 mesh_condition[
                                                                                                     'uvcoords_image'],
                                                                                                 static_plane_face,
                                                                                                 bbox_256)
        if self.use_norefine_rgb:
            rendering_stitch = rendering_image_only_img
        else:

            rendering_images_no_masks = self.triplnae_encoder(rendering_image)
            rendering_images = []
            for index, rendering_image_no_mask in enumerate(rendering_images_no_masks):
                rendering_images_each = torch.cat([rendering_image_no_mask, mask_images[index]], dim=1)
                rendering_images.append(rendering_images_each)
            rendering_images.append(rendering_image)
            rendering_stitch = self.face_backbone.synthesis(ws, rendering_images, return_list=False,
                                                            update_emas=update_emas, **synthesis_kwargs)

        rendering_stitch_, full_alpha_image_ = torch.zeros_like(rendering_stitch), torch.zeros_like(full_alpha_image)
        rendering_stitch_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(rendering_stitch,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(full_alpha_image,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image, rendering_stitch = full_alpha_image_, rendering_stitch_

        # blend features of neural texture and tri-plane
        full_alpha_image = torch.cat(
            (full_alpha_image, torch.zeros_like(full_alpha_image), torch.zeros_like(full_alpha_image)), 1).unsqueeze(2)
        rendering_stitch = torch.cat(
            (rendering_stitch, torch.zeros_like(rendering_stitch), torch.zeros_like(rendering_stitch)), 1)
        rendering_stitch = rendering_stitch.view(*static_plane.shape)
        blended_planes = rendering_stitch * full_alpha_image + static_plane * (1 - full_alpha_image)

        return self.renderer.run_model(blended_planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, mesh_condition, truncation_psi=1, truncation_cutoff=None,
                     update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        batch_size = ws.shape[0]
        texture_feat = self.texture_backbone.synthesis(ws, cond_list=None, return_list=False, update_emas=update_emas,
                                                       **synthesis_kwargs)
        static_feat = self.backbone.synthesis(ws, cond_list=None, return_list=False, update_emas=update_emas,
                                              **synthesis_kwargs)

        static_plane = static_feat
        static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        static_plane_face = static_plane[:, 0]
        # texture_feats = self.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=update_emas,
        #                                                 **synthesis_kwargs)
        #
        # static_feats = self.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=update_emas,
        #                                        **synthesis_kwargs)
        # static_plane = static_feats[-1]
        # static_plane = static_plane.view(len(static_plane), 3, 32, static_plane.shape[-2], static_plane.shape[-1])
        # static_feats[0] = static_feats[0].view(len(static_plane), 3, 32, static_feats[0].shape[-2],
        #                                        static_feats[0].shape[-1])[:, 0]
        # static_feats[-1] = static_plane[:, 0]
        # assert len(static_feats) == len(texture_feats)
        bbox_256 = [57, 185, 64, 192]
        # rendering_images, full_alpha_image, mouth_masks, rendering_images_nomask = self.rasterize(texture_feats,
        #                                                                                           mesh_condition[
        #                                                                                               'uvcoords_image'],
        #                                                                                           static_feats,
        #                                                                                           bbox_256)
        rendering_image, full_alpha_image, rendering_image_only_img, mask_images = self.rasterize_sinle_input(texture_feat,
                                                                                                 mesh_condition[
                                                                                                     'uvcoords_image'],
                                                                                                 static_plane_face,
                                                                                                 bbox_256)
        if self.use_norefine_rgb:
            rendering_stitch = rendering_image_only_img
        else:

            rendering_images_no_masks = self.triplnae_encoder(rendering_image)
            rendering_images = []
            for index, rendering_image_no_mask in enumerate(rendering_images_no_masks):
                rendering_images_each = torch.cat([rendering_image_no_mask, mask_images[index]], dim=1)
                rendering_images.append(rendering_images_each)
            rendering_images.append(rendering_image)
            rendering_stitch = self.face_backbone.synthesis(ws, rendering_images, return_list=False,
                                                            update_emas=update_emas, **synthesis_kwargs)

        rendering_stitch_, full_alpha_image_ = torch.zeros_like(rendering_stitch), torch.zeros_like(full_alpha_image)
        rendering_stitch_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(rendering_stitch,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image_[:, :, bbox_256[0]:bbox_256[1], bbox_256[2]:bbox_256[3]] = F.interpolate(full_alpha_image,
                                                                                                  size=(128, 128),
                                                                                                  mode='bilinear',
                                                                                                  antialias=True)
        full_alpha_image, rendering_stitch = full_alpha_image_, rendering_stitch_

        # blend features of neural texture and tri-plane
        full_alpha_image = torch.cat(
            (full_alpha_image, torch.zeros_like(full_alpha_image), torch.zeros_like(full_alpha_image)), 1).unsqueeze(2)
        rendering_stitch = torch.cat(
            (rendering_stitch, torch.zeros_like(rendering_stitch), torch.zeros_like(rendering_stitch)), 1)
        rendering_stitch = rendering_stitch.view(*static_plane.shape)
        blended_planes = rendering_stitch * full_alpha_image + static_plane * (1 - full_alpha_image)

        return self.renderer.run_model(blended_planes, self.decoder, coordinates, directions, self.rendering_kwargs)

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


from training.networks_stylegan2 import FullyConnectedLayer


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'],
                                lr_multiplier=options['decoder_lr_mul'])
        )

    def forward(self, sampled_features, ray_directions, sampled_embeddings=None):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


# Define Simple Encoder
from training_avatar_texture.networks_stylegan2_styleunet_next3d import EncoderResBlock


class EncoderTriplane(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.encoder = torch.nn.ModuleList()
        config_lists = [
            [64, 128, 1, 1],
            [128, 256, 2, 1],
            [256, 512, 2, 2],
            [512, 512, 2, 4],
            [512, 32, 1, 8],

        ]
        for config_list in config_lists:
            block = EncoderResBlock(33, config_list[0], config_list[1], down=config_list[2], downsample=config_list[3])
            self.encoder.append(block)

    def forward(self, init_input):
        # obtain multi-scale content features
        cond_list = []
        cond_out = None
        x_in = init_input
        for i, _ in enumerate(self.encoder):
            x_in, cond_out = self.encoder[i](x_in, cond_out)
            cond_list.append(cond_out)

        cond_list = cond_list[::-1]

        return cond_list

# class TriplaneEncoder(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         Conv2dLayer(32, 32, kernel_size=1, bias=False, down=8)
#         self.conv_1 = Conv2dLayer(32, 32, kernel_size=1, bias=False, down=8)
#         self.conv_2 = Conv2dLayer(32, 512, kernel_size=1, bias=False, down=8)
#         self.conv_3 = Conv2dLayer(32, 512, kernel_size=1, bias=False, down=4)
#         self.conv_4 = Conv2dLayer(32, 256, kernel_size=1, bias=False, down=2)
#         self.conv_5 = Conv2dLayer(32, 128, kernel_size=1, bias=False )
#
#
#     def forward(self, feature_input):
#         # Aggregate features
#         sampled_features_1 = self.conv_1(feature_input)
#         sampled_features_2 = self.conv_2(feature_input)
#         sampled_features_3 = self.conv_3(feature_input)
#         sampled_features_4 = self.conv_4(feature_input)
#         sampled_features_5 = self.conv_5(feature_input)
#         return [sampled_features_1, sampled_features_2, sampled_features_3, sampled_features_4, sampled_features_5, feature_input]
