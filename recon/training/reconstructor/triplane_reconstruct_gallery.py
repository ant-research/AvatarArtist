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
import torch.nn.functional as F
import dnnlib
from torch_utils import persistence
from einops import rearrange

from recon.training.generator.triplane_v20_original import OSGDecoder

from recon.training.reconstructor.networks_reconstructor import EncoderGlobal, EncoderDetail, EncoderCanonical, \
    DecoderTriplane
from recon.training.reconstructor.triplane_ae import Encoder as TriEncoder
from recon.volumetric_rendering.renderer import ImportanceRenderer, ImportanceRenderer_bsMotion
from recon.volumetric_rendering.ray_sampler import RaySampler, RaySampler_zxc
from recon.volumetric_rendering.renderer import fill_mouth
from Next3d.training_avatar_texture.networks_stylegan2_new import Generator as StyleGAN2Backbone_cond


# Animatable triplane reconstructor Psi in Portrait4D
@persistence.persistent_class
class TriPlaneReconstructorNeutralize(torch.nn.Module):
    def __init__(self,
                 img_resolution=512,
                 mot_dims=512,
                 w_dim=512,
                 sr_num_fp16_res=0,
                 has_background=False,
                 has_superresolution=True,
                 flame_full=True,
                 masked_sampling=False,
                 num_blocks_neutral=4,
                 num_blocks_motion=4,
                 motion_map_layers=2,
                 neural_rendering_resolution=64,
                 deformation_kwargs={},
                 rendering_kwargs={},
                 sr_kwargs={},
                 encoder_pre_weights=None,
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()

        self.mot_dims = mot_dims
        self.motion_map_layers = motion_map_layers
        self.encoder_global = EncoderGlobal(encoder_weights=encoder_pre_weights)
        self.encoder_detail = EncoderDetail()
        self.encoder_global_latent_tri = TriEncoder(n_hiddens=64, image_channel=32, z_channels=128,
                                                    downsample=[4, 4, 4])
        self.encoder_canonical = EncoderCanonical(num_blocks_neutral=num_blocks_neutral,
                                                  num_blocks_motion=num_blocks_motion, mot_dims=mot_dims,
                                                  mapping_layers=motion_map_layers)
        self.generator_triplane = DecoderTriplane()

        self.renderer = ImportanceRenderer_bsMotion()
        self.ray_sampler = RaySampler_zxc()

        decoder_output_dim = 32 if has_superresolution else 3
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32,
                                                                   img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res,
                                                                   sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)

        self.has_superresolution = True
        self.img_resolution = img_resolution
        #
        # if self.has_superresolution:
        #     superres_module_name = rendering_kwargs['superresolution_module'].replace('training.superresolution',
        #                                                                               'models.stylegan.superresolution')
        #     self.superresolution = dnnlib.util.construct_class_by_name(class_name=superres_module_name, channels=32,
        #                                                                img_resolution=img_resolution,
        #                                                                sr_num_fp16_res=sr_num_fp16_res,
        #                                                                sr_antialias=rendering_kwargs['sr_antialias'],
        #                                                                **sr_kwargs)
        # else:
        #     self.superresolution = None
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                                       'decoder_output_dim': decoder_output_dim})



        self.neural_rendering_resolution = neural_rendering_resolution
        self.rendering_kwargs = rendering_kwargs
        z_dim = 512
        w_dim = 512
        c_dim = 25
        synthesis_kwargs = {'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only', 'num_fp16_res': 0,
         'conv_clamp': None}
        mapping_kwargs = {'num_layers': 2}
        self.face_backbone = StyleGAN2Backbone_cond(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32,
                                                    mapping_kwargs= mapping_kwargs, use_tanh=False,
                                                    **synthesis_kwargs)
        self.triplnae_encoder = EncoderTriplane()
    def synthesis(self, imgs_app, imgs_mot, motions_app, motions, c, mesh, latent_recon, triplane_recon, ws_avg,
                  neural_rendering_resolution=None,
                  use_cached_backbone=False, motion_scale=1.0,  **synthesis_kwargs):
        triplane_recon_input = self.get_triplane(ws_avg, triplane_recon, mesh)
        cam = c
        cam2world_matrix = cam[:, :16].view(-1, 4, 4)
        intrinsics = cam[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
        # print(self.neural_rendering_resolution)

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            features_global = self.encoder_global(imgs_app)
            features_detail = self.encoder_detail(imgs_app)
            cano_tri_ref = rearrange(triplane_recon_input, "b f c h w -> b c f h w")
            cano_global = self.encoder_global_latent_tri(cano_tri_ref)
            cano_global = rearrange( cano_global, "b c f h w -> b (c f) h w")
            features_canonical = self.encoder_canonical(features_global, cano_global, motions, motions_app,
                                                        scale=motion_scale)

            features_canonical_lr = features_canonical[0]
            features_canonical_sr = features_canonical[1]
            triplane_recon_ref = rearrange(triplane_recon_input, "b f c h w -> b (f c) h w")
            planes = self.generator_triplane(features_canonical_sr, features_detail, triplane_recon_ref)
            planes = planes.view(len(planes), -1, 32, planes.shape[-2], planes.shape[-1])
            # Reshape output into three 32-channel planes
            # if not isinstance(planes, list):
            #     planes = [planes]
            # planes = [p.view(len(p), -1, 32, p.shape[-2], p.shape[-1]) for p in planes]
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins,
                                                                        ray_directions,
                                                                        self.rendering_kwargs, evaluation=False)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        ws_avg = ws_avg.repeat(rgb_image.shape[0], 1, 1)
        sr_image = self.superresolution(rgb_image, feature_image, ws_avg,
                                        noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                                        **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if
                                           k != 'noise_mode'})

        out = {'image_sr': sr_image, 'image': rgb_image, 'image_depth': depth_image,
               'image_feature': feature_image, 'triplane': planes}
        return out  # static_plane, 'texture_map': texture_feats[-2]}
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
    def get_triplane(self, ws, triplane, mesh_condition):
        b = triplane.shape[0]
        ws = ws.repeat(b, 1, 1)
        # Create a batch of rays for volume rendering

        # Create triplanes by running StyleGAN backbone

        static_plane = triplane[:, 1:, :, :, :]
        static_plane_face = static_plane[:, 0]

        bbox_256 = [57, 185, 64, 192]  # the face region is the center-crop result from the frontal triplane.

        texture_feat = triplane[:, 0:1, :, :, :].squeeze(1)
        rendering_image, full_alpha_image, rendering_image_only_img, mask_images = self.rasterize_sinle_input(
            texture_feat,
            mesh_condition,
            static_plane_face,
            bbox_256
        )
        rendering_images_no_masks = self.triplnae_encoder(rendering_image)
        rendering_images = []
        for index, rendering_image_no_mask in enumerate(rendering_images_no_masks):
            rendering_images_each = torch.cat([rendering_image_no_mask, mask_images[index]], dim=1)
            rendering_images.append(rendering_images_each)
        rendering_images.append(rendering_image)
        rendering_stitch = self.face_backbone.synthesis(ws, rendering_images, return_list=False )

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
        return blended_planes
    def sample_mixed(self, imgs_app, imgs_mot, mesh, ws_avg, motions_app, motions, coordinates, directions, latent_recon,
                     triplane_recon, motion_scale=1.0, **synthesis_kwargs):
        triplane_recon_input = self.get_triplane(ws_avg, triplane_recon, mesh)

        features_global = self.encoder_global(imgs_app)
        features_detail = self.encoder_detail(imgs_app)
        cano_tri_ref = rearrange(triplane_recon_input, "b f c h w -> b c f h w")
        cano_global = self.encoder_global_latent_tri(cano_tri_ref)
        cano_global = rearrange(cano_global, "b c f h w -> b (c f) h w")
        features_canonical = self.encoder_canonical(features_global, cano_global, motions, motions_app,
                                                    scale=motion_scale)

        features_canonical_lr = features_canonical[0]
        features_canonical_sr = features_canonical[1]
        triplane_recon_ref = rearrange(triplane_recon_input, "b f c h w -> b (f c) h w")
        planes = self.generator_triplane(features_canonical_sr, features_detail, triplane_recon_ref)
        planes = planes.view(len(planes), -1, 32, planes.shape[-2], planes.shape[-1])


        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, imgs_app, imgs_mot, motions_app, motions,
                c,  mesh, triplane_recon, ws_avg, neural_rendering_resolution=None,  motion_scale=1.0, **synthesis_kwargs):

        img_dict = self.synthesis(imgs_app, imgs_mot, motions_app, motions, c, mesh, triplane_recon, triplane_recon, ws_avg,
                             neural_rendering_resolution=neural_rendering_resolution,
                             motion_scale=motion_scale,
                             **synthesis_kwargs)

        return img_dict

from Next3d.training_avatar_texture.networks_stylegan2_styleunet_next3d import EncoderResBlock
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