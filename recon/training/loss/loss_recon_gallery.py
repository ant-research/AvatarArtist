# Loss for Portrait4D, modified from EG3D: https://github.com/NVlabs/eg3d

# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms as transform
# from kornia.geometry import warp_affine
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from recon.training.discriminator.dual_discriminator_next3D import filtered_resizing
import cv2
from PIL import Image
# from recon.utils.preprocess import estimate_norm_torch, estimate_norm_torch_pdfgc
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics


# ----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img_app, real_img_mot, real_img_recon, real_depth_recon,
                             real_feature_recon, real_triplane_recon,
                             real_c_recon, motions_app, motions, gain, cur_nimg, real_cano_tri, real_ref_tri,
                             motion_scale=1.0, swapping_prob=0.5, half_static=False):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class AnimatableGalleryPortraitReconLoss(Loss):
    def __init__(self, device, G, D, G_syn, D_patch=None, augment_pipe=None, lpips=None, facenet=None, pd_fgc=None,
                  gmain=1.0, r1_gamma=10, r1_gamma_patch=10, r1_gamma_uv=30,
                 r1_gamma_seg=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01,
                 pl_no_weight_grad=False, blur_init_sigma=0, blur_init_sigma_patch=0, blur_fade_kimg=0,
                 blur_patch_seg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64,
                 neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0,
                 gpc_reg_fade_kimg=1000, gpc_reg_prob=None, discrimination_kimg=1000, dual_discrimination=False,
                 filter_mode='antialiased', patch_scale=1.0, patch_gan=0.2, masked_sampling=None, perturb_params=False, id_loss=None,
                 use_D=True, truncation_psi=0.7, conditioning_params=None, w_avg=None):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.G_syn = G_syn
        self.D_patch = D_patch
        self.augment_pipe = augment_pipe
        self.lpips = lpips
        self.pd_fgc = pd_fgc
        self.gmain = gmain
        self.r1_gamma = r1_gamma
        self.r1_gamma_patch = r1_gamma_patch
        self.r1_gamma_uv = r1_gamma_uv
        self.r1_gamma_seg = r1_gamma_seg
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_init_sigma_patch = blur_init_sigma_patch
        self.blur_fade_kimg = blur_fade_kimg
        self.blur_patch_seg = blur_patch_seg
        self.r1_gamma_init = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.bg_reg = True
        self.c_headpose = False
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.discrimination_kimg = discrimination_kimg
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        self.patch_scale = patch_scale
        self.masked_sampling = masked_sampling
        self.patch_gan = patch_gan
        self.perturb_params = perturb_params
        self.use_D = use_D
        self.truncation_psi = truncation_psi
        self.conditioning_params = torch.load(conditioning_params ).to(device)
        self.w_avg = torch.load(w_avg).to(device)[0]

        self.id_loss = id_loss.to(device)

    # extract pdfgc motion embedding
    # def get_motion_feature(self, imgs, lmks, crop_size=224, crop_len=16):
    #
    #     trans_m = estimate_norm_torch_pdfgc(lmks, imgs.shape[-1])
    #     imgs_warp = warp_affine(imgs, trans_m, dsize=(224, 224))
    #     imgs_warp = imgs_warp[:, :, :crop_size - crop_len * 2, crop_len:crop_size - crop_len]
    #     imgs_warp = torch.clamp(F.interpolate(imgs_warp, size=[crop_size, crop_size], mode='bilinear'), -1, 1)
    #
    #     out = self.pd_fgc(imgs_warp)
    #     motions = torch.cat([out[1], out[2], out[3]], dim=-1)
    #
    #     return motions
    # generate online training data using pre-trained Next3d model. the first frame is fixed
    @torch.no_grad()
    def gen_data_by_G_next3D(self, z,  exp_params, c, model_name_list,
                             render_res=64):
        out_sr_img = []
        out_img_raw = []
        out_img_depth = []
        out_static_plane = []
        out_feature_img = []
        out_triplane = []
        out_texture = []
        out_rendering_stitch = []
        batchsize_subject = z.shape[0]
        # assert batchsize_subject == 3
        # cam_pivot = torch.tensor(self.G_syn.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        # cam_radius = self.G_syn.rendering_kwargs.get('avg_camera_radius', 2.7)
        # conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
        #                                                        device=device)
        # conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(
        #     device)
        # w = G.mapping(z, conditioning_params, truncation_psi=0.7, truncation_cutoff=14)
        # latent = latent.reshape(-1, *latent.shape[2:])
        assert len(model_name_list) == z.shape[0]
        model_name_list = [name for name1 in model_name_list for name in name1 ]
        z = z.reshape(-1, *z.shape[2:])  # (b*3, 512)
        assert len(model_name_list) == z.shape[0]

        exp_params = exp_params.reshape(-1, *exp_params.shape[2:])  # (b*3, 100)
        c = c.reshape(-1, *c.shape[2:])  # (b*3, 25)
        # out_motion = [real_motion_1, real_motion_2]
        # random head rotation
        angle_ys_head = torch.rand((z.shape[0], 1), device=z.device) * 0.60 * 2 - 0.60
        angle_ys_head2 = torch.rand((z.shape[0], 1), device=z.device) * 0.35 * 2 - 0.35 + 0.2
        # angle_ys_head3 = torch.rand((z.shape[0], 1), device=z.device) * 0.25 * 2 - 0.25

        # random camera pose
        cam_pivot_x = torch.rand((z.shape[0], 1), device=z.device) * 0.02 - 0.01
        cam_pivot_y = torch.rand((z.shape[0], 1), device=z.device) * 0.02 - 0.01
        cam_pivot_z = torch.rand((z.shape[0], 1), device=z.device) * 0.02 - 0.01 + 0.03
        cam_pivot = torch.cat([cam_pivot_x * 3, cam_pivot_y * 3, cam_pivot_z * 3], dim=-1)
        # cam_radius = torch.rand((z.shape[0], 1), device=z.device) * 0.8 + 2.7
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_ys_head, np.pi / 2 - angle_ys_head2, cam_pivot,
                                                  radius=2.7,
                                                  batch_size=z.shape[0], device=z.device)
        c_syn = torch.cat([cam2world_pose.reshape(-1, 16), c[:, 16:].reshape(-1, 9)], dim=-1)

        prob = torch.rand((c.shape[0], 1), device=c.device)
        c_syn = c_syn
        c_syn_final_out = c_syn.reshape(-1, 3, c_syn.shape[-1])
        for batch_index, model_name in enumerate(model_name_list):
            z_value = z[batch_index].unsqueeze(0)

            ws = self.G_syn[model_name].mapping(z_value, self.conditioning_params, truncation_psi=self.truncation_psi,
                                                truncation_cutoff=14)
            vert_value = exp_params[batch_index].unsqueeze(0)
            c = c_syn[batch_index].unsqueeze(0)
            out = self.G_syn[model_name].synthesis(ws, c, vert_value, noise_mode='const',
                                                   neural_rendering_resolution=128, return_featmap=True
                                                   )
            # img = (out['image'][0] * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # img = img.permute(1, 2, 0)
            # img = img.cpu().numpy()
            # img = Image.fromarray(np.uint8(img))
            # print('savesavesavesavesavesave')
            # save_dir = f'/home/liuhongyu/code/HeadArtist2/HeadGallery/training-runs-portrait4d/00073--multi_style-gpus2-batch8/{batch_index}.png'
            # print(save_dir)
            # # cv2.imwrite(save_dir, img)
            # img.save(save_dir)
            out_sr_img.append(out['image'])
            out_img_raw.append(out['image_raw'])
            out_img_depth.append(out['image_depth'])
            out_static_plane.append(out['static_plane'])
            out_feature_img.append(out['image_feature'])
            out_triplane.append(out['triplane'])

            out_rendering_stitch.append(out['rendering_stitch'])

        final_out = {'image_sr': torch.cat(out_sr_img), 'image': torch.cat(out_img_raw),
                     'image_depth': torch.cat(out_img_depth), 'static_plane': torch.cat(out_static_plane),
                     'image_feature': torch.cat(out_feature_img), 'triplane': torch.cat(out_triplane),
                     'rendering_stitch': torch.cat(out_rendering_stitch),
                     'c': c_syn_final_out,
                     # 'motions': out_motion
                     }
        return final_out

    def run_G(self, imgs_app, imgs_mot, motions_app, motions, c, mesh, real_cano_tri, real_ref_tri,
              neural_rendering_resolution,
              motion_scale=1.0, swapping_prob=0.5, half_static=False):

        motion_scale = torch.ones([imgs_app.shape[0], 1, 1], device=c.device) * motion_scale
        if swapping_prob is not None:
            imgs_app_swapped = imgs_mot
            prob = torch.rand((imgs_app.shape[0], 1), device=c.device)
            imgs_app_conditioning = torch.where(prob.reshape(imgs_app.shape[0], 1, 1, 1) < swapping_prob,
                                                imgs_app_swapped, imgs_app)
            motion_scale_conditioning = torch.where(prob.reshape(imgs_app.shape[0], 1, 1) < swapping_prob,
                                                    torch.zeros_like(motion_scale), motion_scale)
            motions_app_conditioning = torch.where(prob < swapping_prob, motions, motions_app)
        else:
            imgs_app_conditioning = imgs_app
            motion_scale_conditioning = motion_scale
            motions_app_conditioning = motions_app

        # whether or not the second half of the batchsize are static data
        # If true, set motion scale to zero to deactivate motion-related cross-attention layers. 
        if half_static:
            num_static = imgs_app.shape[0] // 2
            if swapping_prob is None:
                motion_scale_conditioning = torch.cat([motion_scale[:num_static], motion_scale[num_static:] * 0], dim=0)
            else:
                prob = torch.rand((num_static, 1), device=c.device)
                motion_scale_static = torch.where(prob.reshape(num_static, 1, 1) < 1 - swapping_prob,
                                                  torch.zeros_like(motion_scale[num_static:]),
                                                  motion_scale[num_static:])
                motion_scale_conditioning = torch.cat([motion_scale_conditioning[:num_static], motion_scale_static],
                                                      dim=0)

        gen_output = self.G.synthesis(imgs_app_conditioning, imgs_mot, motions_app_conditioning, motions, c, mesh,
                                      real_ref_tri, real_ref_tri, self.w_avg,
                                      neural_rendering_resolution=neural_rendering_resolution,
                                      motion_scale=motion_scale_conditioning)

        return gen_output

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                if self.G.has_superresolution:
                    f = torch.arange(-blur_size, blur_size + 1, device=img['image_sr'].device).div(
                        blur_sigma).square().neg().exp2()
                    img['image_sr'] = upfirdn2d.filter2d(img['image_sr'], f / f.sum())
                else:
                    f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(
                        blur_sigma).square().neg().exp2()
                    img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img_app, real_img_mot, real_img_recon, real_depth_recon,
                             real_feature_recon, real_triplane_recon,
                             real_c_recon,  mesh, motions_app, motions, gain, cur_nimg, real_cano_tri, real_ref_tri,
                             motion_scale=1.0, swapping_prob=0.5, half_static=True ):

        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        # if self.r1_gamma_patch == 0:
        #     phase = {'D_patchreg': 'none', 'D_patchboth': 'Dmain'}.get(phase, phase)

        blur_sigma = 0
        r1_gamma = self.r1_gamma
        # r1_gamma_patch = self.r1_gamma_patch
        # r1_gamma_uv = self.r1_gamma_uv
        # r1_gamma_seg = self.r1_gamma_seg

        if self.neural_rendering_resolution_final is not None:
            alpha = min(
                max((cur_nimg - self.discrimination_kimg * 1e3) / (self.neural_rendering_resolution_fade_kimg * 1e3),
                    0), 1)  # begin fading when D starts to be optimized
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (
                    1 - alpha) + self.neural_rendering_resolution_final * alpha))
            neural_rendering_resolution_patch = self.neural_rendering_resolution_final
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial
            neural_rendering_resolution_patch = neural_rendering_resolution

        if self.G.has_superresolution:
            real_img_raw = filtered_resizing(real_img_recon, size=neural_rendering_resolution, f=self.resample_filter,
                                             filter_mode=self.filter_mode)

            if self.blur_raw_target and blur_sigma > 0:
                blur_size = np.floor(blur_sigma * 3)
                if blur_size > 0:
                    f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(
                        blur_sigma).square().neg().exp2()
                    real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

            real_img = {'image_sr': real_img_recon, 'image': real_img_raw,
                        }
        else:
            real_img = {'image': real_img_recon}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(real_img_app, real_img_mot, motions_app, motions, real_c_recon, mesh, real_cano_tri, real_ref_tri,
                                     neural_rendering_resolution=neural_rendering_resolution,
                                     motion_scale=motion_scale, swapping_prob=swapping_prob, half_static=half_static)

                # main image-level reconstruction loss
                gen_img_recon = gen_img['image_sr']
                gen_img_recon_raw = gen_img['image']
                gen_depth = gen_img['image_depth']
                gen_feature = gen_img['image_feature']
                gen_triplane_recon = gen_img['triplane']

                loss_recon_lpips = self.lpips(gen_img_recon, real_img_recon) + self.lpips(gen_img_recon_raw,
                                                                                          real_img_raw)

                training_stats.report('Loss/G/lrecon_lpips', loss_recon_lpips)

                loss_recon_l1 = torch.abs(gen_img_recon - real_img_recon).mean() + torch.abs(
                    gen_img_recon_raw - real_img_raw).mean()

                training_stats.report('Loss/G/lrecon_l1', loss_recon_l1)

                # use id loss after seeing 400k images
                if cur_nimg < 400 * 1e3:
                    loss_id = 0
                else:
                    loss_id = self.id_loss(gen_img_recon, real_img_recon)
                    training_stats.report('G_Loss/real/loss_id', loss_id)

                # use depth loss before seeing 400k images
                if real_depth_recon is not None:
                    if real_depth_recon.shape != gen_depth.shape:
                        real_depth_recon = F.interpolate(real_depth_recon, size=[gen_depth.shape[2], gen_depth.shape[3]],                                                                                                   mode='bilinear',
                                                                                                  antialias=True)
                    loss_recon_depth = torch.abs(
                        (real_depth_recon - gen_depth)).mean()
                    training_stats.report('Loss/G/lrecon_depth', loss_recon_depth)
                else:
                    loss_recon_depth = 0.

                # use feature map loss before seeing 400k images
                if real_feature_recon is not None:
                    if real_feature_recon.shape != gen_feature.shape:
                        real_feature_recon = F.interpolate(real_feature_recon, size=[gen_feature.shape[2], gen_feature.shape[3]],                                                                                                   mode='bilinear',
                                                                                                  antialias=True)
                    loss_recon_feature = torch.abs(real_feature_recon - gen_feature).mean()
                    training_stats.report('Loss/G/lrecon_feature', loss_recon_feature)
                else:
                    loss_recon_feature = 0.

                # use triplane feature loss before seeing 400k images
                if real_triplane_recon is not None:
                    loss_recon_triplane = torch.abs(real_triplane_recon - gen_triplane_recon).mean()
                    training_stats.report('Loss/G/lrecon_triplane', loss_recon_triplane)
                else:
                    loss_recon_triplane = 0.

                loss_recon = loss_recon_lpips + loss_recon_l1 + loss_recon_depth + loss_recon_feature + loss_recon_triplane*0.1  + loss_id

                # adversarial loss after warm-up stage
                if cur_nimg >= self.discrimination_kimg * 1e3 and self.use_D:
                    gen_logits = self.run_D(gen_img,  real_c_recon, blur_sigma=blur_sigma)
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    training_stats.report('Loss/G/loss', loss_Gmain)
                else:
                    loss_Gmain = None

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_G = loss_recon.mean()
                if loss_Gmain is not None:
                    loss_G += loss_Gmain.mean() * self.gmain
                loss_G.mul(gain).backward()

        # # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs[
            'reg_type'] == 'l1':

            initial_coordinates = torch.rand((real_c_recon.shape[0], 1000, 3), device=real_c_recon.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * \
                                    self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)

            motion_scale = torch.ones([real_img_app.shape[0], 1, 1], device=real_img_app.device) * motion_scale
            if swapping_prob is not None:
                real_img_app_swapped = real_img_mot
                prob = torch.rand((real_img_app.shape[0], 1), device=real_img_app.device)
                real_img_app_conditioning = torch.where(prob.reshape(real_img_app.shape[0], 1, 1, 1) < swapping_prob,
                                                        real_img_app_swapped, real_img_app)
                motion_scale_conditioning = torch.where(prob.reshape(real_img_app.shape[0], 1, 1) < swapping_prob,
                                                        torch.zeros_like(motion_scale), motion_scale)
                motions_app_conditioning = torch.where(prob < swapping_prob, motions, motions_app)
            else:
                real_img_app_conditioning = real_img_app
                motion_scale_conditioning = motion_scale
                motions_app_conditioning = motions_app

            if half_static:
                num_static = real_img_app.shape[0] // 2
                if swapping_prob is None:
                    motion_scale_conditioning = torch.cat([motion_scale[:num_static], motion_scale[num_static:] * 0],
                                                          dim=0)
                else:
                    prob = torch.rand((num_static, 1), device=real_img_app.device)
                    motion_scale_static = torch.where(prob.reshape(num_static, 1, 1) < 1 - swapping_prob,
                                                      torch.zeros_like(motion_scale[num_static:]),
                                                      motion_scale[num_static:])
                    motion_scale_conditioning = torch.cat([motion_scale_conditioning[:num_static], motion_scale_static],
                                                          dim=0)



            out = self.G.sample_mixed(real_img_app_conditioning, real_img_mot, mesh, self.w_avg,   motions_app_conditioning, motions,
                                      all_coordinates, torch.randn_like(all_coordinates),  real_cano_tri, real_ref_tri,
                                      motion_scale=motion_scale_conditioning)
            if isinstance(out, tuple):
                TVloss = 0
                for out_ in out:
                    sigma = out_['sigma'][:, :initial_coordinates.shape[1] * 2]
                    sigma_initial = sigma[:, :sigma.shape[1] // 2]
                    sigma_perturbed = sigma[:, sigma.shape[1] // 2:]
                    TVloss += torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs[
                        'density_reg'] / len(out)
                training_stats.report('Loss/G/TVloss', TVloss)
            else:
                sigma = out['sigma'][:, :initial_coordinates.shape[1] * 2]
                sigma_initial = sigma[:, :sigma.shape[1] // 2]
                sigma_perturbed = sigma[:, sigma.shape[1] // 2:]

                TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs[
                    'density_reg']
                training_stats.report('Loss/G/TVloss', TVloss)

            (TVloss).mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        if cur_nimg >= self.discrimination_kimg * 1e3 and self.use_D:
            loss_Dgen = 0
            if phase in ['Dmain', 'Dboth']:
                with torch.autograd.profiler.record_function('Dgen_forward'):
                    gen_img = self.run_G(real_img_app, real_img_mot, motions_app, motions, real_c_recon, mesh, real_cano_tri,
                                         real_ref_tri,
                                         neural_rendering_resolution=neural_rendering_resolution,
                                         motion_scale=motion_scale, swapping_prob=swapping_prob,
                                         half_static=half_static)

                    gen_logits = self.run_D(gen_img, real_c_recon, blur_sigma=blur_sigma, update_emas=True)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Dgen = torch.nn.functional.softplus(gen_logits)
                with torch.autograd.profiler.record_function('Dgen_backward'):
                    loss_Dgen.mean().mul(gain).backward()

            # Dmain: Maximize logits for real images.
            # Dr1: Apply R1 regularization.
            if phase in ['Dmain', 'Dreg', 'Dboth']:
                name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
                with torch.autograd.profiler.record_function(name + '_forward'):

                    real_img_tmp_image = real_img['image_sr'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_img_tmp_image_raw = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_img_tmp = {'image_sr': real_img_tmp_image, 'image': real_img_tmp_image_raw}
                    real_logits = self.run_D(real_img_tmp, real_c_recon, blur_sigma=blur_sigma)

                    training_stats.report('Loss/scores/real', real_logits)
                    training_stats.report('Loss/signs/real', real_logits.sign())

                    loss_Dreal = 0
                    if phase in ['Dmain', 'Dboth']:
                        loss_Dreal = torch.nn.functional.softplus(-real_logits)
                        training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    loss_Dr1 = 0
                    if phase in ['Dreg', 'Dboth']:
                        if self.dual_discrimination:
                            with torch.autograd.profiler.record_function(
                                    'r1_grads'), conv2d_gradfix.no_weight_gradients():
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()],
                                                               inputs=[real_img_tmp['image_sr'],
                                                                       real_img_tmp['image']], create_graph=True,
                                                               only_inputs=True)
                                r1_grads_image = r1_grads[0]
                                r1_grads_image_raw = r1_grads[1]
                            r1_penalty = r1_grads_image.square().sum([1, 2, 3]) + r1_grads_image_raw.square().sum(
                                [1, 2, 3])
                        else:  # single discrimination
                            with torch.autograd.profiler.record_function(
                                    'r1_grads'), conv2d_gradfix.no_weight_gradients():
                                if self.G.has_superresolution:
                                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()],
                                                                   inputs=[real_img_tmp['image_sr']], create_graph=True,
                                                                   only_inputs=True)
                                else:
                                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()],
                                                                   inputs=[real_img_tmp['image']], create_graph=True,
                                                                   only_inputs=True)
                                r1_grads_image = r1_grads[0]
                            r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                        loss_Dr1 = r1_penalty * (r1_gamma / 2)
                        training_stats.report('Loss/r1_penalty', r1_penalty)
                        training_stats.report('Loss/D/reg', loss_Dr1)

                with torch.autograd.profiler.record_function(name + '_backward'):
                    (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
