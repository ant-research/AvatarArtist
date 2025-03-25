import torch
import numpy as np
import os

from data_process.lib.FaceVerse import get_recon_model
from pytorch3d.structures import Meshes
from .ortho_renderer import get_renderer, batch_orth_proj, angle2matrix, face_vertices, render_after_rasterize


class Faceverse_manager(object):
    def __init__(self, device, base_coeff):
        render_res = 512
        self.ortho_renderer = get_renderer(img_size=render_res, device=device, T=torch.tensor([[0, 0, 10.]], dtype=torch.float32, device=device),
                                           K=[-1.0, -1.0, 0., 0.], orthoCam=True, rasterize_blur_radius=1e-6)

        orth_scale, orth_shift, box_warp = 5.00, np.asarray([0, 0.005, 0.], dtype=np.float32), 2.
        self.orth_scale, self.orth_shift = orth_scale, torch.from_numpy(orth_shift).to(device).unsqueeze(0)
        face_model_dir = 'data_process/lib/FaceVerse/v3'
        self.recon_model, model_dict = get_recon_model(model_path=os.path.join(face_model_dir, 'faceverse_v3_1.npy'), return_dict=True, device='cuda:0')

        vert_uvcoords = model_dict['uv_per_ver']
        if True:  # 扩大face部分在UV图中占据的面积
            vert_idx = (vert_uvcoords[:, 1] > 0.273) * (vert_uvcoords[:, 1] < 0.727) * (vert_uvcoords[:, 0] > 0.195) * (vert_uvcoords[:, 0] < 0.805)
            vert_uvcoords[vert_idx] = (vert_uvcoords[vert_idx] - 0.5) * 1.4 + 0.5

        vert_uvcoords = torch.from_numpy(vert_uvcoords).unsqueeze(0)

        vert_mask = np.load(os.path.join(face_model_dir, 'v31_face_mask_new.npy'))
        vert_mask[model_dict['ver_inds'][0]:model_dict['ver_inds'][2]] = 1

        vert_mask = torch.from_numpy(vert_mask).view(1, -1, 1)
        vert_uvcoords = torch.cat([vert_uvcoords * 2 - 1, vert_mask.clone()], -1).to(device)  # [bz, ntv, 3]
        self.face_uvcoords = face_vertices(vert_uvcoords, self.recon_model.tri.unsqueeze(0))  # 面片不反向
        # vert_mask[0, ~vert_idx] *= 0  # for UV rendering

        self.tform = angle2matrix(torch.tensor([0, 0, 0]).reshape(1, -1)).to(device)
        self.cam = torch.tensor([1., 0, 0]).cuda()
        self.trans_init = torch.from_numpy(np.load(os.path.join(face_model_dir, 'fv2fl_30.npy'))).float().to(device)
        self.crop_param = [128, 114, 256, 256]
        if base_coeff is not None:
            assert isinstance(base_coeff, torch.Tensor) and base_coeff.ndim==1
            self.id_coeff, self.base_avatar_exp_coeff = self.recon_model.split_coeffs(base_coeff.to(device).unsqueeze(0))[:2]

    def make_driven_rendering(self, drive_coeff, base_drive_coeff=None, res=None):
        assert drive_coeff.ndim == 2
        _, exp_coeff, _, _, _, _, eye_coeff, _ = self.recon_model.split_coeffs(drive_coeff)
        exp_coeff[:, -4] = max(min(exp_coeff[:, -4], 0.6), -0.75)
        exp_coeff[:, -2] = max(min(exp_coeff[:, -2], 0.75), -0.75)
        if base_drive_coeff is not None:
            base_drive_exp_coeff = self.recon_model.split_coeffs(base_drive_coeff)[1]
            delta_exp_coeff = exp_coeff - base_drive_exp_coeff
            exp_coeff = delta_exp_coeff + self.base_avatar_exp_coeff

        l_eye_mat = self.recon_model.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_mat = self.recon_model.compute_eye_rotation_matrix(eye_coeff[:, 2:])
        l_eye_mean = self.recon_model.get_l_eye_center(self.id_coeff)
        r_eye_mean = self.recon_model.get_r_eye_center(self.id_coeff)

        vs = self.recon_model.get_vs(self.id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean)
        vert = torch.matmul(vs[0], self.trans_init[:3, :3].T) + self.trans_init[:3, 3:].T

        v = vert.unsqueeze(0)
        transformed_vertices = (torch.bmm(v, self.tform) + self.orth_shift) * self.orth_scale
        transformed_vertices = batch_orth_proj(transformed_vertices, self.cam)
        transformed_vertices[..., -1] *= -1

        mesh = Meshes(transformed_vertices, self.recon_model.tri.unsqueeze(0))
        fragment = self.ortho_renderer.rasterizer(mesh)
        rendering = render_after_rasterize(attributes=self.face_uvcoords, pix_to_face=fragment.pix_to_face,
                                           bary_coords=fragment.bary_coords)  # [1, 4, H, W]
        render_mask = rendering[:, -1:, :, :].clone()
        render_mask *= rendering[:, -2:-1]  # face_mask
        rendering *= render_mask

        if self.crop_param is not None:  # [left, top, width, height]
            rendering = rendering[:, :, self.crop_param[1]:self.crop_param[1] + self.crop_param[3], self.crop_param[0]:self.crop_param[0] + self.crop_param[2]]
        if not ((res is None) or res == rendering.shape[2]):
            rendering = torch.nn.functional.interpolate(rendering, size=(res, res), mode='bilinear', align_corners=False)
        # np.save(os.path.join(dst_sub_dir, name + '.npy'), rendering[0].permute(1, 2, 0).cpu().numpy().astype(np.float16))
        uvcoords_image = rendering.permute(0, 2, 3, 1)[..., :3]
        uvcoords_image[..., -1][uvcoords_image[..., -1] < 0.5] = 0; uvcoords_image[..., -1][uvcoords_image[..., -1] >= 0.5] = 1

        return uvcoords_image
