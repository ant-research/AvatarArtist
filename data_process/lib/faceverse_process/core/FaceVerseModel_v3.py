import os.path

import pytorch3d.transforms
import torch
from torch import nn
import numpy as np

from pytorch3d.structures import Meshes
# from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    OrthographicCameras,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)
from pytorch3d.loss import (
    # mesh_edge_loss,
    mesh_laplacian_smoothing,
    # mesh_normal_consistency,
)
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent

class MeshRendererWithDepth(MeshRenderer):
    def __init__(self, rasterizer, shader):
        super().__init__(rasterizer, shader)

    def forward(self, meshes_world, attributes=None, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        if attributes is not None:
            bary_coords, pix_to_face = fragments.bary_coords, fragments.pix_to_face.clone()

            vismask = (pix_to_face > -1).float()

            D = attributes.shape[-1]
            attributes = attributes.clone()
            attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])

            N, H, W, K, _ = bary_coords.shape
            pix_to_face[pix_to_face == -1] = 0
            idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
            pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
            pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)

            pixel_vals[pix_to_face == -1] = 0  # Replace masked values in output.
            pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
            pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
            return images, fragments.zbuf, pixel_vals
        else:
            return images, fragments.zbuf


def get_renderer(img_size, device, R=None, T=None, K=None, orthoCam=False, rasterize_blur_radius=0.):
    if orthoCam:
        fx, fy, cx, cy = K[0], K[1], K[2], K[3]
        cameras = OrthographicCameras(device=device, R=R, T=T, focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
                                     principal_point=((cx, cy),),
                                     in_ndc=True)
        # cameras = FoVOrthographicCameras(T=T, device=device)
    else:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        fx = -fx * 2.0 / (img_size - 1)
        fy = -fy * 2.0 / (img_size - 1)
        cx = - (cx - (img_size - 1) / 2.0) * 2.0 / (img_size - 1)
        cy = - (cy - (img_size - 1) / 2.0) * 2.0 / (img_size - 1)
        cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
                                     principal_point=((cx, cy),),
                                     in_ndc=True)

    lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]],
                         ambient_color=[[1, 1, 1]],
                         specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])

    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=rasterize_blur_radius,
        faces_per_pixel=1
        # bin_size=0
    )
    blend_params = blending.BlendParams(background_color=[0, 0, 0])
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        )
    )
    return renderer


class FaceVerseModel(nn.Module):
    def __init__(self, model_dict, batch_size=1, device='cuda:1', expr_52=True, **kargs):
        super(FaceVerseModel, self).__init__()

        self.batch_size = batch_size
        self.device = torch.device(device)

        self.rotXYZ = torch.eye(3).view(1, 3, 3).repeat(3, 1, 1).view(3, 1, 3, 3).to(self.device)

        self.renderer = ModelRenderer(device, **kargs)

        self.kp_inds = torch.tensor(model_dict['mediapipe_keypoints'].reshape(-1, 1), requires_grad=False).squeeze().long().to(self.device)
        self.ver_inds = model_dict['ver_inds']
        self.tri_inds = model_dict['tri_inds']

        meanshape = torch.tensor(model_dict['meanshape'].reshape(-1, 3), dtype=torch.float32, requires_grad=False, device=self.device)
        meanshape[:, [1, 2]] *= -1
        meanshape = meanshape * 0.1
        meanshape[:, 1] += 1
        self.meanshape = meanshape.reshape(1, -1)
        self.meantex = torch.tensor(model_dict['meantex'].reshape(1, -1), dtype=torch.float32, requires_grad=False, device=self.device)

        idBase = torch.tensor(model_dict['idBase'].reshape(-1, 3, 150), dtype=torch.float32, requires_grad=False, device=self.device)
        idBase[:, [1, 2]] *= -1
        self.idBase = (idBase * 0.1).reshape(-1, 150)
        self.expr_52 = expr_52
        if expr_52:
            expBase = torch.tensor(np.load(os.path.join(project_root, 'lib/faceverse_process/metamodel/v3/exBase_52.npy')).reshape(-1, 3, 52), dtype=torch.float32, requires_grad=False, device=self.device)
        else:
            expBase = torch.tensor(model_dict['exBase'].reshape(-1, 3, 171), dtype=torch.float32, requires_grad=False, device=self.device)
        expBase[:, [1, 2]] *= -1
        self.expBase = (expBase * 0.1).reshape(-1, 171)
        self.texBase = torch.tensor(model_dict['texBase'], dtype=torch.float32, requires_grad=False, device=self.device)

        self.l_eyescale = model_dict['left_eye_exp']
        self.r_eyescale = model_dict['right_eye_exp']
        # self.vert_mask = model_dict['face_mask']
        self.vert_mask = np.load(os.path.join(project_root,'lib/faceverse_process/metamodel/v31_face_mask_new.npy'))
        self.vert_mask[model_dict['ver_inds'][0]:model_dict['ver_inds'][2]] = 1
        self.vert_mask = torch.tensor(self.vert_mask).view(1, -1, 1).to(self.device)

        self.uv = torch.tensor(model_dict['uv'], dtype=torch.float32, requires_grad=False, device=self.device)
        self.tri = torch.tensor(model_dict['tri'], dtype=torch.int64, requires_grad=False, device=self.device)
        self.tri_uv = torch.tensor(model_dict['tri_uv'], dtype=torch.int64, requires_grad=False, device=self.device)
        self.point_buf = torch.tensor(model_dict['point_buf'], dtype=torch.int64, requires_grad=False, device=self.device)

        self.num_vertex = self.meanshape.shape[1] // 3
        self.id_dims = self.idBase.shape[1]     # 150
        self.tex_dims = self.texBase.shape[1]   # 251
        self.exp_dims = self.expBase.shape[1]   # 171
        self.all_dims = self.id_dims + self.tex_dims + self.exp_dims
        self.init_coeff_tensors()

        # for tracking by landmarks
        self.kp_inds_view = torch.cat([self.kp_inds[:, None] * 3, self.kp_inds[:, None] * 3 + 1, self.kp_inds[:, None] * 3 + 2], dim=1).flatten()
        self.idBase_view = self.idBase[self.kp_inds_view, :].detach().clone()
        self.expBase_view = self.expBase[self.kp_inds_view, :].detach().clone()
        self.meanshape_view = self.meanshape[:, self.kp_inds_view].detach().clone()

        # zxc
        self.identity = torch.eye(3, dtype=torch.float32, device=self.device)
        self.point_shift = torch.nn.Parameter(torch.zeros(self.num_vertex, 3, dtype=torch.float32, device=self.device))     # [N, 3]
        self.axis_angle = False

    def set_renderer(self, intr=None, img_size=256, cam_dist=10., render_depth=False, rasterize_blur_radius=0.):
        self.renderer = ModelRenderer(self.device, intr, img_size, cam_dist, render_depth, rasterize_blur_radius)

    def init_coeff_tensors(self, id_coeff=None, tex_coeff=None, exp_coeff=None, gamma_coeff=None, trans_coeff=None, rot_coeff=None, scale_coeff=None, eye_coeff=None):
        if id_coeff is None:
            self.id_tensor = torch.zeros((1, self.id_dims), dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            assert id_coeff.shape == (1, self.id_dims)
            self.id_tensor = id_coeff.clone().detach().requires_grad_(True)

        if tex_coeff is None:
            self.tex_tensor = torch.zeros((1, self.tex_dims), dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            assert tex_coeff.shape == (1, self.tex_dims)
            self.tex_tensor = tex_coeff.clone().detach().requires_grad_(True)

        if exp_coeff is None:
            self.exp_tensor = torch.zeros((self.batch_size, self.exp_dims), dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            assert exp_coeff.shape == (1, self.exp_dims)
            self.exp_tensor = exp_coeff.clone().detach().requires_grad_(True)

        if gamma_coeff is None:
            self.gamma_tensor = torch.zeros((self.batch_size, 27), dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            self.gamma_tensor = gamma_coeff.clone().detach().requires_grad_(True)

        if trans_coeff is None:
            self.trans_tensor = torch.zeros((self.batch_size, 3), dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            self.trans_tensor = trans_coeff.clone().detach().requires_grad_(True)

        if scale_coeff is None:
            self.scale_tensor = 1.0 * torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
            self.scale_tensor.requires_grad_(True)
        else:
            self.scale_tensor = scale_coeff.clone().detach().requires_grad_(True)

        if rot_coeff is None:
            self.rot_tensor = torch.zeros((self.batch_size, 3), dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            self.rot_tensor = rot_coeff.clone().detach().requires_grad_(True)

        if eye_coeff is None:
            self.eye_tensor = torch.zeros(
                (self.batch_size, 4), dtype=torch.float32,
                requires_grad=True, device=self.device)
        else:
            self.eye_tensor = eye_coeff.clone().detach().requires_grad_(True)

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :self.id_dims]  # identity(shape) coeff
        exp_coeff = coeffs[:, self.id_dims:self.id_dims + self.exp_dims]  # expression coeff
        tex_coeff = coeffs[:, self.id_dims + self.exp_dims:self.all_dims]  # texture(albedo) coeff
        angles = coeffs[:, self.all_dims:self.all_dims + 3]  # ruler angles(x,y,z) for rotation of dim 3
        gamma = coeffs[:, self.all_dims + 3:self.all_dims + 30]  # lighting coeff for 3 channel SH function of dim 27
        translation = coeffs[:, self.all_dims + 30:self.all_dims+33]  # translation coeff of dim 3
        eye_coeff = coeffs[:, self.all_dims + 33:self.all_dims + 37]  # eye coeff of dim 4
        scale = coeffs[:, -1:] if coeffs.shape[1] == self.all_dims + 38 else torch.ones_like(coeffs[:, -1:])

        return id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff, scale

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye, scale):
        coeffs = torch.cat([id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye, scale], dim=1)
        return coeffs

    def get_packed_tensors(self):
        return self.merge_coeffs(self.id_tensor,
                                 self.exp_tensor,
                                 self.tex_tensor,
                                 self.rot_tensor, self.gamma_tensor,
                                 self.trans_tensor, self.eye_tensor, self.scale_tensor)

    # def get_pytorch3d_mesh(self, coeffs, enable_pts_shift=False):
    #     id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, scale = self.split_coeffs(coeffs)
    #     rotation = self.compute_rotation_matrix(angles)
    #
    #     vs = self.get_vs(id_coeff, exp_coeff)
    #     if enable_pts_shift:
    #         vs = vs + self.point_shift.unsqueeze(0).expand_as(vs)
    #     vs_t = self.rigid_transform(vs, rotation, translation, torch.abs(scale))
    #
    #     face_texture = self.get_color(tex_coeff)
    #     face_norm = self.compute_norm(vs, self.tri, self.point_buf)
    #     face_norm_r = face_norm.bmm(rotation)
    #     face_color = self.add_illumination(face_texture, face_norm_r, gamma)
    #
    #     face_color_tv = TexturesVertex(face_color)
    #     mesh = Meshes(vs_t, self.tri.repeat(self.batch_size, 1, 1), face_color_tv)
    #
    #     return mesh

    def cal_laplacian_regularization(self, enable_pts_shift):
        current_mesh = self.get_pytorch3d_mesh(self.get_packed_tensors(), enable_pts_shift=enable_pts_shift)
        disp_reg_loss = mesh_laplacian_smoothing(current_mesh, method="uniform")
        return disp_reg_loss

    def forward(self, coeffs, render=True, camT=None, enable_gama=True, mask_face=False):
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff, scale = self.split_coeffs(coeffs)
        rotation = self.compute_rotation_matrix(angles)
        if camT is not None:
            rotation2 = camT[:3, :3].permute(1, 0).reshape(1, 3, 3)
            translation2 = camT[:3, 3:].permute(1, 0).reshape(1, 1, 3)
            if torch.allclose(rotation2, self.identity):
                translation = translation + translation2
            else:
                rotation = torch.matmul(rotation, rotation2)
                translation = torch.matmul(translation, rotation2) + translation2

        l_eye_mat = self.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_mat = self.compute_eye_rotation_matrix(eye_coeff[:, 2:])
        l_eye_mean = self.get_l_eye_center(id_coeff)
        r_eye_mean = self.get_r_eye_center(id_coeff)

        if render:
            vs = self.get_vs(id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean)
            vs_t = self.rigid_transform(vs, rotation, translation, torch.abs(scale))

            lms_t = self.get_lms(vs_t)
            lms_proj = self.renderer.project_vs(lms_t)
            face_texture = self.get_color(tex_coeff)
            face_norm = self.compute_norm(vs, self.tri, self.point_buf)
            if enable_gama:
                face_norm_r = face_norm.bmm(rotation)
                face_color = self.add_illumination(face_texture, face_norm_r, gamma)
            else:
                face_color = face_texture
            if mask_face: face_color *= self.vert_mask

            face_color_tv = TexturesVertex(face_color)
            mesh = Meshes(vs_t, self.tri.repeat(self.batch_size, 1, 1), face_color_tv)
            rendered_img = self.renderer.renderer(mesh)

            return {'rendered_img': rendered_img,
                    'lms_proj': lms_proj,
                    'face_texture': face_texture,
                    'vs': vs,
                    'vs_t': vs_t,
                    'tri': self.tri,
                    'color': face_color, 'lms_t': lms_t}
        else:
            lms = self.get_vs_lms(id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean)
            lms_t = self.rigid_transform(lms, rotation, translation, torch.abs(scale))

            lms_proj = self.renderer.project_vs(lms_t)
            return {'lms_proj': lms_proj, 'lms_t': lms_t}

    def get_vs(self, id_coeff, exp_coeff, l_eye_mat=None, r_eye_mat=None, l_eye_mean=None, r_eye_mean=None):
        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
                     torch.einsum('ij,aj->ai', self.expBase, exp_coeff) + self.meanshape
        face_shape = face_shape.view(self.batch_size, -1, 3)
        if l_eye_mat is not None:
            face_shape[:, self.ver_inds[0]:self.ver_inds[1]] = torch.matmul(face_shape[:, self.ver_inds[0]:self.ver_inds[1]] - l_eye_mean, l_eye_mat) + l_eye_mean
            face_shape[:, self.ver_inds[1]:self.ver_inds[2]] = torch.matmul(face_shape[:, self.ver_inds[1]:self.ver_inds[2]] - r_eye_mean, r_eye_mat) + r_eye_mean
        return face_shape

    def get_vs_lms(self, id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean):
        face_shape = torch.einsum('ij,aj->ai', self.idBase_view, id_coeff) + \
                     torch.einsum('ij,aj->ai', self.expBase_view, exp_coeff) + self.meanshape_view
        face_shape = face_shape.view(self.batch_size, -1, 3)
        face_shape[:, 473:478] = torch.matmul(face_shape[:, 473:478] - l_eye_mean, l_eye_mat) + l_eye_mean
        face_shape[:, 468:473] = torch.matmul(face_shape[:, 468:473] - r_eye_mean, r_eye_mat) + r_eye_mean
        return face_shape

    def get_l_eye_center(self, id_coeff):
        eye_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + self.meanshape
        eye_shape = eye_shape.view(self.batch_size, -1, 3)[:, self.ver_inds[0]:self.ver_inds[1]]
        eye_shape[:, :, 2] += 0.005
        return torch.mean(eye_shape, dim=1, keepdim=True)

    def get_r_eye_center(self, id_coeff):
        eye_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + self.meanshape
        eye_shape = eye_shape.view(self.batch_size, -1, 3)[:, self.ver_inds[1]:self.ver_inds[2]]
        eye_shape[:, :, 2] += 0.005
        return torch.mean(eye_shape, dim=1, keepdim=True)

    def get_color(self, tex_coeff):
        face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + self.meantex
        face_texture = face_texture.view(self.batch_size, -1, 3)
        return face_texture

    def compute_norm(self, vs, tri, point_buf):
        face_id = tri
        point_id = point_buf
        v1 = vs[:, face_id[:, 0], :]
        v2 = vs[:, face_id[:, 1], :]
        v3 = vs[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)

        v_norm = face_norm[:, point_id, :].sum(2)
        v_norm = v_norm / (v_norm.norm(dim=2).unsqueeze(2) + 1e-9)

        return v_norm

    def project_vs(self, vs):
        vs = torch.matmul(vs, self.reverse_z.repeat((self.batch_size, 1, 1))) + self.camera_pos
        aug_projection = torch.matmul(vs, self.p_mat.repeat((self.batch_size, 1, 1)).permute((0, 2, 1)))
        face_projection = aug_projection[:, :, :2] / torch.reshape(aug_projection[:, :, 2], [self.batch_size, -1, 1])
        return face_projection

    def make_rotMat(self, coeffes=None, angle=None, translation=None, scale=None, no_scale=False):# P * rot * scale + trans -> P * T
        if coeffes is not None:
            _, _, _, angle, _, translation, scale = self.split_coeffs(coeffes)
        rotation = self.compute_rotation_matrix(angle)

        cam_T = torch.eye(4, dtype=torch.float32).to(angle.device)
        cam_T[:3, :3] = rotation[0] if no_scale else torch.abs(scale[0]) * rotation[0]
        cam_T[-1, :3] = translation[0]

        return cam_T

    def compute_eye_rotation_matrix(self, eye):
        if self.axis_angle:
            rotation = pytorch3d.transforms.axis_angle_to_matrix(torch.cat([eye, torch.zeros_like(eye[..., :1])], dim=-1))
            return rotation.permute(0, 2, 1)
        else:
            # 0 left_eye + down - up
            # 1 left_eye + right - left
            # 2 right_eye + down - up
            # 3 right_eye + right - left
            sinx = torch.sin(eye[:, 0])
            siny = torch.sin(eye[:, 1])
            cosx = torch.cos(eye[:, 0])
            cosy = torch.cos(eye[:, 1])
            if self.batch_size != 1:
                rotXYZ = self.rotXYZ.repeat(1, self.batch_size, 1, 1).detach().clone()
            else:
                rotXYZ = self.rotXYZ.detach().clone()
            rotXYZ[0, :, 1, 1] = cosx
            rotXYZ[0, :, 1, 2] = -sinx
            rotXYZ[0, :, 2, 1] = sinx
            rotXYZ[0, :, 2, 2] = cosx
            rotXYZ[1, :, 0, 0] = cosy
            rotXYZ[1, :, 0, 2] = siny
            rotXYZ[1, :, 2, 0] = -siny
            rotXYZ[1, :, 2, 2] = cosy

            rotation = rotXYZ[1].bmm(rotXYZ[0])

            return rotation.permute(0, 2, 1)

    def compute_rotation_matrix(self, angles):
        if self.axis_angle:
            rotation = pytorch3d.transforms.axis_angle_to_matrix(angles)
            return rotation.permute(0, 2, 1)
        else:
            sinx = torch.sin(angles[:, 0])
            siny = torch.sin(angles[:, 1])
            sinz = torch.sin(angles[:, 2])
            cosx = torch.cos(angles[:, 0])
            cosy = torch.cos(angles[:, 1])
            cosz = torch.cos(angles[:, 2])

            if self.batch_size != 1:
                rotXYZ = self.rotXYZ.repeat(1, self.batch_size, 1, 1)
            else:
                rotXYZ = self.rotXYZ.detach().clone()

            rotXYZ[0, :, 1, 1] = cosx
            rotXYZ[0, :, 1, 2] = -sinx
            rotXYZ[0, :, 2, 1] = sinx
            rotXYZ[0, :, 2, 2] = cosx
            rotXYZ[1, :, 0, 0] = cosy
            rotXYZ[1, :, 0, 2] = siny
            rotXYZ[1, :, 2, 0] = -siny
            rotXYZ[1, :, 2, 2] = cosy
            rotXYZ[2, :, 0, 0] = cosz
            rotXYZ[2, :, 0, 1] = -sinz
            rotXYZ[2, :, 1, 0] = sinz
            rotXYZ[2, :, 1, 1] = cosz

            rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

            return rotation.permute(0, 2, 1)

    def add_illumination(self, face_texture, norm, gamma):
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8
        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(a0 * c0 * (nx * 0 + 1))
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(self.batch_size, face_texture.shape[1], 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color

    def rigid_transform(self, vs, rot, trans, scale):
        vs_r = torch.matmul(vs * scale, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)
        return vs_t

    def get_rot_tensor(self):
        return self.rot_tensor

    def get_trans_tensor(self):
        return self.trans_tensor

    def get_exp_tensor(self):
        return self.exp_tensor

    def get_tex_tensor(self):
        return self.tex_tensor

    def get_id_tensor(self):
        return self.id_tensor

    def get_gamma_tensor(self):
        return self.gamma_tensor

    def get_scale_tensor(self):
        return self.scale_tensor

    def get_eye_tensor(self):
        return self.eye_tensor

    def face_vertices(self, vert_attr, faces=None):
        """
        :param vertices: [batch size, number of vertices, C]
        :param faces: [batch size, number of faces, 3]
        :return: [batch size, number of faces, 3, 3]
        """
        assert (vert_attr.ndimension() == 3)
        if faces is not None:
            assert (faces.ndimension() == 3)
        else:
            faces = self.tri.repeat(vert_attr.shape[0], 1, 1)
        assert (vert_attr.shape[0] == faces.shape[0])
        assert (faces.shape[2] == 3)

        bs, nv = vert_attr.shape[:2]
        bs, nf = faces.shape[:2]
        faces = faces + (torch.arange(bs, dtype=torch.int32).to(vert_attr.device) * nv)[:, None, None]
        return vert_attr.reshape((bs * nv, -1))[faces.long()]

class ModelRenderer(nn.Module):
    def __init__(self, device='cuda:0', intr=None, img_size=256, cam_dist=10., render_depth=False, rasterize_blur_radius=0.):
        super(ModelRenderer, self).__init__()
        self.render_depth = render_depth
        self.img_size = img_size
        self.device = torch.device(device)
        self.cam_dist = cam_dist
        if intr is None:
            intr = np.eye(3, dtype=np.float32)
            intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2] = 1315, 1315, img_size // 2, img_size // 2
        self.fx, self.fy, self.cx, self.cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        self.renderer = self._get_renderer(self.device, cam_dist, torch.from_numpy(intr), render_depth=render_depth, rasterize_blur_radius=rasterize_blur_radius)
        self.p_mat = self._get_p_mat(device)
        self.reverse_xz = self._get_reverse_xz(device)
        self.camera_pos = self._get_camera_pose(device, cam_dist)

    def _get_renderer(self, device, cam_dist=10., K=None, render_depth=False, rasterize_blur_radius=0.):
        R, T = look_at_view_transform(cam_dist, 0, 0)  # camera's position
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        fx = -fx * 2.0 / (self.img_size - 1)
        fy = -fy * 2.0 / (self.img_size - 1)
        cx = - (cx - (self.img_size - 1) / 2.0) * 2.0 / (self.img_size - 1)
        cy = - (cy - (self.img_size - 1) / 2.0) * 2.0 / (self.img_size - 1)
        cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
                                     principal_point=((cx, cy),),
                                     in_ndc=True)

        lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]],
                             ambient_color=[[1, 1, 1]],
                             specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=rasterize_blur_radius if render_depth else 0.,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        ) if not render_depth else \
            MeshRendererWithDepth(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=device,
                    cameras=cameras,
                    lights=lights,
                    blend_params=blend_params
                )
            )
        return renderer

    def _get_camera_pose(self, device, cam_dist=10.):
        camera_pos = torch.tensor([0.0, 0.0, cam_dist], device=device).reshape(1, 1, 3)
        return camera_pos

    def _get_p_mat(self, device):
        # half_image_width = self.img_size // 2
        p_matrix = np.array([self.fx, 0.0, self.cx,
                             0.0, self.fy, self.cy,
                             0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 3, 3)
        return torch.tensor(p_matrix, device=device)

    def _get_reverse_xz(self, device):
        reverse_z = np.reshape(
            np.array([-1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=np.float32), [1, 3, 3])
        return torch.tensor(reverse_z, device=device)

    def project_vs(self, vs):
        batchsize = vs.shape[0]

        vs = torch.matmul(vs, self.reverse_xz.repeat((batchsize, 1, 1))) + self.camera_pos
        aug_projection = torch.matmul(
            vs, self.p_mat.repeat((batchsize, 1, 1)).permute((0, 2, 1)))

        face_projection = aug_projection[:, :, :2] / torch.reshape(aug_projection[:, :, 2], [batchsize, -1, 1])
        return face_projection


class MeshRendererWithDepth(MeshRenderer):
    def __init__(self, rasterizer, shader):
        super().__init__(rasterizer, shader)

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

# ['cheek_blow_left_MGB',
#  'cheek_blow_right_MGB',
#  'jaw_back_MGB',
#  'jaw_fwd_MGB',
#  'jaw_left_MGB',
#  'jaw_open_MGB',
#  'jaw_right_MGB',
#  'mouth_cornerPull_left_MGB',
#  'mouth_cornerPull_right_MGB',
#  'mouth_left_MGB',
#  'mouth_right_MGB',
#  'mouth_down_MGB',
#  'mouth_up_MGB'
#  'mouth_stretch_left_MGB',
#  'mouth_stretch_right_MGB',
#  'nose_wrinkle_left_MGB',
#  'nose_wrinkle_right_MGB',
#  'eye_faceScrunch_L_MGB',
#  'eye_blink_L_MGB',
#  'eye_blink_R_MGB',
#  'eye_cheekRaise_L_MGB',
#  'eye_cheekRaise_R_MGB',
#  'brow_raiseOuter_left_MGB',
#  'brow_raiseOuter_right_MGB',
#  'eye_faceScrunch_L_MGB']