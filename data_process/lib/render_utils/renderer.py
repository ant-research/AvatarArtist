# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn
import numpy as np

# from training_avatar_texture.volumetric_rendering.ray_marcher import MipRayMarcher2
# from training_avatar_texture.volumetric_rendering import math_utils

global Meshes, load_obj, rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes


def generate_planes(return_inv=True):   # 与project_onto_planes相对应
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    planes = torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)
    if return_inv:
        return torch.linalg.inv(planes)
    else:
        return planes


# from torch_utils import misc
# @misc.profiled_function
def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


# from torch_utils import persistence
# @persistence.persistent_class
class Pytorch3dRasterizer(nn.Module):
    ## TODO: add support for rendering non-squared images, since pytorc3d supports this now
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
            'cull_backfaces': True
        }
        # raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings['image_size']
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
            
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings['blur_radius'],
            faces_per_pixel=raster_settings['faces_per_pixel'],
            bin_size=0,#raster_settings['bin_size'],
            max_faces_per_bin=raster_settings['max_faces_per_bin'],
            perspective_correct=raster_settings['perspective_correct'],
            cull_backfaces=raster_settings['cull_backfaces']
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        # print(image_size)
        # import ipdb; ipdb.set_trace()
        return pixel_vals


def render_after_rasterize(attributes, pix_to_face, bary_coords):
    vismask = (pix_to_face > -1).float()
    D = attributes.shape[-1]
    attributes = attributes.clone()
    attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
    N, H, W, K, _ = bary_coords.shape
    mask = pix_to_face == -1
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
    pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0  # Replace masked values in output.
    pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
    pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
    return pixel_vals


# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


# ---------------------------- process/generate vertices, normals, faces
def generate_triangles(h, w, margin_x=2, margin_y=5, mask = None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    #.
    # w*h
    triangles = []
    for x in range(margin_x, w-1-margin_x):
        for y in range(margin_y, h-1-margin_y):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:,[0,2,1]]
    return triangles


def transform_points(points, tform, points_scale=None, out_scale=None):
    points_2d = points[:,:,:2]
        
    #'input points must use original range'
    if points_scale:
        assert points_scale[0]==points_scale[1]
        points_2d = (points_2d*0.5 + 0.5)*points_scale[0]
    # import ipdb; ipdb.set_trace()

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
                    torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), 
                    tform
                    ) 
    if out_scale: # h,w of output image size
        trans_points_2d[:,:,0] = trans_points_2d[:,:,0]/out_scale[1]*2 - 1
        trans_points_2d[:,:,1] = trans_points_2d[:,:,1]/out_scale[0]*2 - 1
    trans_points = torch.cat([trans_points_2d[:,:,:2], points[:,:,2:]], dim=-1)
    return trans_points


def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [batch_size, 3] tensor containing X, Y, and Z angles.
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [batch_size, 3, 3]. rotation matrices.
    '''
    angles = angles*(np.pi)/180.
    s = torch.sin(angles)
    c = torch.cos(angles)

    cx, cy, cz = (c[:, 0], c[:, 1], c[:, 2])
    sx, sy, sz = (s[:, 0], s[:, 1], s[:, 2])

    zeros = torch.zeros_like(s[:, 0]).to(angles.device)
    ones = torch.ones_like(s[:, 0]).to(angles.device)

    # Rz.dot(Ry.dot(Rx))
    R_flattened = torch.stack(
    [
      cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
      sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
          -sy,                cy * sx,                cy * cx,
    ],
    dim=0) #[batch_size, 9]
    R = torch.reshape(R_flattened, (-1, 3, 3)) #[batch_size, 3, 3]
    return R

import cv2
# end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpts(image, kpts, color = 'r', end_list=[19]):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1]==4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), radius)
        image = cv2.circle(image,(int(st[0]), int(st[1])), radius, c, radius*2)

    return image


import cv2
# def fill_mouth(images):
#     #Input: images: [batch, 1, h, w]
#     device = images.device
#     mouth_masks = []
#     for image in images:
#         image = image[0].cpu().numpy()
#         image = image * 255.
#         copyImg = image.copy()
#         h, w = image.shape[:2]
#         mask = np.zeros([h+2, w+2],np.uint8)
#         cv2.floodFill(copyImg, mask, (0, 0), (255, 255, 255), (0, 0, 0), (254, 254, 254), cv2.FLOODFILL_FIXED_RANGE)
#         # cv2.imwrite("debug.png", copyImg)
#         copyImg = torch.tensor(copyImg).to(device).to(torch.float32) / 127.5 - 1
#         mouth_masks.append(copyImg.unsqueeze(0))
#     mouth_masks = torch.stack(mouth_masks, 0)
#     mouth_masks = ((mouth_masks * 2. - 1.) * -1. + 1.) / 2.
#     # images = (images.bool() | mouth_masks.bool()).float()
#     res = (images + mouth_masks).clip(0, 1)
#
#     return res

def fill_mouth(images):
    #Input: images: [batch, 1, h, w]
    device = images.device
    mouth_masks = []
    out_mouth_masks = []
    for image in images:
        image = image[0].cpu().numpy()
        image = image * 255.
        copyImg = image.copy()
        h, w = image.shape[:2]
        mask = np.zeros([h+2, w+2], np.uint8)
        cv2.floodFill(copyImg, mask, (0, 0), (255, 255, 255), (0, 0, 0), (254, 254, 254), cv2.FLOODFILL_FIXED_RANGE)
        # cv2.imwrite("mouth_mask_ori.png", 255 - copyImg)
        mouth_mask = torch.tensor(255 - copyImg).to(device).to(torch.float32) / 255.
        mouth_masks.append(mouth_mask.unsqueeze(0))

        copyImg = cv2.erode(copyImg, np.ones((3, 3), np.uint8), iterations=3)
        copyImg = cv2.blur(copyImg, (5, 5))
        # cv2.imwrite("mouth_mask.png", mouth_mask)
        out_mouth_masks.append(torch.tensor(255 - copyImg).to(device).to(torch.float32).unsqueeze(0) / 255.)

    mouth_masks = torch.stack(mouth_masks, 0)
    res = (images + mouth_masks).clip(0, 1)

    return res, torch.stack(out_mouth_masks, dim=0)