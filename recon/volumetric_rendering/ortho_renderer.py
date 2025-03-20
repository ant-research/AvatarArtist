import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)


class MeshRendererWithDepth(MeshRenderer):
    def __init__(self, rasterizer, shader):
        super().__init__(rasterizer, shader)

    def forward(self, meshes_world, attributes=None, need_rgb=True, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = pixel_vals = None
        if attributes is not None:
            bary_coords, pix_to_face = fragments.bary_coords, fragments.pix_to_face.clone()

            vismask = (pix_to_face > -1).float()
            D = attributes.shape[-1]
            attributes = attributes.clone();
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

        if need_rgb:
            images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments.zbuf, pixel_vals


def get_renderer(img_size, device, R=None, T=None, K=None, orthoCam=False, rasterize_blur_radius=0.):
    if R is None:
        R = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)

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
