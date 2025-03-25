import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import torchvision
from lib.render_utils.renderer import (
    batch_orth_proj, angle2matrix, face_vertices, render_after_rasterize
)
from lib.render_utils.ortho_renderer import get_renderer
from lib.FaceVerse.FaceVerseModel_v3 import ModelRenderer
import torchvision.utils as utils
from tqdm import tqdm
from lib.FaceVerse import get_recon_model
import time
from pytorch3d.structures import Meshes
import json
import multiprocessing
import shutil

count, total = multiprocessing.Value('i', 0), multiprocessing.Value('i', 0)


def load_obj_data(filename):
    """Load model data from .obj file."""
    v_list, vt_list, vc_list, vn_list = [], [], [], []
    f_list, fn_list, ft_list = [], [], []

    with open(filename, 'r') as fp:
        lines = fp.readlines()

    def seg_element_data(ele_str):
        """Parse face element data."""
        eles = ele_str.strip().split('/')
        fv, ft, fn = None, None, None
        if len(eles) == 1:
            fv = int(eles[0]) - 1
        elif len(eles) == 2:
            fv, ft = int(eles[0]) - 1, int(eles[1]) - 1
        elif len(eles) == 3:
            fv, fn = int(eles[0]) - 1, int(eles[2]) - 1
            ft = None if eles[1] == '' else int(eles[1]) - 1
        return fv, ft, fn

    for line in lines:
        if len(line) < 2:
            continue
        line_data = line.strip().split(' ')

        if line_data[0] == 'v':
            v_list.append(tuple(map(float, line_data[1:4])))
            vc_list.append(tuple(map(float, line_data[4:7])) if len(line_data) == 7 else (0.5, 0.5, 0.5))

        elif line_data[0] == 'vt':
            vt_list.append(tuple(map(float, line_data[1:3])))

        elif line_data[0] == 'vn':
            vn_list.append(tuple(map(float, line_data[1:4])))

        elif line_data[0] == 'f':
            fv0, ft0, fn0 = seg_element_data(line_data[1])
            fv1, ft1, fn1 = seg_element_data(line_data[2])
            fv2, ft2, fn2 = seg_element_data(line_data[3])
            f_list.append((fv0, fv1, fv2))
            if None not in (ft0, ft1, ft2):
                ft_list.append((ft0, ft1, ft2))
            if None not in (fn0, fn1, fn2):
                fn_list.append((fn0, fn1, fn2))

    return {
        'v': np.asarray(v_list), 'vt': np.asarray(vt_list), 'vc': np.asarray(vc_list),
        'vn': np.asarray(vn_list), 'f': np.asarray(f_list), 'ft': np.asarray(ft_list),
        'fn': np.asarray(fn_list)
    }


def save_obj_data(model, filename, log=True):
    """Save model data to .obj file."""
    assert 'v' in model and model['v'].size != 0

    with open(filename, 'w') as fp:
        if 'v' in model:
            for v, vc in zip(model['v'], model.get('vc', [])):
                fp.write(f"v {v[0]} {v[1]} {v[2]} {vc[2]} {vc[1]} {vc[0]}\n")
            for v in model['v']:
                fp.write(f"v {v[0]} {v[1]} {v[2]}\n")

        if 'vn' in model:
            for vn in model['vn']:
                fp.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")

        if 'vt' in model:
            for vt in model['vt']:
                fp.write(f"vt {vt[0]} {vt[1]}\n")

        if 'f' in model:
            for f_, ft_, fn_ in zip(model['f'], model.get('ft', []), model.get('fn', [])):
                f, ft, fn = np.array(f_) + 1, np.array(ft_) + 1, np.array(fn_) + 1
                fp.write(f"f {f[0]}/{ft[0]}/{fn[0]} {f[1]}/{ft[1]}/{fn[1]} {f[2]}/{ft[2]}/{fn[2]}\n")

    if log:
        print(f"Saved mesh as {filename}")


def gen_mouth_mask(lms_2d, new_crop=True):
    """Generate a mouth mask based on 2D landmarks."""
    lm = lms_2d[np.newaxis, ...]

    if new_crop:
        lm_mouth_outer = lm[:, [164, 18, 57, 287]]
        mouth_mask = np.concatenate([
            np.min(lm_mouth_outer[..., 1], axis=1, keepdims=True),
            np.max(lm_mouth_outer[..., 1], axis=1, keepdims=True),
            np.min(lm_mouth_outer[..., 0], axis=1, keepdims=True),
            np.max(lm_mouth_outer[..., 0], axis=1, keepdims=True)], axis=1
        )
    else:
        lm_mouth_outer = lm[:, [0, 17, 61, 291, 39, 269, 405, 181]]
        mouth_avg = np.mean(lm_mouth_outer, axis=1, keepdims=False)
        ups, bottoms = np.max(lm_mouth_outer[..., 0], axis=1, keepdims=True), np.min(lm_mouth_outer[..., 0], axis=1,
                                                                                     keepdims=True)
        lefts, rights = np.min(lm_mouth_outer[..., 1], axis=1, keepdims=True), np.max(lm_mouth_outer[..., 1], axis=1,
                                                                                      keepdims=True)
        mask_res = np.max(np.concatenate((ups - bottoms, rights - lefts), axis=1), axis=1, keepdims=True) * 1.2
        mask_res = mask_res.astype(int)
        mouth_mask = np.concatenate([
            (mouth_avg[:, 1:] - mask_res // 2).astype(int),
            (mouth_avg[:, 1:] + mask_res // 2).astype(int),
            (mouth_avg[:, :1] - mask_res // 2).astype(int),
            (mouth_avg[:, :1] + mask_res // 2).astype(int)], axis=1
        )

    return mouth_mask[0]
def render_orth(tracking_dir, save_dir, face_model_dir, fv2fl_T, orth_transforms, render_vis=True, save_mesh_dir=None):
    """
    Perform orthographic rendering of face models.

    Args:
        tracking_dir (str): Directory containing tracking data.
        save_dir (str): Directory to save rendered results.
        face_model_dir (str): Directory containing face model files.
        fv2fl_T (np.ndarray): Transformation matrix.
        orth_transforms (dict): Orthographic transformation parameters.
        render_vis (bool): Whether to save visualization images.
        save_mesh_dir (str, optional): Directory to save mesh files.

    Returns:
        None
    """
    debug = False
    save_mesh_flag = save_mesh_dir is not None
    res = 256

    # Initialize orthographic renderer
    ortho_renderer = get_renderer(
        img_size=res,
        device='cuda:0',
        T=torch.tensor([[0, 0, 10.]], dtype=torch.float32, device='cuda:0'),
        K=[-1.0, -1.0, 0., 0.],
        orthoCam=True,
        rasterize_blur_radius=1e-6
    )

    orth_scale = orth_transforms['scale']
    orth_shift = torch.from_numpy(orth_transforms['shift']).cuda().unsqueeze(0)

    # Load face model
    face_model_path = os.path.join(face_model_dir, 'faceverse_v3_1.npy')
    recon_model, model_dict = get_recon_model(model_path=face_model_path, return_dict=True, device='cuda:0')

    vert_uvcoords = model_dict['uv_per_ver']

    # Expand the UV area for better face fitting
    vert_idx = (vert_uvcoords[:, 1] > 0.273) & (vert_uvcoords[:, 1] < 0.727) & \
               (vert_uvcoords[:, 0] > 0.195) & (vert_uvcoords[:, 0] < 0.805)
    vert_uvcoords[vert_idx] = (vert_uvcoords[vert_idx] - 0.5) * 1.4 + 0.5

    vert_uvcoords = torch.from_numpy(vert_uvcoords).unsqueeze(0).cuda()
    faces = uvfaces = torch.from_numpy(model_dict['tri']).unsqueeze(0).cuda()

    # Load face mask
    vert_mask = np.load(os.path.join(face_model_dir, 'v31_face_mask_new.npy'))
    vert_mask[model_dict['ver_inds'][0]:model_dict['ver_inds'][2]] = 1
    vert_mask = torch.from_numpy(vert_mask).view(1, -1, 1).cuda()

    vert_uvcoords = vert_uvcoords * 2 - 1
    vert_uvcoords = torch.cat([vert_uvcoords, vert_mask], dim=-1)  # [bz, ntv, 3]
    face_uvcoords = face_vertices(vert_uvcoords, uvfaces).cuda()

    # Prepare to save mesh if required
    if save_mesh_flag:
        tri = recon_model.tri.cpu().numpy().squeeze()
        uv = recon_model.uv.cpu().numpy().squeeze()
        tri_uv = recon_model.tri_uv.cpu().numpy().squeeze()

    # Transformation matrix
    trans_init = torch.from_numpy(fv2fl_T).cuda()
    R_ = trans_init[:3, :3]
    t_ = trans_init[:3, 3:]

    tform = angle2matrix(torch.tensor([0, 0, 0]).reshape(1, -1)).cuda()
    cam = torch.tensor([1., 0, 0]).cuda()

    mouth_masks = []
    total_num = len(os.listdir(tracking_dir))
    progress_bar = tqdm(os.listdir(tracking_dir))

    t0 = time.time()
    count = 0

    for name in progress_bar:
        prefix = '0'
        dst_sub_dir = os.path.join(save_dir, prefix)
        os.makedirs(dst_sub_dir, exist_ok=True)

        coeff = torch.from_numpy(np.load(os.path.join(tracking_dir, name, 'coeffs.npy'))).unsqueeze(0).cuda()
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff, scale = recon_model.split_coeffs(coeff)

        # Compute vertices
        vs = recon_model.get_vs(id_coeff, exp_coeff)
        vert = torch.matmul(vs[0], R_.T) + t_.T

        v = vert.unsqueeze(0)
        transformed_vertices = (torch.bmm(v, tform) + orth_shift) * orth_scale
        transformed_vertices = batch_orth_proj(transformed_vertices, cam)
        transformed_vertices = torch.bmm(transformed_vertices,
                                         angle2matrix(torch.tensor([0, 180, 0]).reshape(1, -1)).cuda())

        # Save mesh if required
        if save_mesh_flag:
            mesh = {'v': transformed_vertices.squeeze().cpu().numpy(), 'vt': uv, 'f': tri, 'ft': tri_uv}
            os.makedirs(os.path.join(save_mesh_dir, prefix), exist_ok=True)
            save_obj_data(mesh, os.path.join(save_mesh_dir, prefix, name.split('.')[0] + '.obj'), log=False)

        # Rasterization and rendering
        mesh = Meshes(transformed_vertices, faces.long())
        fragment = ortho_renderer.rasterizer(mesh)

        rendering = render_after_rasterize(
            attributes=face_uvcoords,
            pix_to_face=fragment.pix_to_face,
            bary_coords=fragment.bary_coords
        )

        uvcoords_images, render_mask = rendering[:, :-1, :, :], rendering[:, -1:, :, :]
        render_mask *= uvcoords_images[:, -1:]
        uvcoords_images *= render_mask

        np.save(os.path.join(dst_sub_dir, name.split('.')[0] + '.npy'), rendering[0].permute(1, 2, 0).cpu().numpy())

        if render_vis:
            utils.save_image(uvcoords_images, os.path.join(dst_sub_dir, name.split('.')[0] + '.png'), normalize=True,
                             range=(-1, 1))

        # Compute 2D landmarks
        lms_3d = recon_model.get_lms(transformed_vertices).cpu().squeeze().numpy()
        lms_2d = np.round((lms_3d[:, :2] + 1) * 0.5 * res).astype(np.uint8)
        mouth_mask = gen_mouth_mask(lms_2d)
        mouth_masks.append([f'{prefix}/{name.split(".")[0]}.png', mouth_mask.tolist()])

        count += 1
        progress_bar.set_description(f'{name.split(".")[0]} {int(1000 * (time.time() - t0) / count):03d}')

    # Save mouth masks
    with open(os.path.join(save_dir, 'mouth_masks.json'), "w") as f:
        json.dump(mouth_masks, f, indent=4)

def render_orth_mp(
    tracking_dir, save_dir, face_model_dir, fv2fl_T, orth_transforms, focal_ratio,
    render_vis=False, save_mesh_dir=None, save_uv_dir=None, num_thread=1,
    render_normal_uv=False, prefix_ls=None, crop_param=None, use_smooth=False,
    save_coeff=False, skip=False
):
    """
    Perform multi-threaded orthographic rendering of face models.

    Args:
        tracking_dir (str): Directory containing tracking data.
        save_dir (str): Directory to save rendered results.
        face_model_dir (str): Directory containing face model files.
        fv2fl_T (np.ndarray): Transformation matrix.
        orth_transforms (dict): Orthographic transformation parameters.
        focal_ratio (float): Camera focal length ratio.
        render_vis (bool): Whether to save visualization images.
        save_mesh_dir (str, optional): Directory to save mesh files.
        save_uv_dir (str, optional): Directory to save UV maps.
        num_thread (int): Number of threads for parallel processing.
        render_normal_uv (bool): Whether to render normal UV maps.
        prefix_ls (list, optional): List of prefixes to process.
        crop_param (dict, optional): Cropping parameters.
        use_smooth (bool): Whether to use smoothed coefficients.
        save_coeff (bool): Whether to save coefficients.
        skip (bool): Whether to skip already processed directories.

    Returns:
        None
    """
    print(f'Num Threads: {num_thread}')

    if num_thread > 1:
        # Prepare data for multiprocessing
        data_ls = [
            {
                'tracking_dir': os.path.join(tracking_dir, prefix),
                'save_dir': save_dir,
                'face_model_dir': face_model_dir,
                'fv2fl_T': fv2fl_T,
                'orth_transforms': orth_transforms,
                'render_vis': render_vis,
                'save_mesh_dir': save_mesh_dir,
                'save_uv_dir': save_uv_dir,
                'prefix': prefix,
                'render_normal_uv': render_normal_uv,
                'crop_param': crop_param,
                'use_smooth': use_smooth,
                'focal_ratio': focal_ratio,
                'save_coeff': save_coeff
            }
            for prefix in os.listdir(tracking_dir)
            if os.path.isdir(os.path.join(tracking_dir, prefix)) and
               (not os.path.exists(os.path.join(save_dir, prefix)) if skip else True)
        ]

        num_thread = min(num_thread, len(data_ls))
        with multiprocessing.Pool(num_thread) as pool:
            pool.map(perform_render, data_ls)
    else:
        # Single-threaded execution
        if prefix_ls is None:
            for prefix in os.listdir(tracking_dir):
                if os.path.isdir(os.path.join(tracking_dir, prefix)):
                    perform_render({
                        'tracking_dir': os.path.join(tracking_dir, prefix),
                        'save_dir': save_dir,
                        'face_model_dir': face_model_dir,
                        'fv2fl_T': fv2fl_T,
                        'orth_transforms': orth_transforms,
                        'render_vis': render_vis,
                        'save_mesh_dir': save_mesh_dir,
                        'save_uv_dir': save_uv_dir,
                        'prefix': prefix,
                        'render_normal_uv': render_normal_uv,
                        'crop_param': crop_param,
                        'use_smooth': use_smooth,
                        'focal_ratio': focal_ratio,
                        'save_coeff': save_coeff
                    })
        else:
            for prefix in prefix_ls:
                prefix = prefix if prefix else '0'
                perform_render({
                    'tracking_dir': tracking_dir,
                    'save_dir': save_dir,
                    'face_model_dir': face_model_dir,
                    'fv2fl_T': fv2fl_T,
                    'focal_ratio': focal_ratio,
                    'orth_transforms': orth_transforms,
                    'render_vis': render_vis,
                    'save_mesh_dir': save_mesh_dir,
                    'save_uv_dir': save_uv_dir,
                    'prefix': prefix,
                    'render_normal_uv': render_normal_uv,
                    'crop_param': crop_param,
                    'use_smooth': use_smooth,
                    'save_coeff': save_coeff
                })

def perform_render(data):
    """
    Perform rendering and optionally save UV maps.

    Args:
        data (dict): Dictionary containing rendering parameters.

    Returns:
        None
    """
    render_orth_(data)

    if data.get('save_uv_dir') is not None:
        save_uv_(data)

def save_uv_(data):
    """
    Save UV maps, including normal maps and projected position maps.

    Args:
        data (dict): Dictionary containing rendering parameters.

    Returns:
        None
    """
    # Extract parameters from data dictionary
    tracking_dir = data['tracking_dir']
    save_uv_dir = data['save_uv_dir']
    face_model_dir = data['face_model_dir']
    prefix = data['prefix']
    focal_ratio = data['focal_ratio']
    render_normal_uv = data['render_normal_uv']

    img_res, render_res = 512, 256  # Default image resolution is 512

    # Initialize UV renderer
    uv_renderer = get_renderer(
        img_size=render_res,
        device='cuda:0',
        T=torch.tensor([[0, 0, 10.]], dtype=torch.float32, device='cuda:0'),
        K=[-1.0, -1.0, 0., 0.],
        orthoCam=True,
        rasterize_blur_radius=1e-6
    )

    # Camera intrinsic matrix
    cam_K = np.eye(3, dtype=np.float32)
    cam_K[0, 0] = cam_K[1, 1] = focal_ratio * img_res
    cam_K[0, 2] = cam_K[1, 2] = img_res // 2

    # Initialize model renderer
    renderer = ModelRenderer(img_size=img_res, device='cuda:0', intr=cam_K, cam_dist=5.0)

    # Load face model
    face_model_path = os.path.join(face_model_dir, 'faceverse_v3_1.npy')
    recon_model, model_dict = get_recon_model(model_path=face_model_path, return_dict=True, device='cuda:0', img_size=img_res, intr=cam_K, cam_dist=5)

    vert_uvcoords = model_dict['uv_per_ver']

    # Expand the UV area for better face fitting
    vert_idx = (vert_uvcoords[:, 1] > 0.273) & (vert_uvcoords[:, 1] < 0.727) & \
               (vert_uvcoords[:, 0] > 0.195) & (vert_uvcoords[:, 0] < 0.805)
    vert_uvcoords[vert_idx] = (vert_uvcoords[vert_idx] - 0.5) * 1.4 + 0.5

    vert_uvcoords = torch.from_numpy(vert_uvcoords).unsqueeze(0).cuda()
    faces = torch.from_numpy(model_dict['tri']).unsqueeze(0).cuda()

    # Load face mask
    vert_mask = np.load(os.path.join(face_model_dir, 'v31_face_mask_new.npy'))
    vert_mask[model_dict['ver_inds'][0]:model_dict['ver_inds'][2]] = 1
    vert_mask = torch.from_numpy(vert_mask).view(1, -1, 1).cuda()

    vert_uvcoords = vert_uvcoords * 2 - 1
    vert_mask[0, ~vert_idx] *= 0  # For UV rendering
    vert_uvcoords = torch.cat([vert_uvcoords, (1 - vert_mask)], dim=-1)

    # UV rasterization
    uv_fragment = uv_renderer.rasterizer(Meshes(vert_uvcoords, faces.long()))

    # Load UV face mask
    uv_face_eye_mask = cv2.imread(os.path.join(face_model_dir, 'dense_uv_expanded_mask_onlyFace.png'))[..., 0]
    uv_face_eye_mask = torch.from_numpy(uv_face_eye_mask.astype(np.float32) / 255).view(1, 256, 256, 1).permute(0, 3, 1, 2)

    os.makedirs(os.path.join(save_uv_dir, prefix), exist_ok=True)

    print(f'Rendering: {tracking_dir}')
    for name in os.listdir(tracking_dir):
        if not os.path.exists(os.path.join(tracking_dir, name, 'finish')):
            print(f'Missing: {os.path.join(tracking_dir, name, "finish")}')
            continue

        coeff = torch.from_numpy(np.load(os.path.join(tracking_dir, name, 'coeffs.npy'))).unsqueeze(0).cuda()
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff, scale = recon_model.split_coeffs(coeff)

        # Compute eye transformations
        l_eye_mat = recon_model.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_mat = recon_model.compute_eye_rotation_matrix(eye_coeff[:, 2:])
        l_eye_mean = recon_model.get_l_eye_center(id_coeff)
        r_eye_mean = recon_model.get_r_eye_center(id_coeff)

        # Compute vertex positions
        vs = recon_model.get_vs(id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean)

        # Save canonical vertex normal map in UV
        if render_normal_uv:
            vert_norm = recon_model.compute_norm(vs, recon_model.tri, recon_model.point_buf)
            vert_norm = torch.clip((vert_norm + 1) * 127.5, 0, 255)
            vert_norm = torch.cat([vert_norm, vert_mask], dim=-1)

            rendered_normal = render_after_rasterize(
                attributes=face_vertices(vert_norm, faces),
                pix_to_face=uv_fragment.pix_to_face,
                bary_coords=uv_fragment.bary_coords
            ).cpu()

            rendered_normal = rendered_normal[:, :3] * (rendered_normal[:, -1:].clone() * rendered_normal[:, -2:-1]) * uv_face_eye_mask
            normal_img = torch.clamp(rendered_normal[0, :3, :, :], 0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            cv2.imwrite(os.path.join(save_uv_dir, prefix, f'{name}_uvnormal.png'), normal_img[:, :, ::-1])

        # Save projected position map in UV
        rotation = recon_model.compute_rotation_matrix(angles)
        vs_t = recon_model.rigid_transform(vs, rotation, translation, torch.abs(scale))
        vs_norm = recon_model.compute_norm(vs_t, recon_model.tri, recon_model.point_buf)
        vs_proj = renderer.project_vs(vs_t) / img_res * 2 - 1  # Normalize to [-1, 1]

        vert_attr = torch.cat([vs_proj, vert_mask * (vs_norm[..., 2:] > 0.1).float()], dim=-1)

        uv_pverts = render_after_rasterize(
            attributes=face_vertices(vert_attr, faces),
            pix_to_face=uv_fragment.pix_to_face,
            bary_coords=uv_fragment.bary_coords
        ).cpu()

        uv_pverts = (uv_pverts[:, :-1] * uv_pverts[:, -1:])  # Projected position map in UV
        uv_pverts[:, -1:] *= uv_face_eye_mask

        np.save(os.path.join(save_uv_dir, prefix, f'{name}.npy'), uv_pverts[0].permute(1, 2, 0).numpy().astype(np.float16))

        # Load original image
        image_path = os.path.join(os.path.dirname(save_uv_dir), 'images512x512', prefix, f'{name}.png')
        images = cv2.imread(image_path)
        images = torch.from_numpy(images.astype(np.float32) / 255).view(1, 512, 512, 3).permute(0, 3, 1, 2)

        uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[..., :2], mode='bilinear', align_corners=False)
        uv_texture_gt = uv_gt * uv_pverts[:, -1:] + torch.ones_like(uv_gt) * (1 - uv_pverts[:, -1:])

        cv2.imwrite(os.path.join(save_uv_dir, prefix, f'{name}_uvgttex.png'), (uv_texture_gt[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))

def render_orth_(data):
    """
    Perform orthographic rendering of face models.

    Args:
        data (dict): Dictionary containing rendering parameters.

    Returns:
        None
    """
    # Extract parameters from the dictionary
    tracking_dir = data['tracking_dir']
    save_dir = data['save_dir']
    face_model_dir = data['face_model_dir']
    fv2fl_T = data['fv2fl_T']
    orth_transforms = data['orth_transforms']
    prefix = data['prefix']
    render_vis = data['render_vis']
    save_mesh_dir = data['save_mesh_dir']
    crop_param = data['crop_param']
    use_smooth = data['use_smooth']
    save_coeff = data['save_coeff']

    save_mesh_flag = save_mesh_dir is not None
    res, render_res = 256, 512  # Final crop ensures 256x256 output

    # Initialize orthographic renderer
    ortho_renderer = get_renderer(
        img_size=render_res,
        device='cuda:0',
        T=torch.tensor([[0, 0, 10.]], dtype=torch.float32, device='cuda:0'),
        K=[-1.0, -1.0, 0., 0.],
        orthoCam=True,
        rasterize_blur_radius=1e-6
    )

    orth_scale = orth_transforms['scale']
    orth_shift = torch.from_numpy(orth_transforms['shift']).cuda().unsqueeze(0)

    # Load face model
    face_model_path = os.path.join(face_model_dir, 'faceverse_v3_1.npy')
    recon_model, model_dict = get_recon_model(model_path=face_model_path, return_dict=True, device='cuda:0')

    vert_uvcoords = model_dict['uv_per_ver']

    # Expand the UV area for better face fitting
    vert_idx = (vert_uvcoords[:, 1] > 0.273) & (vert_uvcoords[:, 1] < 0.727) & \
               (vert_uvcoords[:, 0] > 0.195) & (vert_uvcoords[:, 0] < 0.805)
    vert_uvcoords[vert_idx] = (vert_uvcoords[vert_idx] - 0.5) * 1.4 + 0.5

    vert_uvcoords = torch.from_numpy(vert_uvcoords).unsqueeze(0).cuda()
    faces = uvfaces = torch.from_numpy(model_dict['tri']).unsqueeze(0).cuda()

    # Load face mask
    vert_mask = np.load(os.path.join(face_model_dir, 'v31_face_mask_new.npy'))
    vert_mask[model_dict['ver_inds'][0]:model_dict['ver_inds'][2]] = 1
    vert_mask = torch.from_numpy(vert_mask).view(1, -1, 1).cuda()

    vert_uvcoords = vert_uvcoords * 2 - 1
    vert_uvcoords = torch.cat([vert_uvcoords, vert_mask.clone()], dim=-1)
    face_uvcoords = face_vertices(vert_uvcoords, uvfaces)

    vert_mask[0, ~vert_idx] *= 0  # For UV rendering

    # Prepare to save mesh if required
    if save_mesh_flag:
        tri = recon_model.tri.cpu().numpy().squeeze()
        uv = recon_model.uv.cpu().numpy().squeeze()
        tri_uv = recon_model.tri_uv.cpu().numpy().squeeze()
        os.makedirs(os.path.join(save_mesh_dir, prefix), exist_ok=True)

    # Transformation matrix
    trans_init = torch.from_numpy(fv2fl_T).cuda()
    R_ = trans_init[:3, :3]
    t_ = trans_init[:3, 3:]

    tform = angle2matrix(torch.tensor([0, 0, 0]).reshape(1, -1)).cuda()
    cam = torch.tensor([1., 0, 0]).cuda()

    mouth_masks = []

    print(f'Rendering: {tracking_dir}')
    for name in os.listdir(tracking_dir):
        if not os.path.exists(os.path.join(tracking_dir, name, 'finish')):
            print(f'Missing: {os.path.join(tracking_dir, name, "finish")}')
            continue

        dst_sub_dir = os.path.join(save_dir, prefix)
        os.makedirs(dst_sub_dir, exist_ok=True)

        # Load coefficients
        coeff_path = os.path.join(tracking_dir, name, 'smooth_coeffs.npy' if use_smooth else 'coeffs.npy')
        if save_coeff:
            shutil.copy(coeff_path, os.path.join(dst_sub_dir, f'{name}_coeff.npy'))

        coeff = torch.from_numpy(np.load(coeff_path)).unsqueeze(0).cuda()
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff, scale = recon_model.split_coeffs(coeff)

        # Compute eye transformations
        l_eye_mat = recon_model.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_mat = recon_model.compute_eye_rotation_matrix(eye_coeff[:, 2:])
        l_eye_mean = recon_model.get_l_eye_center(id_coeff)
        r_eye_mean = recon_model.get_r_eye_center(id_coeff)

        # Compute vertex positions
        vs = recon_model.get_vs(id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean)
        vert = torch.matmul(vs[0], R_.T) + t_.T

        v = vert.unsqueeze(0)
        transformed_vertices = (torch.bmm(v, tform) + orth_shift) * orth_scale
        transformed_vertices = batch_orth_proj(transformed_vertices, cam)

        # Reverse Z-axis for proper rendering
        transformed_vertices[..., -1] *= -1

        # Save mesh if required
        if save_mesh_flag:
            mesh = {'v': transformed_vertices.squeeze().cpu().numpy(), 'vt': uv, 'f': tri, 'ft': tri_uv}
            save_obj_data(mesh, os.path.join(save_mesh_dir, prefix, f'{name}.obj'), log=False)

        # Rasterization and rendering
        mesh = Meshes(transformed_vertices, faces.long())
        fragment = ortho_renderer.rasterizer(mesh)

        rendering = render_after_rasterize(
            attributes=face_uvcoords,
            pix_to_face=fragment.pix_to_face,
            bary_coords=fragment.bary_coords
        )

        render_mask = rendering[:, -1:, :, :].clone()
        render_mask *= rendering[:, -2:-1]
        rendering *= render_mask

        # Apply cropping if needed
        if crop_param is not None:
            rendering = rendering[:, :, crop_param[1]:crop_param[1] + crop_param[3], crop_param[0]:crop_param[0] + crop_param[2]]

        if res != rendering.shape[2]:
            rendering = F.interpolate(rendering, size=(res, res), mode='bilinear', align_corners=False)

        np.save(os.path.join(dst_sub_dir, f'{name}.npy'), rendering[0].permute(1, 2, 0).cpu().numpy().astype(np.float16))

        # Compute mouth mask
        lms_3d = recon_model.get_lms(transformed_vertices).cpu().squeeze().numpy()
        lms_2d = np.round((lms_3d[:, :2] + 1) * 0.5 * res).astype(np.uint8)
        mouth_mask = gen_mouth_mask(lms_2d, new_crop=False)
        mouth_masks.append([f'{prefix}/{name}.png', mouth_mask.tolist()])

        # Visualization
        if render_vis:
            boxes = torch.tensor([[mouth_mask[2], mouth_mask[0], mouth_mask[3], mouth_mask[1]]])
            vis_uvcoords = utils.draw_bounding_boxes(((rendering[0, :-1, :, :] + 1) * 127.5).to(dtype=torch.uint8).cpu(), boxes, colors=(0, 255, 0), width=1)
            vis_image = torchvision.transforms.ToPILImage()(vis_uvcoords)
            vis_image.save(os.path.join(dst_sub_dir, f'{name}.png'))
def fill_mouth(images):
    """
    Fill the mouth area in images.

    Args:
        images: Input images, shape [batch, 1, H, W].

    Returns:
        Images with filled mouth regions.
    """
    device = images.device
    mouth_masks = []

    for image in images:
        img = (image[0].cpu().numpy() * 255.).astype(np.uint8)
        copy_img = img.copy()
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
        cv2.floodFill(copy_img, mask, (0, 0), 255, loDiff=0, upDiff=254, flags=cv2.FLOODFILL_FIXED_RANGE)
        copy_img = (torch.tensor(copy_img, device=device).float() / 127.5) - 1
        mouth_masks.append(copy_img.unsqueeze(0))

    mouth_masks = torch.stack(mouth_masks, dim=0)
    mouth_masks = ((mouth_masks * 2 - 1) * -1 + 1) / 2
    return torch.clamp(images + mouth_masks, 0, 1)


def rasterize(verts, faces, face_attr, rasterizer, cam_dist=10):
    """Perform rasterization of vertices and faces."""
    verts[:, :, 2] += cam_dist
    return rasterizer(verts, faces, face_attr, 256, 256)


def ortho_render(verts, faces, face_attr, renderer):
    """Perform orthographic rendering."""
    mesh = Meshes(verts, faces.long())
    return renderer(mesh, face_attr, need_rgb=False)[-1]


def calculate_new_intrinsic(intr, mode, param):
    """
    Calculate new intrinsic matrix based on transformation mode.

    Args:
        intr: Original intrinsic matrix.
        mode: Transformation mode ('resize', 'crop', 'padding').
        param: Transformation parameters.

    Returns:
        Modified intrinsic matrix.
    """
    cam_K = intr.copy()

    if mode == 'resize':
        cam_K[0] *= param[0]
        cam_K[1] *= param[1]
    elif mode == 'crop':
        cam_K[0, 2] -= param[0]  # -left
        cam_K[1, 2] -= param[1]  # -top
    elif mode == 'padding':
        cam_K[0, 2] += param[2]  # + padding left
        cam_K[1, 2] += param[0]  # + padding top
    else:
        raise ValueError("Invalid transformation mode")

    return cam_K


def make_cam_dataset_FFHQ(tracking_dir, fv2fl_T, focal_ratio=2.568, use_smooth=False, test_data=False):
    """
    Create camera dataset for FFHQ.

    Args:
        tracking_dir: Directory containing tracking data.
        fv2fl_T: Transformation matrix from faceverse to face landmarks.
        focal_ratio: Camera focal length ratio.
        use_smooth: Whether to use smoothed coefficients.
        test_data: Whether to create a test dataset.

    Returns:
        Camera parameters, condition parameters, expression and eye movement parameters.
    """
    cam_K = np.eye(3, dtype=np.float32)
    cam_K[0, 0] = cam_K[1, 1] = focal_ratio
    cam_K[0, 2] = cam_K[1, 2] = 0.5

    cam_params, cond_cam_params, fv_exp_eye_params = ({}, {}, {}) if test_data else ([], [], [])

    for prefix in tqdm(os.listdir(tracking_dir)):
        if not os.path.isdir(os.path.join(tracking_dir, prefix)):
            continue

        if test_data:
            cam_params[prefix], cond_cam_params[prefix], fv_exp_eye_params[prefix] = [], [], []

        for name in os.listdir(os.path.join(tracking_dir, prefix)):
            if not os.path.exists(os.path.join(tracking_dir, prefix, name, 'finish')):
                continue

            metaFace_extr = np.load(
                os.path.join(tracking_dir, prefix, name,
                             'metaFace_extr_smooth.npz' if use_smooth else 'metaFace_extr.npz')
            )

            camT_mesh2cam = metaFace_extr['transformation']
            camT_cam2mesh = np.linalg.inv(camT_mesh2cam)
            camT_cam2mesh = np.dot(fv2fl_T, camT_cam2mesh)

            angle = metaFace_extr['self_angle']
            trans = metaFace_extr['self_translation']

            coeff = np.load(os.path.join(tracking_dir, prefix, name, 'coeffs.npy'))
            exp_coeff = coeff[150:150 + 171]  # Expression coefficients
            eye_coeff = coeff[572 + 33:572 + 37]  # Eye movement coefficients

            img_path = f"{prefix}/{name}.png"
            cam_data = np.concatenate([camT_cam2mesh.reshape(-1), cam_K.reshape(-1)]).tolist()
            cond_data = np.concatenate([angle, trans]).tolist()
            expr_eye_data = np.concatenate([exp_coeff, eye_coeff]).tolist()

            if test_data:
                cam_params[prefix].append([img_path, cam_data])
                cond_cam_params[prefix].append([img_path, cond_data])
                fv_exp_eye_params[prefix].append([img_path, expr_eye_data])
            else:
                cam_params.append([img_path, cam_data])
                cond_cam_params.append([img_path, cond_data])
                fv_exp_eye_params.append([img_path, expr_eye_data])

    return cam_params, cond_cam_params, fv_exp_eye_params






