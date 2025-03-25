import pickle
import numpy as np
import os
import torch
import cv2

def get_box_warp_param(X_bounding, Y_bounding, Z_bounding):
    fx = 2 / (X_bounding[1] - X_bounding[0])
    cx = fx * (X_bounding[0] + X_bounding[1]) * 0.5
    fy = 2 / (Y_bounding[1] - Y_bounding[0])
    cy = fy * (Y_bounding[0] + Y_bounding[1]) * 0.5
    fz = 2 / (Z_bounding[1] - Z_bounding[0])
    cz = fz * (Z_bounding[0] + Z_bounding[1]) * 0.5
    return (fx, fy, fz), (-cx, -cy, -cz)

class UniformBoxWarp_new(torch.nn.Module):
    def __init__(self, scales, trans):
        super().__init__()
        self.register_buffer('scale_factor', torch.tensor(scales, dtype=torch.float32).reshape(1, 3))
        self.register_buffer('trans_factor', torch.tensor(trans, dtype=torch.float32).reshape(1, 3))
        # self.scale_factor = torch.tensor(scales, dtype=torch.float32).reshape(1, 3)
        # self.trans_factor = torch.tensor(trans, dtype=torch.float32).reshape(1, 3)

    def forward(self, coordinates):
        # [N, 3] OR [B, N, 3]
        scale_factor = self.scale_factor.unsqueeze(0) if coordinates.ndim==3 else self.scale_factor
        trans_factor = self.trans_factor.unsqueeze(0) if coordinates.ndim==3 else self.trans_factor
        return (coordinates * scale_factor) + trans_factor

def pad_bbox(bbox, img_wh, padding_ratio=0.2):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    size_bb = int(max(width, height) * (1 + padding_ratio))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(img_wh[0] - x1, size_bb)
    size_bb = min(img_wh[1] - y1, size_bb)

    return [x1, y1, x1 + size_bb, y1 + size_bb]


def mymkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lm_weights(device, use_mediapipe=False):
    if use_mediapipe:
        w = torch.ones(478).to(device)
        lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
        l_eye = [263, 249, 390, 373, 374, 380, 381, 382, 263, 466, 388, 387, 386, 385, 384, 398]
        l_brow = [276, 283, 282, 295, 300, 293, 334, 296]
        r_eye = [33, 7, 163, 144, 145, 153, 154, 155, 33, 246, 161, 160, 159, 158, 157, 173]
        r_brow = [46, 53, 52, 65, 70, 63, 105, 66]
        oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132,
                93, 234, 127, 162, 21, 54, 103, 67, 109]
        w[lips] = 5
        w[l_eye] = 50   # 5 zxc
        w[r_eye] = 50   # 5 zxc
        w[l_brow] = 5
        w[r_brow] = 5
        w[468:] = 5
    else:
        w = torch.ones(66).to(device)
        w[28:31] = 5
        w[36:48] = 5
        w[48:66] = 5
    norm_w = w / w.sum()
    return norm_w

def load_obj_data(filename):
    """load model data from .obj file"""
    v_list = []  # vertex coordinate
    vt_list = []  # vertex texture coordinate
    vc_list = []  # vertex color
    vn_list = []  # vertex normal
    f_list = []  # face vertex indices
    fn_list = []  # face normal indices
    ft_list = []  # face texture indices

    # read data
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        if len(line) < 2:
            continue
        line_data = line.strip().split(' ')
        # parse vertex cocordinate
        if line_data[0] == 'v':
            v_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))
            if len(line_data) == 7:
                vc_list.append((float(line_data[4]), float(line_data[5]), float(line_data[6])))
            else:
                vc_list.append((0.5, 0.5, 0.5))

        # parse vertex texture coordinate
        if line_data[0] == 'vt':
            vt_list.append((float(line_data[1]), float(line_data[2])))

        # parse vertex normal
        if line_data[0] == 'vn':
            vn_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))

        # parse face
        if line_data[0] == 'f':
            # used for parsing face element data
            def segElementData(ele_str):
                fv = None
                ft = None
                fn = None
                eles = ele_str.strip().split('/')
                if len(eles) == 1:
                    fv = int(eles[0]) - 1
                elif len(eles) == 2:
                    fv = int(eles[0]) - 1
                    ft = int(eles[1]) - 1
                elif len(eles) == 3:
                    fv = int(eles[0]) - 1
                    fn = int(eles[2]) - 1
                    ft = None if eles[1] == '' else int(eles[1]) - 1
                return fv, ft, fn

            fv0, ft0, fn0 = segElementData(line_data[1])
            fv1, ft1, fn1 = segElementData(line_data[2])
            fv2, ft2, fn2 = segElementData(line_data[3])
            f_list.append((fv0, fv1, fv2))
            if ft0 is not None and ft1 is not None and ft2 is not None:
                ft_list.append((ft0, ft1, ft2))
            if fn0 is not None and fn1 is not None and fn2 is not None:
                fn_list.append((fn0, fn1, fn2))

    v_list = np.asarray(v_list)
    vn_list = np.asarray(vn_list)
    vt_list = np.asarray(vt_list)
    vc_list = np.asarray(vc_list)
    f_list = np.asarray(f_list)
    ft_list = np.asarray(ft_list)
    fn_list = np.asarray(fn_list)

    model = {'v': v_list, 'vt': vt_list, 'vc': vc_list, 'vn': vn_list,
             'f': f_list, 'ft': ft_list, 'fn': fn_list}
    return model

def save_obj_data(model, filename, log=True):
    import numpy as np
    assert 'v' in model and model['v'].size != 0

    with open(filename, 'w') as fp:
        if 'v' in model and model['v'].size != 0:
            if 'vc' in model and model['vc'].size != 0:
                assert model['vc'].size == model['v'].size
                for v, vc in zip(model['v'], model['vc']):
                    fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], vc[2], vc[1], vc[0]))
            else:
                for v in model['v']:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if 'vn' in model and model['vn'].size != 0:
            for vn in model['vn']:
                fp.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))

        if 'vt' in model and model['vt'].size != 0:
            for vt in model['vt']:
                fp.write('vt %f %f\n' % (vt[0], vt[1]))

        if 'f' in model and model['f'].size != 0:
            if 'fn' in model and model['fn'].size != 0 and 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['fn'].size
                assert model['f'].size == model['ft'].size
                for f_, ft_, fn_ in zip(model['f'], model['ft'], model['fn']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                             (f[0], ft[0], fn[0], f[1], ft[1], fn[1], f[2], ft[2], fn[2]))
            elif 'fn' in model and model['fn'].size != 0:
                assert model['f'].size == model['fn'].size
                for f_, fn_ in zip(model['f'], model['fn']):
                    f = np.copy(f_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d//%d %d//%d %d//%d\n' % (f[0], fn[0], f[1], fn[1], f[2], fn[2]))
            elif 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['ft'].size
                for f_, ft_ in zip(model['f'], model['ft']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fp.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
            else:
                for f_ in model['f']:
                    f = np.copy(f_) + 1
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if log:
        print("Saved mesh as " + filename)

def save_obj(path, v, f=None, c=None):
    with open(path, 'w') as file:
        for i in range(len(v)):
            if c is not None:
                file.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))
            else:
                file.write('v %f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], 1, 1, 1))

        file.write('\n')
        if f is not None:
            for i in range(len(f)):
                file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    file.close()


def intrmat2pmat(K, img_w, img_h, znear=0.01, zfar=100.):
    ## 要求所有量均为正值

    mat_p = np.zeros((4, 4), dtype=np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    mat_p[0, 0] = 2.0 * fx / img_w
    mat_p[0, 2] = 1. - 2.0 * cx / img_w
    mat_p[1, 1] = 2.0 * fy / img_h
    mat_p[1, 2] = 2.0 * cy / img_h - 1.

    if False:  # pytorch3d
        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space postive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.
        mat_p[3, 2] = z_sign

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane.
        mat_p[2, 2] = z_sign * zfar / (zfar - znear)
        mat_p[2, 3] = -(zfar * znear) / (zfar - znear)
    else:  # opengl
        z_sign = -1.
        mat_p[3, 2] = z_sign
        mat_p[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
        mat_p[2, 3] = -2. * znear * zfar / (zfar - znear)

    return mat_p.astype(np.float32)


def print_args(opt):
    args = vars(opt)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')


def find_bbox(mask):
    """
    function that takes synthetic rendering of the track head, finds its bbox, enlarges it, and returns the coords
    relative to image size (multiply by im dimensions to get pixel location of enlarged bbox

    :param im:
    :return:
    """

    H, W = np.shape(mask)
    where = np.where(mask > 0)
    h_min = np.min(where[0])
    h_max = np.max(where[0])
    w_min = np.min(where[1])
    w_max = np.max(where[1])
    # print(H, W, h_min, h_max, w_min, w_max)

    h_span = h_max - h_min
    w_span = w_max - w_min
    ratio = 0.5
    h_min -= ratio * 0.8 * h_span
    h_max += ratio * 0.8 * h_span
    w_min -= ratio * 0.5 * w_span
    w_max += ratio * 0.5 * w_span

    h_min = int(np.clip(h_min, 0, H - 1))
    h_max = int(np.clip(h_max, 0, H - 1))
    w_min = int(np.clip(w_min, 0, W - 1))
    w_max = int(np.clip(w_max, 0, W - 1))
    # print(H, W, h_min, h_max, w_min, w_max)
    return np.array([h_min / H, h_max / H, w_min / W, w_max / W])


def normalize(x, axis=-1, eps=1e-8):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    out = x / (norm + eps)
    out[np.repeat(norm <= eps, x.shape[axis], axis)] = 0
    return out


def get_normal_map(pos_map):
    p_ctr = pos_map[1:-1, 1:-1, :]
    vw = p_ctr - pos_map[1:-1, 2:, :]
    vs = pos_map[2:, 1:-1, :] - p_ctr
    ve = p_ctr - pos_map[1:-1, :-2, :]
    vn = pos_map[:-2, 1:-1, :] - p_ctr
    normal_1 = normalize(np.cross(vs, vw))  # (H-2,W-2,3)
    normal_2 = normalize(np.cross(vw, vn))
    normal_3 = normalize(np.cross(vn, ve))
    normal_4 = normalize(np.cross(ve, vs))
    normal = normal_1 + normal_2 + normal_3 + normal_4
    normal = normalize(normal, axis=2)
    normal = np.pad(normal, ((1, 1), (1, 1), (0, 0)), 'constant')  # (H,W,3)
    return normal


def get_normal_map_torch(p):
    normalize = lambda array: torch.nn.functional.normalize(array, p=2, dim=3, eps=1e-8)
    p_ctr = p[:, 1:-1, 1:-1, :]
    vw = p_ctr - p[:, 1:-1, 2:, :]
    vs = p[:, 2:, 1:-1, :] - p_ctr
    ve = p_ctr - p[:, 1:-1, :-2, :]
    vn = p[:, :-2, 1:-1, :] - p_ctr
    normal_1 = torch.cross(vs, vw)  # (B,H-2,W-2,3)
    normal_2 = torch.cross(vn, ve)
    normal_1 = normalize(normal_1)
    normal_2 = normalize(normal_2)
    normal = normal_1 + normal_2
    normal = normalize(normal)
    paddings = (0, 0, 1, 1, 1, 1, 0, 0)
    normal = torch.nn.functional.pad(normal, paddings, 'constant')  # (B,H,W,3)
    return normal  # (B,H,W,3)


def map_depth_to_3D(depth, mask, K_inv, T_inv=np.eye(4), mode='k4a'):
    colidx = np.arange(depth.shape[1])
    rowidx = np.arange(depth.shape[0])
    colidx_map, rowidx_map = np.meshgrid(colidx, rowidx)
    colidx_map = colidx_map.astype(np.float)  # + 0.5
    rowidx_map = rowidx_map.astype(np.float)  # + 0.5
    col_indices = colidx_map[mask > 0]
    # row_indices = (depth.shape[0] - rowidx_map)[mask > 0]  ####
    row_indices = rowidx_map[mask > 0]  # if mode == 'k4a' else (depth.shape[0] - rowidx_map)[mask > 0]
    homo_padding = np.ones((col_indices.shape[0], 1), dtype=np.float32)
    homo_indices = np.concatenate((col_indices[..., None], row_indices[..., None], homo_padding), axis=1)

    normalized_points = K_inv[None, ...] @ homo_indices[..., None]

    # z_values = (depth / 1000)[mask > 0]
    z_values = depth[mask > 0]

    valid_points = normalized_points.squeeze() * z_values[..., None]
    # print('cam_K', valid_points[:, 1].max() - valid_points[:, 1].min())
    # if mode == 'opengl':
    #     valid_points[:, 2] = -valid_points[:, 2]  ###
    R = T_inv[:3, :3]
    t = T_inv[:3, 3]
    points = R[None, ...] @ valid_points[..., None] + t[None, ..., None]
    points = points.squeeze()
    # print('cam_T', points[:, 1].max() - points[:, 1].min())
    return points


def depth2normal_perse(depth, intr):
    # depth: [B,H,W]
    # intrinsics: [fx, fy, cx, cy]
    normalize = lambda array: torch.nn.functional.normalize(array, p=2, dim=3, eps=1e-8)
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    B, H, W = depth.shape
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy

    Y, X = torch.meshgrid(torch.tensor(range(H)), torch.tensor(range(W)))
    X = X.unsqueeze(0).repeat(B, 1, 1).float()  # (B,H,W)
    Y = Y.unsqueeze(0).repeat(B, 1, 1).float()

    x_cord_p = (X - cx) * inv_fx * depth
    y_cord_p = (Y - cy) * inv_fy * depth

    p = torch.stack([x_cord_p, y_cord_p, depth], dim=3)  # (B,H,W,3)

    # vector of p_3d in west, south, east, north direction
    p_ctr = p[:, 1:-1, 1:-1, :]
    vw = p_ctr - p[:, 1:-1, 2:, :]
    vs = p[:, 2:, 1:-1, :] - p_ctr
    ve = p_ctr - p[:, 1:-1, :-2, :]
    vn = p[:, :-2, 1:-1, :] - p_ctr
    normal_1 = torch.cross(vs, vw)  # (B,H-2,W-2,3)
    normal_2 = torch.cross(vn, ve)
    normal_1 = normalize(normal_1)
    normal_2 = normalize(normal_2)
    normal = normal_1 + normal_2
    normal = normalize(normal)
    paddings = (0, 0, 1, 1, 1, 1, 0, 0)
    normal = torch.nn.functional.pad(normal, paddings, 'constant')  # (B,H,W,3)
    return normal  # (B,H,W,3)


def depth2normal_ortho(depth, dx, dy):
    # depth: [B,H,W]
    B, H, W = depth.shape
    normalize = lambda array: torch.nn.functional.normalize(array, p=2, dim=3, eps=1e-8)
    Y, X = torch.meshgrid(torch.tensor(range(H)), torch.tensor(range(W)))
    X = X.unsqueeze(0).repeat(B, 1, 1).float()  # (B,H,W)
    Y = Y.unsqueeze(0).repeat(B, 1, 1).float()

    x_cord = X * dx
    y_cord = Y * dy
    p = torch.stack([x_cord, y_cord, depth], dim=3)  # (B,H,W,3)
    # vector of p_3d in west, south, east, north direction
    p_ctr = p[:, 1:-1, 1:-1, :]
    vw = p_ctr - p[:, 1:-1, 2:, :]
    vs = p[:, 2:, 1:-1, :] - p_ctr
    ve = p_ctr - p[:, 1:-1, :-2, :]
    vn = p[:, :-2, 1:-1, :] - p_ctr
    normal_1 = torch.cross(vs, vw)  # (B,H-2,W-2,3)
    normal_2 = torch.cross(vn, ve)
    normal_1 = normalize(normal_1)
    normal_2 = normalize(normal_2)
    normal = normal_1 + normal_2
    normal = normalize(normal)
    paddings = (0, 0, 1, 1, 1, 1, 0, 0)
    normal = torch.nn.functional.pad(normal, paddings, 'constant')  # (B,H,W,3)
    return normal  # (B,H,W,3)


def draw_pupil(img, pred_lms, pupil_r, pupil_r_flag, pupil_l, pupil_l_flag):
    if pupil_r_flag:
        center_eye_l = pred_lms[36]
        center_eye_r = pred_lms[39]
        center_eye_u = pred_lms[37] / 2 + pred_lms[38] / 2
        center_eye_d = pred_lms[40] / 2 + pred_lms[41] / 2
        center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4
        pupil = center_eye + (center_eye_r - center_eye_l) * (pupil_r[0] + 0.0) + (center_eye_d - center_eye_u) * pupil_r[1]
        pupil = (pupil + 0.5).astype(np.int32)
        cv2.circle(img, (int(pupil[0]), int(pupil[1])), 3, [0, 255, 0], -1)
    if pupil_l_flag:
        center_eye_l = pred_lms[42]
        center_eye_r = pred_lms[45]
        center_eye_u = pred_lms[43] / 2 + pred_lms[44] / 2
        center_eye_d = pred_lms[46] / 2 + pred_lms[47] / 2
        center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4
        pupil = center_eye + (center_eye_r - center_eye_l) * (pupil_l[0] - 0.0) + (center_eye_d - center_eye_u) * pupil_l[1]
        pupil = (pupil + 0.5).astype(np.int32)
        cv2.circle(img, (int(pupil[0]), int(pupil[1])), 3, [0, 255, 0], -1)
    return img

import matplotlib.pyplot as plt
distance = lambda a, b: np.sqrt(np.sum(np.square(a - b)))
def get_pupil(img, lms, x_grid, y_grid, thresh=30, disp_ratio=0.15):
    height, width, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(img_gray.shape, np.uint8)
    lms_this = lms.astype(np.int64)

    # right eye
    ptsr = lms_this[36:42]  # .reshape((-1, 1, 2))
    mask_r = cv2.polylines(mask.copy(), [ptsr], True, 1)
    mask_r = cv2.fillPoly(mask_r, [ptsr], 1)
    img_eye_r = img_gray * mask_r + (1 - mask_r) * 255
    thres = int(np.min(img_eye_r)) + thresh
    mask_r = mask_r.astype(np.float32) * (img_eye_r < thres).astype(np.float32)
    # if np.sum(mask_r) < 10:
    #     pupil_r, pupil_r_flag = np.array([0.0, 0.0], dtype=np.float32), False
    # else:
    r_eye_x = np.sum(x_grid * mask_r) / np.sum(mask_r)
    r_eye_y = np.sum(y_grid * mask_r) / np.sum(mask_r)

    pupil = np.array([r_eye_x, r_eye_y], dtype=np.float32)
    # print(pupil)
    center_eye_l = lms_this[36]
    center_eye_r = lms_this[39]
    center_eye_u = lms_this[37] / 2 + lms_this[38] / 2
    center_eye_d = lms_this[40] / 2 + lms_this[41] / 2
    center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4

    dis1_l = distance(center_eye_l, center_eye_r)
    dis2_l = distance(center_eye_u, center_eye_d)
    # print(dis2_l, dis1_l)
    if dis2_l / dis1_l < disp_ratio:
        pupil_r, pupil_r_flag = np.array([0.0, 0.0], dtype=np.float32), False
    else:
        eye1_l = np.dot(pupil - center_eye, center_eye_r - center_eye_l) / dis1_l ** 2
        eye2_l = np.dot(pupil - center_eye, center_eye_d - center_eye_u) / dis2_l ** 2
        pupil_r = np.array([eye1_l, eye2_l], dtype=np.float32)
        pupil_r_flag = True

    # left eye
    ptsl = lms_this[42:48]  # .reshape((-1, 1, 2))
    mask_l = cv2.polylines(mask.copy(), [ptsl], True, 1)
    mask_l = cv2.fillPoly(mask_l, [ptsl], 1)
    img_eye_l = img_gray * mask_l + (1 - mask_l) * 255
    thres = int(np.min(img_eye_l)) + thresh
    mask_l = mask_l.astype(np.float32) * (img_eye_l < thres).astype(np.float32)
    # if np.sum(mask_l) < 10:
    #     pupil_l, pupil_l_flag = np.array([0.0, 0.0], dtype=np.float32), False
    # else:
    l_eye_x = np.sum(x_grid * mask_l) / np.sum(mask_l)
    l_eye_y = np.sum(y_grid * mask_l) / np.sum(mask_l)
    pupil = np.array([l_eye_x, l_eye_y], dtype=np.float32)

    center_eye_l = lms_this[42]
    center_eye_r = lms_this[45]
    center_eye_u = lms_this[43] / 2 + lms_this[44] / 2
    center_eye_d = lms_this[46] / 2 + lms_this[47] / 2
    center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4

    dis1_l = distance(center_eye_l, center_eye_r)
    dis2_l = distance(center_eye_u, center_eye_d)
    if dis2_l / dis1_l < disp_ratio:
        pupil_l, pupil_l_flag = np.array([0.0, 0.0], dtype=np.float32), False
    else:
        eye1_l = np.dot(pupil - center_eye, center_eye_r - center_eye_l) / dis1_l ** 2
        eye2_l = np.dot(pupil - center_eye, center_eye_d - center_eye_u) / dis2_l ** 2
        pupil_l = np.array([eye1_l, eye2_l], dtype=np.float32)
        pupil_l_flag = True
    return pupil_r, pupil_r_flag, pupil_l, pupil_l_flag


def get_pupil_gazeTracking(frame, lms, gaze, blinking_thresh=3.8):
    # blinking_thresh = 4.8
    gaze.refresh(frame)
    print(gaze.eye_left.blinking, gaze.eye_right.blinking)
    if gaze.pupil_left_coords() is None or gaze.pupil_left_coords() is None:
        return [None] * 4
    mask = np.zeros(frame.shape, np.uint8)
    lms_this = lms.astype(np.int64)
    # right eye
    ptsr = lms_this[36:42]  # .reshape((-1, 1, 2))
    mask_r = cv2.polylines(mask.copy(), [ptsr], True, 1)
    mask_r = cv2.fillPoly(mask_r, [ptsr], 1)
    if gaze.eye_left.blinking > blinking_thresh:
        pupil_r, pupil_r_flag = np.array([0.0, 0.0], dtype=np.float32), False
    else:
        pupil = np.array(gaze.pupil_left_coords(), dtype=np.float32)    #二者定义相反
        if mask_r[int(pupil[1]), int(pupil[0]), 0] == 0:
            return [None] * 4
        # print(pupil)
        center_eye_l = lms_this[36]
        center_eye_r = lms_this[39]
        center_eye_u = lms_this[37] / 2 + lms_this[38] / 2
        center_eye_d = lms_this[40] / 2 + lms_this[41] / 2
        center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4

        dis1_l = distance(center_eye_l, center_eye_r)
        dis2_l = distance(center_eye_u, center_eye_d)
        if dis2_l / dis1_l < 0.1:
            pupil_r, pupil_r_flag = np.array([0.0, 0.0], dtype=np.float32), False
        else:
            eye1_l = np.dot(pupil - center_eye, center_eye_r - center_eye_l) / dis1_l ** 2
            eye2_l = np.dot(pupil - center_eye, center_eye_d - center_eye_u) / dis2_l ** 2
            pupil_r = np.array([eye1_l, eye2_l], dtype=np.float32)
            pupil_r_flag = True

    # left eye
    ptsl = lms_this[42:48]  # .reshape((-1, 1, 2))
    mask_l = cv2.polylines(mask.copy(), [ptsl], True, 1)
    mask_l = cv2.fillPoly(mask_l, [ptsl], 1)
    if gaze.eye_right.blinking > blinking_thresh:
        pupil_l, pupil_l_flag = np.array([0.0, 0.0], dtype=np.float32), False
    else:
        pupil = np.array(gaze.pupil_right_coords(), dtype=np.float32)
        # print(pupil, mask_l.shape)
        if mask_l[int(pupil[1]), int(pupil[0]), 0] == 0:
            return [None] * 4
        center_eye_l = lms_this[42]
        center_eye_r = lms_this[45]
        center_eye_u = lms_this[43] / 2 + lms_this[44] / 2
        center_eye_d = lms_this[46] / 2 + lms_this[47] / 2
        center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4

        dis1_l = distance(center_eye_l, center_eye_r)
        dis2_l = distance(center_eye_u, center_eye_d)
        if dis2_l / dis1_l < 0.1:
            pupil_l, pupil_l_flag = np.array([0.0, 0.0], dtype=np.float32), False
        else:
            eye1_l = np.dot(pupil - center_eye, center_eye_r - center_eye_l) / dis1_l ** 2
            eye2_l = np.dot(pupil - center_eye, center_eye_d - center_eye_u) / dis2_l ** 2
            pupil_l = np.array([eye1_l, eye2_l], dtype=np.float32)
            pupil_l_flag = True
    return pupil_r, pupil_r_flag, pupil_l, pupil_l_flag


def tougue_detect(img, lms, x_grid, y_grid, disp_ratio=0.1, region_ration=0.05):
    '''
    lms:[左嘴角58，上嘴唇50，右嘴角62，下颚(10, 8, 6)] 逆时针
    '''

    dis1_l = distance(lms[64], lms[60])
    dis2_l = distance(lms[58], lms[62])
    if dis1_l / dis2_l < disp_ratio:    # 闭嘴状态
        tougue, tougue_flag = np.array([0.0, 0.0], dtype=np.float32), False
    else:
        lms_this = lms.astype(np.int64)
        ptsr = np.stack([lms_this[58], lms_this[50], lms_this[62], lms_this[10], lms_this[8], lms_this[6]], 0)
        # ptsr = np.stack([lms_this[58], lms_this[53], lms_this[8]], 0)
        # ptsr = lms_this[48:60]
        # print(lms.shape)
        #
        # for i in range(58, 66):
        #     cv2.circle(img,
        #                (round(lms_this[i, 0]), round(lms_this[i, 1])), 1,
        #                (255, 0, 0), -1)
        # for i in range(58, 66):
        #     cv2.putText(img, str(i),
        #                 (round(lms_this[i, 0]), round(lms_this[i, 1])),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
        # #
        # plt.imshow(img[:, :, ::-1])
        # plt.show()
        # exit(0)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        h = ((h.astype(np.float32) + 90) % 180).astype(np.uint8)
        h = ((h.astype(np.float32).clip(80, 120) - 80) / 40 * 255).astype(np.uint8)
        h = cv2.bilateralFilter(h, 30, 20, 15)

        mask_r = np.zeros(h.shape, np.uint8)
        mask_r = cv2.polylines(mask_r, [ptsr], True, 1)
        mask_r = cv2.fillPoly(mask_r, [ptsr], 1)

        mask_tougue = cv2.threshold(h, 100, 1.0, cv2.THRESH_BINARY_INV)[1]
        mask_tougue = mask_tougue * mask_r  # [H, W]

        plt.imshow(np.concatenate([(mask_tougue[:, :, None] * img).astype(np.uint8), img], 1)[:, :, ::-1])
        plt.show()
        exit(0)
        # if np.sum(mask_tougue) < np.sum(mask_r) * region_ration:
        #     return np.array([0.0, 0.0], dtype=np.float32), False

        # center_tougue_x = np.sum(x_grid * mask_tougue) / np.sum(mask_tougue)
        center_tougue_y = np.sum(y_grid * mask_tougue) / np.sum(mask_tougue)

        # line_mask = mask_tougue[int(center_tougue_y)]
        # centerLine_numPixel = np.sum(line_mask)
        tougue_y = mask_tougue.nonzero()[0].max()
        for i in range(int(center_tougue_y), mask_tougue.nonzero()[0].max() + 1):
            if np.sum(mask_tougue[i]) < 0.5 * np.sum(mask_tougue[int(center_tougue_y)]):
                tougue_y = i - 1
                break

        tougue_x = np.sum(x_grid[tougue_y] * mask_tougue[tougue_y]) / np.sum(mask_tougue[tougue_y])

        tougue = np.array([tougue_x, tougue_y], dtype=np.float32)
        tougue_u = lms_this[50]
        tougue_d = lms_this[8]
        tougue_r = lms_this[62]
        tougue_l = lms_this[58]
        center_tougue = (tougue_u + tougue_r + tougue_d + tougue_l) / 4

        tougue_1 = np.dot(tougue - center_tougue, tougue_u - tougue_d) / distance(tougue_u, tougue_d) ** 2
        tougue_2 = np.dot(tougue - center_tougue, tougue_r - tougue_l) / distance(tougue_r, tougue_l) ** 2
        tougue = np.array([tougue_1, tougue_2], dtype=np.float32)
        tougue_flag = True
    return tougue, tougue_flag


def draw_tougue(img, pred_lms, tougue, tougue_flag):
    if tougue_flag:
        tougue_u = pred_lms[50]
        tougue_d = pred_lms[8]
        tougue_r = pred_lms[62]
        tougue_l = pred_lms[58]
        center_tougue = (tougue_u + tougue_r + tougue_d + tougue_l) / 4
        tougue = center_tougue + (tougue_u - tougue_d) * tougue[0] + (tougue_r - tougue_l) * tougue[1]
        tougue = (tougue + 0.5).astype(np.int32)
        # cv2.circle(img, (int(tougue[0]), int(tougue[1])), 3, [0, 255, 0], -1)
        # img = cv2.polylines(img, [np.stack([pred_lms[58], pred_lms[62], tougue], 0)], True, 1)
        img = cv2.fillPoly(img, [np.stack([pred_lms[58], pred_lms[62], tougue], 0).astype(np.int32)], color=[255, 0, 0])
    return img

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
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
