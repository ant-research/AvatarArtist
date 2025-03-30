import torch
import numpy as np
from skimage import transform as trans
from kornia.geometry import warp_affine
import torch.nn.functional as F


def extract_3p_flame(lm):
    p0 = lm[36:42].mean(0)
    p1 = lm[42:48].mean(0)
    p2 = lm[60:68].mean(0)
    lm3p = np.stack([p0, p1, p2], axis=0)  # (3,2)
    return lm3p


def estimate_norm_pdfgc(lm_70p, H, reverse_y=True):
    # modified from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m            --numpy.array  (2, 3)
    Parameters:
        lm                 --numpy.array  (70, 2), y direction is opposite to v direction
        H                  --int/float , image height
    """
    lm = extract_3p_flame(lm_70p)
    if reverse_y:
        lm[:, -1] = H - 1 - lm[:, -1]
    tform = trans.SimilarityTransform()
    src = np.array([[87, 59], [137, 59], [112, 120]], dtype=np.float32)  # in size of 224
    tform.estimate(lm, src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]
def estimate_norm_torch_pdfgc(lm_70p, H, reverse_y=True):
    lm_70p_ = lm_70p.detach().cpu().numpy()
    M = []
    for i in range(lm_70p_.shape[0]):
        M.append(estimate_norm_pdfgc(lm_70p_[i], H, reverse_y=reverse_y))
    M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_70p.device)
    return M

def get_motion_feature(pd_fgc, imgs, lmks, crop_size=224, crop_len=16, reverse_y=False):
    trans_m = estimate_norm_torch_pdfgc(lmks, imgs.shape[-1], reverse_y=reverse_y)
    imgs_warp = warp_affine(imgs, trans_m, dsize=(224, 224))
    imgs_warp = imgs_warp[:, :, :crop_size - crop_len * 2, crop_len:crop_size - crop_len]
    imgs_warp = torch.clamp(F.interpolate(imgs_warp, size=[crop_size, crop_size], mode='bilinear'), -1, 1)

    out = pd_fgc(imgs_warp)
    motions = torch.cat([out[1], out[2], out[3]], dim=-1)

    return motions