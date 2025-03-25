import numpy as np
import torch
import torch.nn.functional as F


def photo_loss(pred_img, gt_img, img_mask):
    pred_img = pred_img.float()
    loss = torch.sqrt(torch.sum(torch.square(
        pred_img - gt_img), 3))*img_mask/255
    loss = torch.sum(loss, dim=(1, 2)) / torch.sum(img_mask, dim=(1, 2))
    loss = torch.mean(loss)

    return loss


def lm_loss(pred_lms, gt_lms, weight, img_size=224, keep_dim=False, enhance_pupil=False):
    loss = torch.sum(torch.square(pred_lms/img_size - gt_lms / img_size), dim=2) * weight.reshape(1, -1)
    if enhance_pupil:
        loss[:, 468:] *= 10
    if not keep_dim:
        loss = torch.mean(loss.sum(1))

    return loss


def get_l2(tensor, reduction='sum'):
    if reduction == 'sum':
        return torch.square(tensor).sum()
    else:
        return torch.square(tensor).mean()

def reflectance_loss(tex, skin_mask):

    skin_mask = skin_mask.unsqueeze(2)
    tex_mean = torch.sum(tex*skin_mask, 1, keepdims=True)/torch.sum(skin_mask)
    loss = torch.sum(torch.square((tex-tex_mean)*skin_mask/255.)) / \
        (tex.shape[0]*torch.sum(skin_mask))

    return loss


def gamma_loss(gamma):

    gamma = gamma.reshape(-1, 3, 9)
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    gamma_loss = torch.mean(torch.square(gamma - gamma_mean))

    return gamma_loss
