from __future__ import absolute_import
import torch
from torch import nn
from .encoder_id import Backbone
import torch.nn.functional as F


class IDLoss(nn.Module):
    def __init__(self, model_path, num_scales=1):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.num_scales = num_scales


        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, y):
        n_samples = x.shape[0]
        loss = 0.0
        for _scale in range(self.num_scales):
            x_feats = self.extract_feats(x)
            y_feats = self.extract_feats(y)
            for i in range(n_samples):
                diff_target = y_feats[i].dot(x_feats[i])
                loss += 1 - diff_target

            if _scale != self.num_scales - 1:
                x = F.interpolate(x, mode='bilinear', scale_factor=0.5, align_corners=False,
                                  recompute_scale_factor=True)
                y = F.interpolate(y, mode='bilinear', scale_factor=0.5, align_corners=False,
                                  recompute_scale_factor=True)
        return loss / n_samples
    def psp_forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs