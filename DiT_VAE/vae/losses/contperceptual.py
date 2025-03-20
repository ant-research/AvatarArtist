import torch
import torch.nn as nn
from einops import rearrange
from rendering import RenderingClass
from .lpips_loss import LPIPS

class LPIPSithTVLoss(nn.Module):
    def __init__(self, lpips_ckpt, vgg_ckpt, device, render_network_pkl, std_v, mean_v, ws_avg_pkl, logvar_init=0.0,
                 kl_weight=1e-5, faceloss_weight=1.0,
                 pixelloss_weight=1.0, depthloss_weight=0, face_feature_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, percep_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, imgloss_weight=0.,
                 disc_loss="hinge", max_bs=None, latent_tv_weight=2e-3, use_double_norm=False, use_max_min=False,
                 use_std_mean=True):

        super().__init__()
        self.kl_weight = kl_weight
        self.face_feature_weight = face_feature_weight
        self.pixel_weight = pixelloss_weight
        self.face_weight = faceloss_weight

        self.depthloss_weight = depthloss_weight
        self.latent_tv_weight = latent_tv_weight
        self.Rendering = RenderingClass(device, render_network_pkl, ws_avg_pkl)
        self.perceptual_loss = LPIPS(lpips_ckpt, vgg_ckpt).eval().to(device)
        self.perceptual_weight = perceptual_weight
        self.imgloss_weight = imgloss_weight
        self.std_v = torch.load(std_v)
        self.mean_v = torch.load(mean_v)

        self.use_std_mean = use_std_mean
        self.std_v = self.std_v.reshape(1, -1, 1, 1, 1).to(device)
        self.mean_v = self.mean_v.reshape(1, -1, 1, 1, 1).to(device)
        self.percep_factor = percep_factor
        # self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def denormolize(self, inputs_rendering, reconstructions_rendering):
        inputs_rendering = inputs_rendering * self.std_v + self.mean_v
        reconstructions_rendering = reconstructions_rendering * self.std_v + self.mean_v
        return inputs_rendering, reconstructions_rendering

    def forward(self, inputs, reconstructions, posteriors, vert_values,
                split="train"):
        inputs_rendering = rearrange(inputs, 'b c t h w -> b t c h w')
        reconstructions_rendering = rearrange(reconstructions, 'b c t h w -> b t c h w')

        # if inputs.dim() == 5:
        #     inputs = rearrange(inputs, 'b c t h w -> (b t) c h w')
        # if reconstructions.dim() == 5:
        #     reconstructions = rearrange(reconstructions, 'b c t h w -> (b t) c h w')
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]
        log = {"{}/rec_loss".format(split): rec_loss.detach().mean()}
        loss = self.pixel_weight * rec_loss

        inputs_rendering_original, reconstructions_rendering_original = self.denormolize(inputs_rendering,
                                                                                         reconstructions_rendering)

        inputs_img, recon_img, inputs_depth, recon_depth, inputs_face_feature, recon_face_feature = self.Rendering.rendering_for_training(
            reconstructions_rendering_original.contiguous(),
            inputs_rendering_original.contiguous(), vert_values)

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(recon_img.contiguous(), inputs_img.contiguous())
            p_loss = torch.sum(p_loss) / p_loss.shape[0]
            loss += self.perceptual_weight * p_loss
            log["{}/percep_loss".format(split)] = p_loss.detach().mean()

        if self.depthloss_weight > 0:
            rec_loss_depth = torch.abs(recon_depth.contiguous() - inputs_depth.contiguous())
            rec_loss_depth = torch.sum(rec_loss_depth) / rec_loss_depth.shape[0]
            loss += self.depthloss_weight * rec_loss_depth
            log["{}/depth_loss".format(split)] = rec_loss_depth.detach().mean()

        if self.latent_tv_weight > 0:
            latent = posteriors.mean
            latent_tv_y = torch.abs(latent[:, :, :-1] - latent[:, :, 1:]).sum() / latent.shape[0]
            latent_tv_x = torch.abs(latent[:, :, :, :-1] - latent[:, :, :, 1:]).sum() / latent.shape[0]
            latent_tv_loss = latent_tv_y + latent_tv_x
            loss += latent_tv_loss * self.latent_tv_weight
            log["{}/tv_loss".format(split)] = latent_tv_loss.detach().mean()



        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss += kl_loss * self.kl_weight
        log["{}/kl_loss".format(split)] = kl_loss.detach().mean()

        log["{}/total_loss".format(split)] = loss.detach().mean()

        return loss, log


