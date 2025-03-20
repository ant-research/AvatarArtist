import torch
import torch.nn.functional as F
from .aemodules3d import SamePadConv3d
from .utils.common_utils import instantiate_from_config
from .distributions import DiagonalGaussianDistribution
from einops import rearrange

def conv3d(in_channels, out_channels, kernel_size, conv3d_type='SamePadConv3d'):
    if conv3d_type == 'SamePadConv3d':
        return SamePadConv3d(in_channels, out_channels, kernel_size=kernel_size, padding_type='replicate')
    else:
        raise NotImplementedError


class AutoencoderKL(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 std=1.,
                 mean=0.,
                 prob=0.2,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = instantiate_from_config(ddconfig['encoder'])
        self.decoder = instantiate_from_config(ddconfig['decoder'])
        # self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.std = std
        self.mean = mean
        self.prob = prob
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        try:
            self._cur_epoch = sd['epoch']
            sd = sd["state_dict"]
        except:
            pass
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, **kwargs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True, **kwargs):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 4:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    # def training_step(self, inputs):
    #
    #     reconstructions, posterior = self(inputs)
    #     aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior,
    #                                     last_layer=self.get_last_layer(), split="train")
    #
    #     return aeloss, log_dict_ae

    # def validation_step(self, batch, batch_idx):
    #     inputs = self.get_input(batch, self.image_key)
    #     reconstructions, posterior = self(inputs)
    #     aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
    #                                     last_layer=self.get_last_layer(), split="val")
    #
    #     discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="val")
    #
    #     self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
    #     self.log_dict(log_dict_ae)
    #     self.log_dict(log_dict_disc)
    #     return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class AutoencoderKLRollOut(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 std=1.,
                 mean=0.,
                 prob=0.2,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = instantiate_from_config(ddconfig['encoder'])
        self.decoder = instantiate_from_config(ddconfig['decoder'])
        # self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.std = std
        self.mean = mean
        self.prob = prob
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        try:
            self._cur_epoch = sd['epoch']
            sd = sd["state_dict"]
        except:
            pass
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    def rollout(self, triplane):

        triplane = rearrange(triplane, "b c f h w -> b f c h w")
        b, f, c, h, w = triplane.shape
        triplane = triplane.permute(0, 2, 3, 1, 4).reshape(-1, c, h, f * w)
        return triplane
    def unrollout(self, triplane):
        res = triplane.shape[-2]
        ch = triplane.shape[1]
        triplane = triplane.reshape(-1, ch // 3, res, 3, res).permute(0, 3, 1, 2, 4).reshape(-1, 3, ch, res, res)
        triplane = rearrange(triplane, "b f c h w -> b c f h w")
        return triplane

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, **kwargs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True, **kwargs):

        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 4:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    # def training_step(self, inputs):
    #
    #     reconstructions, posterior = self(inputs)
    #     aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior,
    #                                     last_layer=self.get_last_layer(), split="train")
    #
    #     return aeloss, log_dict_ae

    # def validation_step(self, batch, batch_idx):
    #     inputs = self.get_input(batch, self.image_key)
    #     reconstructions, posterior = self(inputs)
    #     aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
    #                                     last_layer=self.get_last_layer(), split="val")
    #
    #     discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="val")
    #
    #     self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
    #     self.log_dict(log_dict_ae)
    #     self.log_dict(log_dict_disc)
    #     return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

