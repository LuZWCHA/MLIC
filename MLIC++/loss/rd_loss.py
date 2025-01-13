import math
import lpips.pretrained_networks
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

# from loss import perceptual_loss as ps

from models.vgg import Vgg16 
from lpips import lpips


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        loss = torch.mean(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.charbonnier = CharbonnierLoss()
        self.lmbda = lmbda
        self.metrics = metrics

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metrics == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == 'ms-ssim':
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]
        elif self.metrics == 'charbonnier':
            out["mse_loss"] = self.charbonnier(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out

class RateDistortionLossExp(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.charbonnier = CharbonnierLoss()
        self.lmbda = lmbda
        self.metrics = metrics

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            [(torch.log(likelihoods).sum(dim=(1,2,3)) / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()]
        )
        
        if self.metrics == 'mse':
            out["mse_loss"] = torch.sum(self.mse(output["x_hat"], target),dim=(1,2,3))
            # out["ms_ssim_loss"] = None
            # print(N, out["mse_loss"].shape, out["bpp_loss"].shape, output["likelihoods"].keys())
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == 'ms-ssim':
            # out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]
        elif self.metrics == 'charbonnier':
            out["mse_loss"] = self.charbonnier(output["x_hat"], target)
            # out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type = 'hinge',
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """

        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False, mask=None):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the target is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """

        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.mae = nn.L1Loss()

    def gram_matrix(self, x: torch.Tensor):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def forward(self,target_feature, inputs):
        G = self.gram_matrix(inputs)
        target = self.gram_matrix(target_feature)
        self.loss = self.mae(G, target)
        return self.loss 


class RateDistortionPOELICLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.gan = GANLoss()
        self.style = StyleLoss()
        self.lpips =  lpips.LPIPS(net='vgg', verbose=False, eval_mode=True)

        self.lmbda = lmbda
        self.vgg = Vgg16().eval()

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda
        
    def forward(self, output, target: torch.Tensor, mask=1):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        x_hat = output["x_hat"]
        x_tidle: torch.Tensor = mask * target + (1-mask) * x_hat
        out["x_tidle"] = x_tidle
        
        kernel_h = 16
        kernel_w = 16
        
        x_tidle_patch = x_tidle.unfold(3, kernel_h, kernel_w).unfold(2, kernel_h, kernel_w).permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, kernel_h, kernel_w)
        target_patch = target.unfold(3, kernel_h, kernel_w).unfold(2, kernel_h, kernel_w).permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, kernel_h, kernel_w)
        # (N, C, kernel_h, kernel_w)，其中 N = (H // kernel_h) * (W // kernel_w) * B
        # C=3, kernel_h=16, kernel_w=16
        
        # 将 x_tidle_patch 和 target_patch 输入到 vgg 网络中
        x_tidle_feat  = self.vgg(x_tidle_patch)
        target_feat = self.vgg(target_patch)
        
        out["charbonnier"] = self.charbonnier(x_hat, x_tidle)
        out["lpips_loss"]  = self.lpips.forward(x_tidle_patch, target_patch, normalize=True)
        
        x_tidle_feat  = [feat for feat in  x_tidle_feat]
        target_feat = [feat for feat in  target_feat]
        style_loss = 0
        for i in range(4):
            style_loss += torch.mean(self.style(x_tidle_feat[i], target_feat[i]))
        out["style_loss"]  = style_loss
        # out["face_loss"] =  self.mse(mask * target, mask * x_hat)

        '''
        # rd lambda
        "lambda_char": 2e-6,
        "lambda_lpips": 1,
        "lambda_style": 1e2,
        "lambda_face": 0,
        "lambda_gan": 1,
        "lambda_rate": 0.3,
        '''
        out["loss"] = self.lmbda * 255**2 * out["charbonnier"] + out["bpp_loss"] + out["lpips_loss"] + 100 * out["style_loss"]
        return out
