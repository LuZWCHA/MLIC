import torch
import torch.nn as nn
from . import pretrained_networks as pn
from . import perceptual_loss as pl


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    scale_factor_H, scale_factor_W = 1. * out_HW[0] / in_H, 1. * out_HW[1] / in_W

    return nn.Upsample(scale_factor=(scale_factor_H, scale_factor_W), mode='bilinear', align_corners=False)(in_tens)

# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()

        if(self.pnet_type in ['vgg', 'vgg16']):
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif(self.pnet_type == 'alex'):
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif(self.pnet_type == 'squeeze'):
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if(self.pnet_type == 'squeeze'):  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(
            in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = pl.normalize_tensor(
                outs0[kk]), pl.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) **2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk].model(diffs[kk]), out_HW=in0.shape[2:])
                       for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(
                    diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True),
                                out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(
                    dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]
        # 获取每一层的中间结果
        if(retPerLayer):
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor(
            [-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor(
            [.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
        
        

class PerceptualLoss(torch.nn.Module):
    # VGG using our perceptually-learned weights (LPIPS metric)
    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0], version='0.1'):
        # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG as a perceptual loss
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        super(PerceptualLoss, self).__init__()
        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = []
        self.model = model
        self.net = net
        model_path = None
        self.is_train = False
        if(self.model == 'net-lin'):  # pretrained net + linear layer
            self.net = PNetLin(pnet_rand=False, pnet_tune=False, pnet_type=net,
                                        use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = {}
            if not use_gpu:
                kw['map_location'] = 'cpu'
            if(model_path is None):
                import inspect
                import os
                model_path = os.path.abspath(
                    os.path.join('.', 'weights/%s.pth' % (net)))

            if(not self.is_train):
                print('Loading model from: %s' % model_path)
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)

        print('...[%s] initialized' % self.model)
        print('...Done')
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if(use_gpu):
            self.gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
            # self.net.to(self.gpu_ids[0])
            self.net.to(device)
            self.net = CustomDataParallel(self.net, device_ids=self.gpu_ids)
        if(self.is_train):
            self.rankLoss = self.rankLoss.to(self.gpu_ids[0])  # just put this on GPU0

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        return self.net.forward(target, pred, retPerLayer=False)