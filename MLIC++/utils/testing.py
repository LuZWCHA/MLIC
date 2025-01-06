import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.metrics import compute_metrics
from utils.utils import *


def test_one_epoch_vbr(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    for level in range(len(model.lmbda)):
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()
        psnr = AverageMeter()
        lpips_loss = AverageMeter()
        ms_ssim = AverageMeter()
        avg_dists = AverageMeter()

        with torch.no_grad():
            for i, d in enumerate(test_dataloader):
                d = d.to(device)
                B, C, H, W = d.shape

                pad_h = 0
                pad_w = 0
                if H % 64 != 0:
                    pad_h = 64 * (H // 64 + 1) - H
                if W % 64 != 0:
                    pad_w = 64 * (W // 64 + 1) - W

                img_pad = F.pad(d, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
                out_net = model(img_pad, s=level)
                out_net['x_hat'] = out_net['x_hat'][:, :, :H, :W]
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                if out_criterion["mse_loss"] is not None:
                    mse_loss.update(out_criterion["mse_loss"])
                if out_criterion["ms_ssim_loss"] is not None:
                    ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

                rec = torch2img(out_net['x_hat'])
                img = torch2img(d)
                p, m, lpips_m, dists = compute_metrics(rec, img)
                psnr.update(p)
                ms_ssim.update(m)
                lpips_loss.update(lpips_m)
                avg_dists.update(dists)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                rec.save(os.path.join(save_dir, '%03d_rec_lv%03d.png' % i % level))
                img.save(os.path.join(save_dir, '%03d_gt_lv%03d.png' % i % level))

        tb_logger.add_scalar(f'[val]: loss', loss.avg, epoch + 1)
        tb_logger.add_scalar(f'[val]: bpp_loss', bpp_loss.avg, epoch + 1)
        tb_logger.add_scalar(f'[val]: psnr', psnr.avg, epoch + 1)
        tb_logger.add_scalar(f'[val]: ms-ssim', ms_ssim.avg, epoch + 1)
        tb_logger.add_scalar(f'[val]: lpips', lpips_loss.avg, epoch + 1)
        tb_logger.add_scalar(f'[val]: dists', avg_dists.avg, epoch + 1)

        if out_criterion["mse_loss"] is not None:
            logger_val.info(
                f"Test epoch {epoch}: Average losses at level {level}: "
                f"Loss: {loss.avg:.4f} | "
                f"MSE loss: {mse_loss.avg:.6f} | "
                f"LPISP loss: {lpips_loss.avg:.6f} | "
                f"DISTS loss: {avg_dists.avg:.6f} | "
                f"Bpp loss: {bpp_loss.avg:.4f} | "
                f"Aux loss: {aux_loss.avg:.2f} | "
                f"PSNR: {psnr.avg:.6f} | "
                f"MS-SSIM: {ms_ssim.avg:.6f}"
            )
            tb_logger.add_scalar(f'[val]: mse_loss', mse_loss.avg, epoch + 1)
        if out_criterion["ms_ssim_loss"] is not None:
            logger_val.info(
                f"Test epoch {epoch}: Average losses at level {level}: "
                f"Loss: {loss.avg:.4f} | "
                f"MS-SSIM loss: {ms_ssim_loss.avg:.6f} | "
                f"LPISP loss: {lpips_loss.avg:.6f} | "
                f"DISTS loss: {avg_dists.avg:.6f} | "
                f"Bpp loss: {bpp_loss.avg:.4f} | "
                f"Aux loss: {aux_loss.avg:.2f} | "
                f"PSNR: {psnr.avg:.6f} | "
                f"MS-SSIM: {ms_ssim.avg:.6f}"
            )
            tb_logger.add_scalar(f'[val]: ms_ssim_loss', ms_ssim_loss.avg, epoch + 1)

    return loss.avg

def test_one_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    lpips_loss = AverageMeter()
    ms_ssim = AverageMeter()
    avg_dists = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            B, C, H, W = d.shape

            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W

            img_pad = F.pad(d, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
            out_net = model(img_pad)
            out_net['x_hat'] = out_net['x_hat'][:, :, :H, :W]
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

            rec = torch2img(out_net['x_hat'])
            img = torch2img(d)
            p, m, lpips_m, dists = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)
            lpips_loss.update(lpips_m)
            avg_dists.update(dists)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar(f'[val]: loss', loss.avg, epoch + 1)
    tb_logger.add_scalar(f'[val]: bpp_loss', bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar(f'[val]: psnr', psnr.avg, epoch + 1)
    tb_logger.add_scalar(f'[val]: ms-ssim', ms_ssim.avg, epoch + 1)
    tb_logger.add_scalar(f'[val]: lpips', lpips_loss.avg, epoch + 1)
    tb_logger.add_scalar(f'[val]: dists', avg_dists.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.6f} | "
            f"LPISP loss: {lpips_loss.avg:.6f} | "
            f"DISTS loss: {avg_dists.avg:.6f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar(f'[val]: mse_loss', mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.6f} | "
            f"LPISP loss: {lpips_loss.avg:.6f} | "
            f"DISTS loss: {avg_dists.avg:.6f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar(f'[val]: ms_ssim_loss', ms_ssim_loss.avg, epoch + 1)

    return loss.avg

def compress_one_image(model, x, stream_path, H, W, img_name):
    with torch.no_grad():
        out = model.compress(x)

    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    return bpp, out["cost_time"]


def decompress_one_image(model, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    with torch.no_grad():
        out = model.decompress(strings, shape)

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    cost_time = out["cost_time"]
    return x_hat, cost_time

def compress_one_image_vbr(model, x, stream_path, H, W, img_name, level=0, force=False):
    from models.mlicpp_vbr import MLICPlusPlusVbr
    model: MLICPlusPlusVbr
    L = level
    with torch.no_grad():
        out = model.compress(x, stage=2, s=int(L), inputscale=0 if not force else level)
    
    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W, L))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    return bpp, out["cost_time"]


def decompress_one_image_vbr(model, stream_path, img_name, force=False):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size_and_level = read_uints(f, 3)
        strings, shape = read_body(f)

    with torch.no_grad():
        out = model.decompress(strings, shape, s=int(original_size_and_level[2]), stage=2, inputscale=0 if not force else original_size_and_level[2])

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size_and_level[0], 0 : original_size_and_level[1]]
    cost_time = out["cost_time"]
    return x_hat, cost_time

def get_gaussian_kernel(kernel_size=3, sigma=0.5, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

import torch
from torch import Tensor, nn
from torchvision.io import decode_jpeg, encode_jpeg


import sys
import torch
from deepspeed.profiling.flops_profiler import get_model_profile

# from lvc_dec_complexity import LVC_exp_spy_res

torch.backends.cudnn.deterministic = True


def get_macs(net):
    class MyModel(nn.Module):

        def __init__(self, net, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.net = net

        def forward(self, x):
            self.net.net_decoder_forward(x)


    my_wrapper = MyModel(net).to("cuda:0")
    my_wrapper.eval()

    #frame = torch.ones([1,3,640,640], dtype=torch.float32, device=device)
    width, height = 1920, 1088
    flops, macs, params = get_model_profile(my_wrapper, (1, 3, width, height))

    print("params: ", params)
    print("flops: ", flops)
    print("macs: ", macs)


def apply_jpeg(x: Tensor, quality: int) -> Tensor:
    return decode_jpeg(encode_jpeg(x, quality))


def test_model(test_dataloader, net, logger_test, save_dir, epoch):
    net.eval()
    device = next(net.parameters()).device
    gaussian_kernel = get_gaussian_kernel().to(device)
    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_lpips_loss = AverageMeter()
    avg_dists = AverageMeter()
    avg_bpp = AverageMeter()
    avg_enc_time = AverageMeter()
    avg_dec_time = AverageMeter()
    cons = 0.100
    
    get_macs(net)

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            bpp = 1e6
            B, C, ORI_H, ORI_W = img.shape
            ori_img = img.to(device)
            while True:
                img = img.to(device)
                B, C, H, W = img.shape
                # print(H, W)
                pad_h = 0
                pad_w = 0
                if H % 64 != 0:
                    pad_h = 64 * (H // 64 + 1) - H
                if W % 64 != 0:
                    pad_w = 64 * (W // 64 + 1) - W
                # print(H, W)
                img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
                
                # warmup GPU
                if i == 0:
                    bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
                # avoid resolution leakage
                net.update_resolutions(16, 16)
                bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))

                # avoid resolution leakage
                net.update_resolutions(16, 16)
                x_hat, dec_time = decompress_one_image(model=net, stream_path=save_dir, img_name=str(i))
                # print(bpp)
                if bpp > cons:
                    img = gaussian_kernel(img)
                else:
                    break
                    # print(bpp)

            rec = torch2img(x_hat)
            img = torch2img(ori_img)
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            p, m, lp, dists = compute_metrics(rec, img)
            avg_psnr.update(p)
            avg_ms_ssim.update(m)
            avg_dists.update(dists)
            avg_bpp.update(bpp)
            avg_lpips_loss.update(lp)
            avg_enc_time.update(enc_time)
            avg_dec_time.update(dec_time)
            logger_test.info(
                f"Image[{i}] | "
                f"Bpp loss: {bpp:.2f} | "
                f"PSNR: {p:.4f} | "
                f"LPIPS: {lp:.4f} | "
                f"DISTS: {dists:.4f} | "
                f"MS-SSIM: {m:.4f} | "
                f"Encoding Latency: {enc_time:.4f} | "
                f"Decoding Latency: {dec_time:.4f}"
            )
    logger_test.info(
        f"Epoch:[{epoch}] | "
        f"Avg Bpp: {avg_bpp.avg:.4f} | "
        f"Avg PSNR: {avg_psnr.avg:.4f} | "
        f"Avg LPIPS: {avg_lpips_loss.avg:.4f} | "
        f"Avg DISTS: {avg_dists.avg:.4f} | "
        f"Avg MS-SSIM: {avg_ms_ssim.avg:.4f} | "
        f"Avg Encoding Latency:: {avg_enc_time.avg:.4f} | "
        f"Avg Decoding Latency:: {avg_dec_time.avg:.4f}"
    )



def test_model_vbr(test_dataloader, net, logger_test, save_dir, epoch, custom_scales=None):
    net.eval()
    device = next(net.parameters()).device
    gaussian_kernel = get_gaussian_kernel().to(device)

    force = False
    levels = range(len(net.lmbda))
    if custom_scales is not None:
        levels = custom_scales
        force = True

    level_metrics = dict()

    for l in range(len(levels)):
        level_metrics[l] = dict(
            avg_psnr = AverageMeter(),
            avg_ms_ssim = AverageMeter(),
            avg_lpips_loss = AverageMeter(),
            avg_dists = AverageMeter(),
            avg_bpp = AverageMeter(),
            avg_enc_time = AverageMeter(),
            avg_dec_time = AverageMeter()
        )
    
    cons = 0.100
    
    get_macs(net)

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            bpp = 1e6
            B, C, ORI_H, ORI_W = img.shape
            ori_img = img.to(device)
            while True:
                img = img.to(device)
                B, C, H, W = img.shape
                # print(H, W)
                pad_h = 0
                pad_w = 0
                if H % 64 != 0:
                    pad_h = 64 * (H // 64 + 1) - H
                if W % 64 != 0:
                    pad_w = 64 * (W // 64 + 1) - W
                # print(H, W)
                img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
                
                # warmup GPU
                if i == 0:
                    bpp, enc_time = compress_one_image_vbr(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i), level=0)
                
                for i, level in enumerate(levels):

                    if force:
                        scale = level
                    else:
                        scale = net.Gains[int(level)].item()

                    # avoid resolution leakage
                    net.update_resolutions(16, 16)
                    bpp, enc_time = compress_one_image_vbr(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=f"{i:03}_lv{level:02}", level=level, force=force)

                    # avoid resolution leakage
                    net.update_resolutions(16, 16)
                    x_hat, dec_time = decompress_one_image_vbr(model=net, stream_path=save_dir, img_name=f"{i:03}_lv{level:02}", force=force)

                    rec = torch2img(x_hat)
                    img = torch2img(ori_img)
                    img.save(os.path.join(save_dir, '%03d_gt.png' % i))
                    rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
                    p, m, lp, dists = compute_metrics(rec, img)
                    level_metrics[i]["avg_psnr"].update(p)
                    level_metrics[i]["avg_ms_ssim"].update(m)
                    level_metrics[i]["avg_dists"].update(dists)
                    level_metrics[i]["avg_bpp"].update(bpp)
                    level_metrics[i]["avg_lpips_loss"].update(lp)
                    level_metrics[i]["avg_enc_time"].update(enc_time)
                    level_metrics[i]["avg_dec_time"].update(dec_time)
                    logger_test.info(
                        f"Image[{i}] | "
                        f"Scale[{scale}] | "
                        f"Bpp: {bpp:.2f} | "
                        f"PSNR: {p:.4f} | "
                        f"LPIPS: {lp:.4f} | "
                        f"DISTS: {dists:.4f} | "
                        f"MS-SSIM: {m:.4f} | "
                        f"Encoding Latency: {enc_time:.4f} | "
                        f"Decoding Latency: {dec_time:.4f}"
                    )
                    
                if bpp > cons:
                    img = gaussian_kernel(img)
                else:
                    break
                    # print(bpp)

            # rec = torch2img(x_hat)
            # img = torch2img(ori_img)
            # img.save(os.path.join(save_dir, '%03d_gt.png' % i))
            # rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            # p, m, lp, dists = compute_metrics(rec, img)
            # avg_psnr.update(p)
            # avg_ms_ssim.update(m)
            # avg_dists.update(dists)
            # avg_bpp.update(bpp)
            # avg_lpips_loss.update(lp)
            # avg_enc_time.update(enc_time)
            # avg_dec_time.update(dec_time)
            # logger_test.info(
            #     f"Image[{i}] | "
            #     f"Bpp loss: {bpp:.2f} | "
            #     f"PSNR: {p:.4f} | "
            #     f"LPIPS: {lp:.4f} | "
            #     f"DISTS: {dists:.4f} | "
            #     f"MS-SSIM: {m:.4f} | "
            #     f"Encoding Latency: {enc_time:.4f} | "
            #     f"Decoding Latency: {dec_time:.4f}"
            # )

    for i in range(len(levels)):
        m = level_metrics[i]
        avg_bpp = m["avg_bpp"]
        avg_psnr = m["avg_psnr"]
        avg_lpips_loss = m["avg_lpips_loss"]
        avg_dists = m["avg_dists"]
        avg_ms_ssim = m["avg_ms_ssim"]
        avg_enc_time = m["avg_enc_time"]
        avg_dec_time = m["avg_dec_time"]
        logger_test.info(f"-------------------- Level: {levels[i]} --------------------")
        logger_test.info(
            f"Epoch:[{epoch}] | "
            f"Avg Bpp: {avg_bpp.avg:.4f} | "
            f"Avg PSNR: {avg_psnr.avg:.4f} | "
            f"Avg LPIPS: {avg_lpips_loss.avg:.4f} | "
            f"Avg DISTS: {avg_dists.avg:.4f} | "
            f"Avg MS-SSIM: {avg_ms_ssim.avg:.4f} | "
            f"Avg Encoding Latency:: {avg_enc_time.avg:.4f} | "
            f"Avg Decoding Latency:: {avg_dec_time.avg:.4f}"
        )
