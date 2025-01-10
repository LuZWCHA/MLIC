import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import math
import torch.nn as nn
from playground.ddp import *

def get_gaussian_kernel(kernel_size=3, sigma=1, channels=3):
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

from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from compressai.models import CompressionModel

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, rank=1, amp=True
):
    
    scaler = GradScaler(enabled=amp)
    model.train()
    device = next(model.parameters()).device
    gauss_k = get_gaussian_kernel().to(device)
    for i, d in enumerate(train_dataloader):
        if isinstance(d, torch.Tensor):
            img = d.to(device)
        elif isinstance(d, dict):
            img = d["image"].to(device)
            img_paths = d["path"]
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        with autocast(enabled=amp):
            # SR
            # blur_d = gauss_k(d)
            out_net = model(img)

            out_criterion = criterion(out_net, img)

        scaler.scale(out_criterion["loss"]).backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        # optimizer.step()
        scaler.step(optimizer)

        aux_loss = model.aux_loss() if isinstance(model, CompressionModel) else model.module.aux_loss()
        scaler.scale(aux_loss).backward()
        # aux_optimizer.step()
        scaler.step(aux_optimizer)
        scaler.update()

        current_step += 1
        
        if rank <= 0:
            if current_step % 20 == 0:
                tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
                tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
                tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
                tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
                if out_criterion["mse_loss"] is not None:
                    tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
                if out_criterion["ms_ssim_loss"] is not None:
                    tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)

            if i % 20 == 0:
                if out_criterion["ms_ssim_loss"] is None:
                    logger_train.info(
                        f"Train epoch {epoch}: ["
                        f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                        f" ({100. * i / len(train_dataloader):.0f}%)] "
                        f'Loss: {out_criterion["loss"].item():.4f} | '
                        f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                        f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                        f"Aux loss: {aux_loss.item():.2f}"
                    )
                else:
                    logger_train.info(
                        f"Train epoch {epoch}: ["
                        f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                        f" ({100. * i / len(train_dataloader):.0f}%)] "
                        f'Loss: {out_criterion["loss"].item():.4f} | '
                        f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                        f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                        f"Aux loss: {aux_loss.item():.2f}"
                    )

    return current_step


def train_one_epoch_dual(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, rank=1, amp=True
):
    
    scaler = GradScaler(enabled=amp)
    model.train()
    device = next(model.parameters()).device
    gauss_k = get_gaussian_kernel().to(device)
    
    for i, d in enumerate(train_dataloader):
        if isinstance(d, torch.Tensor):
            img = d.to(device)
        elif isinstance(d, dict):
            img = d["image"].to(device)
            img_paths = d["path"]
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        with autocast(enabled=amp):
            # SR
            # blur_d = gauss_k(d)
            # with torch.no_grad():
            first_out_net = model(img)
            first_out_criterion = criterion(first_out_net, img)
            first_aux_loss = model.aux_loss() if isinstance(model, CompressionModel) else model.module.aux_loss()
            
            compressed_img = first_out_criterion["x_hat"].detach()
            out_net = model(compressed_img)
            old_lmbda = criterion.lmbda
            new_lmbda = criterion.lmbda * 0.5
            criterion.set_lmbda(new_lmbda)
            out_criterion = criterion(out_net, img)
            criterion.set_lmbda(old_lmbda)
            
        scaler.scale(first_out_criterion["loss"] + out_criterion["loss"]).backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        # optimizer.step()
        scaler.step(optimizer)

        aux_loss = model.aux_loss() if isinstance(model, CompressionModel) else model.module.aux_loss()
        scaler.scale(aux_loss + first_aux_loss).backward()
        # aux_optimizer.step()
        scaler.step(aux_optimizer)
        scaler.update()

        current_step += 1
        
        if rank <= 0:
            if current_step % 20 == 0:
                tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
                tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
                tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
                tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
                if out_criterion["mse_loss"] is not None:
                    tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
                if out_criterion["ms_ssim_loss"] is not None:
                    tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)

            if i % 20 == 0:
                if out_criterion["ms_ssim_loss"] is None:
                    logger_train.info(
                        f"Train epoch {epoch}: ["
                        f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                        f" ({100. * i / len(train_dataloader):.0f}%)] "
                        f'Loss: {out_criterion["loss"].item():.4f} | '
                        f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                        f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                        f"Aux loss: {aux_loss.item():.2f}"
                    )
                else:
                    logger_train.info(
                        f"Train epoch {epoch}: ["
                        f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                        f" ({100. * i / len(train_dataloader):.0f}%)] "
                        f'Loss: {out_criterion["loss"].item():.4f} | '
                        f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                        f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                        f"Aux loss: {aux_loss.item():.2f}"
                    )

    return current_step


import torch.optim as optim
from torch.autograd import grad

def MinNormSolver(gradients, max_iter=10):
    N = len(gradients)
    alpha = torch.ones(N, device=gradients[0].device) / N
    for _ in range(max_iter):
        combined = torch.sum(alpha[:, None] * gradients, dim=0)
        dot_products = torch.sum(combined * gradients, dim=1)
        min_idx = torch.argmin(dot_products)
        direction = torch.zeros_like(alpha)
        direction[min_idx] = 1.0 - alpha[min_idx]
        alpha += 0.5 * direction  # Update rule, can be adjusted
        alpha = torch.clamp(alpha, min=0.0)
        alpha /= alpha.sum()
    return alpha


from models.mlicpp_vbr import MLICPlusPlusVbr
def train_one_epoch_mmo(
    model: MLICPlusPlusVbr, criterion, train_dataloader, optimizer_theta, optimizer_phi, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, rank=1, world_size=1, amp=False
):
    scaler = GradScaler(enabled=amp)
    model.train()
    device = next(model.parameters()).device
    gauss_k = get_gaussian_kernel().to(device)
    for i, d in enumerate(train_dataloader):
        if isinstance(d, torch.Tensor):
            img = d.to(device)
        elif isinstance(d, dict):
            img = d["image"].to(device)
            img_paths = d["path"]
        
        # SR
        # blur_d = gauss_k(d)

        aux_optimizer.zero_grad()

        # 共享参数
        optimizer_theta.zero_grad()
        
        gradients_theta = []
        N = model.levels
        
        shared_parameters, _ = model.mmo_parameters()
        gradients_shared = []
        # 计算每个任务的损失和梯度
        for i in range(N):
            optimizer_phi[i].zero_grad()
            out_net = model(img, s=i)
            out_criterion = criterion(out_net, img)
            
            loss = out_criterion["loss"]
            scaler.scale(loss).backward(retain_graph=True)
            
            # Collect gradients for θ
            gradients_theta = [sp.grad.detach().clone() for sp in shared_parameters]
            
            # Update φi
            scaler.step(optimizer_phi[i])
            gradients_shared.append(gradients_theta)
        
        # Flatten gradients for easier handling
        flattened_gradients = [torch.cat([g.flatten() for g in grad_list]) for grad_list in gradients_shared]
        
        # Use MinNormSolver to find optimal alpha coefficients
        alpha = MinNormSolver(flattened_gradients)

        # Combine gradients
        combined_gradient = sum(alpha[i] * flattened_gradients[i] for i in range(model.levels))
        
        # Unflatten combined gradient and set to shared parameters
        idx = 0
        for param in shared_parameters:
            num_params = param.numel()
            param.grad = combined_gradient[idx:idx+num_params].view(param.shape)
            idx += num_params
        
        # Synchronize combined gradients across all processes
        for param in shared_parameters:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size  # Average gradients
        
        # Step the optimizer for shared parameters
        scaler.step(optimizer_theta)

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        aux_loss = model.aux_loss() if isinstance(model, CompressionModel) else model.module.aux_loss()
        scaler.scale(aux_loss).backward()
        # aux_optimizer.step()
        scaler.step(aux_optimizer)
        scaler.update()

        current_step += 1
        if rank <= 0:
            if current_step % 20 == 0:
                tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
                tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
                tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer_theta.param_groups[0]['lr'], current_step)
                tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
                if out_criterion["mse_loss"] is not None:
                    tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
                if out_criterion["ms_ssim_loss"] is not None:
                    tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)

            if i % 20 == 0:
                if out_criterion["ms_ssim_loss"] is None:
                    logger_train.info(
                        f"Train epoch {epoch}: ["
                        f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                        f" ({100. * i / len(train_dataloader):.0f}%)] "
                        f'Loss: {out_criterion["loss"].item():.4f} | '
                        f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                        f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                        f"Aux loss: {aux_loss.item():.2f}"
                    )
                else:
                    logger_train.info(
                        f"Train epoch {epoch}: ["
                        f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                        f" ({100. * i / len(train_dataloader):.0f}%)] "
                        f'Loss: {out_criterion["loss"].item():.4f} | '
                        f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                        f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                        f"Aux loss: {aux_loss.item():.2f}"
                    )

    return current_step


def warmup_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, lr_scheduler
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        if epoch < 1:
            lr_scheduler.step()
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_step += 1
        if current_step % 20 == 0:
            tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb_logger.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_step)
            tb_logger.add_scalar('{}'.format('[train]: aux_loss'), aux_loss.item(), current_step)
            if out_criterion["mse_loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
            if out_criterion["ms_ssim_loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(), current_step)

        if i % 20 == 0:
            if out_criterion["ms_ssim_loss"] is None:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Lr: {optimizer.param_groups[0]["lr"]:.6f} | '
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )
            else:
                logger_train.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] "
                    f'Lr: {optimizer.param_groups[0]["lr"]:.6f} | '
                    f'Loss: {out_criterion["loss"].item():.4f} | '
                    f'MS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.4f} | '
                    f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                    f"Aux loss: {aux_loss.item():.2f}"
                )

    return current_step
