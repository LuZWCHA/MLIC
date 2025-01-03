import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import math
import torch.nn as nn

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

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step
):
    amp = True
    scaler = GradScaler(enabled=amp)
    model.train()
    device = next(model.parameters()).device
    gauss_k = get_gaussian_kernel().to(device)
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        # SR
        # blur_d = gauss_k(d)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        with autocast(enabled=amp):
            out_net = model(d)

            out_criterion = criterion(out_net, d)

        scaler.scale(out_criterion["loss"]).backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        # optimizer.step()
        scaler.step(optimizer)

        aux_loss = model.aux_loss()
        scaler.scale(aux_loss).backward()
        # aux_optimizer.step()
        scaler.step(aux_optimizer)
        scaler.update()

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

        if i % 50 == 0:
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

class MinNormSolver:
    """
    求解最小范数问题的工具类。
    """
    @staticmethod
    def find_min_norm_element(grads, max_iter=20, tol=1e-4):
        """
        找到最小范数解。
        :param grads: 梯度列表，每个元素是一个 torch.Tensor。
        :param max_iter: 最大迭代次数。
        :param tol: 收敛容忍度。
        :return: 权重 alpha。
        """
        num_tasks = len(grads)
        alpha = torch.ones(num_tasks) / num_tasks  # 初始化为均匀权重
        grads_tensor = torch.stack(grads)  # 将梯度列表堆叠为一个张量以便向量化操作
        
        for iteration in range(max_iter):
            # 计算当前梯度（向量化操作）
            grad = torch.matmul(alpha, grads_tensor)
            
            # 找到最速下降方向（向量化操作）
            norms = torch.norm(grads_tensor - grad, dim=1)
            min_norm_dir = torch.argmin(norms)
            
            # 计算步长
            step_size = 2.0 / (iteration + 2.0)  # 步长逐渐减小
            
            # 更新权重
            new_alpha = torch.zeros(num_tasks)
            new_alpha[min_norm_dir] = 1.0
            alpha = (1 - step_size) * alpha + step_size * new_alpha
            
            # 检查收敛
            if torch.norm(new_alpha - alpha) < tol:
                break
        
        return alpha

from models.mlicpp_vbr import MLICPlusPlusVbr
def train_one_epoch_mmo(
    model: MLICPlusPlusVbr, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step
):
    amp = False
    scaler = GradScaler(enabled=amp)
    model.train()
    device = next(model.parameters()).device
    gauss_k = get_gaussian_kernel().to(device)
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        # SR
        # blur_d = gauss_k(d)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # 计算每个任务的损失和梯度
        losses = []
        grads_theta = []
        # grads_gain = []
        # with autocast(enabled=amp):
        for s, lmbda in enumerate(model.lmbda):
            out_net = model.forward(d, stage=2, s=s)
            criterion.set_lmbda(lmbda)
            out_criterion = criterion(out_net, d)
            loss = out_criterion["loss"]
            losses.append(loss)
            grad_theta = grad(loss, model.parameters(), retain_graph=True)
            grads_theta.append(grad_theta)

            # model.Gain grad is always zero
            # grad_gain = grad(loss, [model.Gain[i]], retain_graph=True)
            # grads_gain.append(grad_gain)

        # 计算权重 alpha
        alpha = MinNormSolver.find_min_norm_element(grads_theta)

        # 更新共享参数 theta
        for param, grad_theta in zip(model.parameters(), zip(*grads_theta)):
            param.grad = sum(a * g for a, g in zip(alpha, grad_theta))

        # scaler.scale(out_criterion["loss"]).backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # scaler.step(optimizer)

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        # scaler.step(aux_optimizer)
        # scaler.update()

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

        if i % 50 == 0:
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
