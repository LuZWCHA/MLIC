import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from utils.utils import AverageMeter, disable_logging_prefix, get_system_info_str, pretty_print_dict
from utils.optimizers import configure_optimizers, configure_optimizers_mmo
from loss.rd_loss import RateDistortionLoss
from models import get_model
from playground.dataset import ImageFolder2, RandomResize
from utils.metrics import compute_metrics
from utils.utils import torch2img
import torch.nn.functional as F
from PIL import Image

# 提高PIL的大图像像素限制
Image.MAX_IMAGE_PIXELS = 200_000_000  # 设置为 2 亿像素

from base_trainer import BaseTrainer

class Trainer(BaseTrainer):

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.eval_first = True
        if self.is_main_process():
            self.logger_train.info(get_system_info_str())

    def _load_checkpoint(self):
        """加载模型检查点，支持恢复训练或仅加载模型权重"""
        checkpoint = torch.load(self.args.checkpoint)

        # 处理模型权重（移除 "module." 前缀以兼容单机和分布式训练）
        new_sd = {}
        for k, v in checkpoint['state_dict'].items():
            while k.startswith("module."):
                k = k[7:]  # 移除 "module." 前缀
            new_sd[k] = v
            
        if self.ddp_enable:
            self.model.module.load_state_dict(new_sd)
        else:
            self.model.load_state_dict(new_sd)

        # 如果是从检查点恢复训练
        if self.args.resume:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])

            # 重新初始化学习率调度器
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[450, 550], gamma=0.1
            )

            # 恢复训练状态
            self.start_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['loss']
            self.current_step = self.start_epoch * math.ceil(len(self.train_loader.dataset) / self.args.batch_size)

        # 释放检查点以节省内存
        checkpoint = None

    def _setup_train_data(self):
        """初始化训练数据集"""
        train_transforms = transforms.Compose([
            transforms.AutoAugment(),
            RandomResize(),
            transforms.RandomCrop(self.args.patch_size, pad_if_needed=True),
            transforms.ToTensor()
        ])
        self.train_dataset = ImageFolder2(self.args.dataset, split="train", transform=train_transforms)

        self.train_sampler = DistributedSampler(self.train_dataset) if self.ddp_enable else None
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=(self.train_sampler is None),
            pin_memory=(self.device == "cuda"),
            sampler=self.train_sampler
        )
        return train_loader
        

    def _setup_test_data(self):
        """初始化测试数据集"""
        test_transforms = transforms.Compose([transforms.ToTensor()])
        self.test_dataset = ImageFolder2(self.args.dataset, split="test", transform=test_transforms)

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=(self.device == "cuda")
        )
        return test_loader
        

    def _setup_model(self):
        model = get_model(self.args.model_name).to(self.device)
        if self.args.pretrained:
            checkpoint = torch.load(self.args.pretrained)
            new_sd = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(new_sd)

        if self.ddp_enable:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])
            
        return model

    def _setup_optimizers(self):
        self.optimizer, self.aux_optimizer = configure_optimizers(self.model, self.args)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 100], gamma=0.1)

    def _setup_criterion(self):
        self.criterion = RateDistortionLoss(lmbda=self.args.lmbda, metrics=self.args.metrics)

    def _init_train_metrics(self):
        """初始化训练指标"""
        self.train_loss = AverageMeter()
        self.train_bpp_loss = AverageMeter()
        self.train_sm_loss = AverageMeter()
        self.train_aux_loss = AverageMeter()
        
    def _init_val_metrics(self):
        """初始化验证指标"""
        self.val_loss = AverageMeter()
        self.val_bpp_loss = AverageMeter()
        self.val_mse_loss = AverageMeter()
        self.val_ms_ssim = AverageMeter()
        self.val_aux_loss = AverageMeter()
        self.val_psnr = AverageMeter()
        self.val_lpips = AverageMeter()
        self.val_dists = AverageMeter()

    def _update_train_metrics(self, step_metrics):
        """更新训练指标"""
        self.train_loss.update(step_metrics["loss"])
        if "ms_ssim_loss" in step_metrics and step_metrics["ms_ssim_loss"] is not None:
            self.train_sm_loss.update(step_metrics["ms_ssim_loss"])
        if "mse_loss" in step_metrics and step_metrics["mse_loss"] is not None:
            self.train_sm_loss.update(step_metrics["mse_loss"])
        self.train_bpp_loss.update(step_metrics["bpp_loss"])
        self.train_aux_loss.update(step_metrics["aux_loss"])
        
    def _update_val_metrics(self, step_metrics):
        """更新验证指标"""
        self.val_loss.update(step_metrics["loss"])
        self.val_bpp_loss.update(step_metrics["bpp_loss"])
        self.val_psnr.update(step_metrics["psnr"])
        self.val_ms_ssim.update(step_metrics["ms_ssim"])
        self.val_mse_loss.update(step_metrics["similar_loss"])
        self.val_aux_loss.update(step_metrics["aux_loss"])
        self.val_dists.update(step_metrics["dists"])
        self.val_lpips.update(step_metrics["lpips"])

    def _log_train(self, epoch, batch_idx):
        """记录训练日志"""
        log_message = (
            f"Train epoch {epoch}: [{batch_idx * self.args.batch_size}/{len(self.train_loader.dataset)} "
            f"({100. * batch_idx / len(self.train_loader):.0f}%)] "
            f"Loss: {self.train_loss.avg:.4f} | "
            f"Similarity loss: {self.train_sm_loss.avg:.4f} | "
            f"Bpp loss: {self.train_bpp_loss.avg:.2f} | "
            f"Aux loss: {self.train_aux_loss.avg:.2f}"
        )
        self.logger_train.info(log_message)
        
        tb_train_loss_tag = 'Train/Loss'
        tb_train_sm_loss_tag = 'Train/Similarity'
        tb_train_bpp_loss_tag = 'Train/Bpp'
        tb_train_aux_loss_tag = 'Train/AuxLoss'
        
        step = epoch * len(self.train_loader) + batch_idx
        self.tb_logger.add_scalar(tb_train_loss_tag, self.train_loss.avg, step)
        self.tb_logger.add_scalar(tb_train_sm_loss_tag, self.train_sm_loss.avg, step)
        self.tb_logger.add_scalar(tb_train_bpp_loss_tag, self.train_bpp_loss.avg, step)
        self.tb_logger.add_scalar(tb_train_aux_loss_tag, self.train_aux_loss.avg, step)

    def _log_val(self, epoch):
        """记录验证日志"""
        log_message = (
            f"Test epoch {epoch}: Loss: {self.val_loss.avg:.4f} | Bpp: {self.val_bpp_loss.avg:.4f} | "
            f"PSNR: {self.val_psnr.avg:.4f} | MS-SSIM: {self.val_ms_ssim.avg:.4f} | DIST: {self.val_dists.avg:.4f} | "
            f"LPIPS: {self.val_lpips.avg:.4f} | "
        )
        self.logger_val.info(log_message)
        # 提取 TensorBoard 日志标签
        
        tb_val_loss_tag = 'Val/Loss'
        tb_val_bpp_tag = 'Val/Bpp'
        tb_val_psnr_tag = 'Val/PSNR'
        tb_val_ms_ssim_tag = 'Val/MS-SSIM'
        tb_val_dist_tag = 'Val/DIST'
        tb_val_lpips_tag = 'Val/LPIPS'
        self.tb_logger.add_scalar(tb_val_loss_tag, self.val_loss.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_bpp_tag, self.val_bpp_loss.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_psnr_tag, self.val_psnr.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_ms_ssim_tag, self.val_ms_ssim.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_dist_tag, self.val_dists.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_lpips_tag, self.val_lpips.avg, epoch + 1)

    def _get_val_metrics(self):
        """获取验证指标, 计算best loss来保存checkpoint"""
        return {
            "loss": self.val_loss.avg,
            "bpp_loss": self.val_bpp_loss.avg,
            "psnr": self.val_psnr.avg,
            "ms_ssim": self.val_ms_ssim.avg,
            "dist": self.val_dists.avg,
            "lpips": self.val_lpips.avg,
        }

    def train_step(self, batch, scaler):
        """训练步骤"""
        images = batch["image"].to(self.device)
        self.optimizer.zero_grad()
        self.aux_optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.args.amp):
            out_net = self.model(images)
            out_criterion = self.criterion(out_net, images)

        scaler.scale(out_criterion["loss"]).backward()
        scaler.step(self.optimizer)

        aux_loss = self.model.module.aux_loss() if self.ddp_enable else self.model.aux_loss()
        scaler.scale(aux_loss).backward()
        
        if self.args.clip_max_norm > 0:
            scaler.unscale_(self.aux_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm)
        
        scaler.step(self.aux_optimizer)
        scaler.update()

        losses = {
            "loss": out_criterion["loss"].item(),
            "bpp_loss": out_criterion["bpp_loss"].item(),
            "aux_loss": aux_loss.item(),
        }
        
        if "mse_loss" in out_criterion and out_criterion["mse_loss"] is not None:
            losses["mse_loss"] = out_criterion["mse_loss"].item()
        if "ms_ssim_loss" in out_criterion and out_criterion["ms_ssim_loss"] is not None:
            losses["ms_ssim_loss"] = out_criterion["ms_ssim_loss"].item()
        
        return losses

    def val_step(self, batch):
        """验证步骤"""
        images = batch["image"].to(self.device)
        paths = batch["path"]
        B, C, H, W = images.shape

        pad_h = 0 if H % 64 == 0 else 64 * (H // 64 + 1) - H
        pad_w = 0 if W % 64 == 0 else 64 * (W // 64 + 1) - W
        images_pad = F.pad(images, (0, pad_w, 0, pad_h), mode='constant', value=0)

        out_net = self.model.module(images_pad) if self.ddp_enable else self.model(images_pad)
        out_net['x_hat'] = out_net['x_hat'][:, :, :H, :W]
        out_criterion = self.criterion(out_net, images)

        aux_loss = self.model.module.aux_loss() if self.ddp_enable else self.model.aux_loss()
        bpp_loss = out_criterion["bpp_loss"]
        loss = out_criterion["loss"]

        rec = torch2img(out_net['x_hat'])
        img = torch2img(images)
        psnr, ms_ssim, lpips_m, dists = compute_metrics(rec, img)
        
        stem = Path(paths[0]).stem
        if not os.path.exists(self.val_images_dir):
            os.makedirs(self.val_images_dir)
        rec.save(os.path.join(self.val_images_dir, '%s_rec.png' % stem))
        img.save(os.path.join(self.val_images_dir, '%s_gt.png' % stem))

        if out_criterion["mse_loss"] is not None:
            smiliar = out_criterion["mse_loss"]
            
        if out_criterion["ms_ssim_loss"] is not None:
            smiliar = out_criterion["ms_ssim_loss"]

        metrics = {
            "path": Path(batch["path"][0]).name,
            "loss": loss.item(),
            "aux_loss": aux_loss.item(),
            "bpp_loss": bpp_loss.item(),
            "similar_loss": smiliar.item(),
            "psnr": psnr,
            "ms_ssim": ms_ssim,
            "lpips": lpips_m,
            "dists": dists,
        }
        
        # if self.is_main_process():
        with disable_logging_prefix(self.logger_val):
            self.logger_val.info(pretty_print_dict([metrics]))
        
        return metrics
        
if __name__ == '__main__':
    from config.args import train_options
    args = train_options()
    trainer = Trainer(args)
    trainer.fit()
    

from models import *
class VBRTrainer(Trainer):
    
    def _setup_optimizers(self):
        self.optimizer, self.aux_optimizer, self.gain_optimizers = \
        configure_optimizers_mmo(self.model, self.args)
        
    
    def __min_norm_solver(self, gradients, max_iter=10):
        """最小范数求解器"""
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
    
    def train_step(self, batch, scaler:torch.cuda.amp.GradScaler):
        """训练步骤"""
        self.model: MLICPlusPlusSDVbr
        images = batch["image"].to(self.device)
        self.optimizer.zero_grad()
        self.aux_optimizer.zero_grad()
        gradients_theta = []
        gradients_shared = []
        N = self.model.levels
        shared_parameters, _ = self.model.mmo_parameters()

        with torch.cuda.amp.autocast(enabled=self.args.amp):
            for i in range(N):
                self.gain_optimizers[i].zero_grad()
                out_net = self.model.forward(images, stage=2, s=i)
                out_criterion = self.criterion(out_net, images)
                loss = out_criterion["loss"]
                # Collect gradients for θ
                gradients_theta = [sp.grad.detach().clone() for sp in shared_parameters]
            
                scaler.scale(loss).backward(retain_graph=True)
                scaler.step(self.gain_optimizers[i])
                gradients_shared.append(gradients_theta)

        # Flatten gradients for easier handling
        flattened_gradients = [torch.cat([g.flatten() for g in grad_list]) for grad_list in gradients_shared]
        
        # Use MinNormSolver to find optimal alpha coefficients
        alpha = self.__min_norm_solver(flattened_gradients)

        # Combine gradients
        combined_gradient = sum(alpha[i] * flattened_gradients[i] for i in range(self.model.levels))
        
        # Unflatten combined gradient and set to shared parameters
        idx = 0
        for param in shared_parameters:
            num_params = param.numel()
            param.grad = combined_gradient[idx:idx+num_params].view(param.shape)
            idx += num_params

        if self.ddp_enable:
            world_size = dist.get_world_size()
        else:
            world_size = 1
        
        # Synchronize combined gradients across all processes
        for param in shared_parameters:
            if self.ddp_enable:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size  # Average gradients
    
        scaler.step(self.optimizer)

        aux_loss = self.model.module.aux_loss() if self.ddp_enable else self.model.aux_loss()
        scaler.scale(aux_loss).backward()
        
        if self.args.clip_max_norm > 0:
            # scaler.unscale_(self.optimizer)
            scaler.unscale_(self.aux_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm)
        
        scaler.step(self.aux_optimizer)
        scaler.update()

        losses = {
            "loss": out_criterion["loss"].item(),
            "bpp_loss": out_criterion["bpp_loss"].item(),
            "aux_loss": aux_loss.item(),
        }
        
        if "mse_loss" in out_criterion and out_criterion["mse_loss"] is not None:
            losses["mse_loss"] = out_criterion["mse_loss"].item()
        if "ms_ssim_loss" in out_criterion and out_criterion["ms_ssim_loss"] is not None:
            losses["ms_ssim_loss"] = out_criterion["ms_ssim_loss"].item()
        
        return losses
    
    def val_step(self, batch, level):
        """验证步骤"""
        images = batch["image"].to(self.device)
        paths = batch["path"]
        B, C, H, W = images.shape

        pad_h = 0 if H % 64 == 0 else 64 * (H // 64 + 1) - H
        pad_w = 0 if W % 64 == 0 else 64 * (W // 64 + 1) - W
        images_pad = F.pad(images, (0, pad_w, 0, pad_h), mode='constant', value=0)        
        
        out_net = self.model.module(images_pad, s=level) if self.ddp_enable else self.model(images_pad)
        out_net['x_hat'] = out_net['x_hat'][:, :, :H, :W]
        out_criterion = self.criterion(out_net, images)

        aux_loss = self.model.module.aux_loss() if self.ddp_enable else self.model.aux_loss()
        bpp_loss = out_criterion["bpp_loss"]
        loss = out_criterion["loss"]

        rec = torch2img(out_net['x_hat'])
        img = torch2img(images)
        psnr, ms_ssim, lpips_m, dists = compute_metrics(rec, img)
        
        stem = Path(paths[0]).stem
        if not os.path.exists(self.val_images_dir):
            os.makedirs(self.val_images_dir)
        rec.save(os.path.join(self.val_images_dir, '%s_rec_lv%03d.png' % stem % level))
        img.save(os.path.join(self.val_images_dir, '%s_gt_lv%03d.png' % stem % level))

        if out_criterion["mse_loss"] is not None:
            smiliar = out_criterion["mse_loss"]
            
        if out_criterion["ms_ssim_loss"] is not None:
            smiliar = out_criterion["ms_ssim_loss"]

        metrics = {
            "path": batch["path"][0],
            "level": level,
            "loss": loss.item(),
            "aux_loss": aux_loss.item(),
            "bpp_loss": bpp_loss.item(),
            "similar_loss": smiliar.item(),
            "psnr": psnr,
            "ms_ssim": ms_ssim,
            "lpips": lpips_m,
            "dists": dists,
        }

        # losses_level.append(metrics)
        # if self.is_main_process():
        self.logger_val.info(pretty_print_dict(metrics))
            
        return metrics
    
    def _log_train(self, epoch, batch_idx):
        return super()._log_train(epoch, batch_idx)
    
    
    def validate_epoch(self, epoch):
        """验证一个 epoch"""
        self.model.eval()

        for i in range(self.model.levels):
        # 初始化验证指标
            self._init_val_metrics()

            with torch.inference_mode():
                for batch_idx, batch in enumerate(self.test_loader):
                    # 执行验证步骤
                    step_metrics_levels = self.val_step(batch, level=i)

                    # 更新验证指标
                    self._update_val_metrics(step_metrics_levels)

            # 记录日志
            if self.is_main_process():
                self._log_val(epoch, i)

        # 返回验证指标
        # Use the last one
        return self._get_val_metrics()
    
    def _log_val(self, epoch, level):
        """记录验证日志"""
        log_message = (
            f"Test epoch {epoch} at lv{level}: Loss: {self.val_loss.avg:.4f} | Bpp: {self.val_bpp_loss.avg:.4f} | "
            f"PSNR: {self.val_psnr.avg:.4f} | MS-SSIM: {self.val_ms_ssim.avg:.4f} | DIST: {self.val_dists.avg:.4f} | "
            f"LPIPS: {self.val_lpips.avg:.4f}"
        )
        self.logger_val.info(log_message)
        # 提取 TensorBoard 日志标签
        
        tb_val_loss_tag = f'Val/Loss_lv{level}'
        tb_val_bpp_tag = f'Val/Bpp_lv{level}'
        tb_val_psnr_tag = f'Val/PSNR_lv{level}'
        tb_val_ms_ssim_tag = f'Val/MS-SSIM_lv{level}'
        tb_val_dist_tag = f'Val/DIST_lv{level}'
        tb_val_lpips_tag = f'Val/LPIPS_lv{level}'
        self.tb_logger.add_scalar(tb_val_loss_tag, self.val_loss.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_bpp_tag, self.val_bpp_loss.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_psnr_tag, self.val_psnr.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_ms_ssim_tag, self.val_ms_ssim.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_dist_tag, self.val_dists.avg, epoch + 1)
        self.tb_logger.add_scalar(tb_val_lpips_tag, self.val_lpips.avg, epoch + 1)
