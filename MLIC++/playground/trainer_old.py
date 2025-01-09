import math
import os
import logging
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.logger import setup_logger
from utils.utils import AverageMeter, get_system_info_str, save_checkpoint
from utils.optimizers import configure_optimizers
from loss.rd_loss import RateDistortionLoss
from config.args import train_options
from config.config import model_config
from models import get_model
from playground.dataset import ImageFolder2, RandomResize
from utils.metrics import compute_metrics
from utils.utils import torch2img
import torch.nn.functional as F
from base_trainer import BaseDataModule

from playground.dataset import ImageFolder2, RandomResize

class Trainer:
    def __init__(self, args=None, **kwargs):
        """
        初始化 Trainer。

        参数:
            args: 命令行参数或配置对象（可以是字典或 argparse.Namespace）。
            **kwargs: 额外的参数，用于覆盖 args 中的参数。
        """
        # 如果 args 是字典，将其转换为 SimpleNamespace
        if isinstance(args, dict):
            args = SimpleNamespace(**args)

        # 如果 args 为 None，使用默认参数
        if args is None:
            args = self._get_default_args()

        # 覆盖 args 中的参数（如果 kwargs 中有同名参数）
        for key, value in kwargs.items():
            if hasattr(args, key):  # 如果 args 中有该参数，则覆盖
                setattr(args, key, value)
            else:
                print(f"Warning: '{key}' is not a valid argument and will be ignored.")
                
        self.args = args
        self.config = model_config(args.model_name)
        self.device = self._setup_device()
        self.ddp_enable = self.args.cuda and torch.cuda.device_count() > 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))

        # 初始化分布式训练
        if self.ddp_enable:
            dist.init_process_group(backend="nccl" if self.args.cuda else "gloo")

        # 提取路径常量
        self.experiment_dir = os.path.join('./experiments', self.args.experiment)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.val_images_dir = os.path.join(self.experiment_dir, 'val_images')
        self.tb_log_dir = os.path.join('./tb_logger', self.args.experiment)

        # 提取日志格式字符串
        self.train_log_format = (
            "Train epoch {epoch}: [{processed}/{total} ({percentage:.0f}%)] "
            "Loss: {loss:.4f} | Bpp loss: {bpp_loss:.2f} | Aux loss: {aux_loss:.2f}"
        )
        self.val_log_format = (
            "Test epoch {epoch}: Loss: {loss:.4f} | Bpp: {bpp:.4f} | "
            "PSNR: {psnr:.4f} | MS-SSIM: {ms_ssim:.4f}"
        )

        # 提取 TensorBoard 日志标签
        self.tb_train_loss_tag = 'Train/Loss'
        self.tb_val_loss_tag = 'Val/Loss'
        self.tb_val_bpp_tag = 'Val/Bpp'
        self.tb_val_psnr_tag = 'Val/PSNR'
        self.tb_val_ms_ssim_tag = 'Val/MS-SSIM'

        self._setup_logging()
        self._setup_data()
        self._setup_model()
        self._setup_optimizers()
        self._setup_criterion()

        # 加载检查点（如果提供了 checkpoint 路径）
        if self.args.checkpoint is not None:
            self._load_checkpoint()

    def _setup_train_data(self):
        """初始化训练数据集"""
        train_transforms = transforms.Compose([
            transforms.AutoAugment(),
            RandomResize(),
            transforms.RandomCrop(self.args.patch_size, pad_if_needed=True),
            transforms.ToTensor()
        ])
        self.train_dataset = ImageFolder2(self.args.dataset, split="train/openimagev7/testset", transform=train_transforms)

        self.train_sampler = DistributedSampler(self.train_dataset) if self.ddp_enable else None
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=(self.train_sampler is None),
            pin_memory=(self.device == "cuda"),
            sampler=self.train_sampler
        )

    def _setup_test_data(self):
        """初始化测试数据集"""
        test_transforms = transforms.Compose([transforms.ToTensor()])
        self.test_dataset = ImageFolder2(self.args.dataset, split="test", transform=test_transforms)

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=(self.device == "cuda")
        )

    def _setup_data(self):
        """初始化训练和测试数据集"""
        self._setup_train_data()
        self._setup_test_data()

    def is_main_process(self):
        """判断当前进程是否是主进程（rank 0）"""
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True  # 单机训练时默认是主进程

    def _setup_device(self):
        gpu_ids = self._parse_gpu_ids(self.args.gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
        return "cuda" if self.args.cuda and torch.cuda.is_available() else "cpu"

    def _parse_gpu_ids(self, gpu_ids_str):
        gpu_ids_str = gpu_ids_str.replace('，', ',')
        gpu_ids_list = gpu_ids_str.split(',')
        gpu_ids = []
        for gpu_id_str in gpu_ids_list:
            gpu_id = gpu_id_str.strip()
            if gpu_id:
                try:
                    gpu_ids.append(int(gpu_id))
                except ValueError:
                    print(f"Warning: '{gpu_id}' is not a valid integer and will be ignored.")
        return gpu_ids

    def _setup_logging(self):
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        setup_logger('train', self.experiment_dir, 'train_' + self.args.experiment, level=logging.INFO, screen=True, tofile=True)
        setup_logger('val', self.experiment_dir, 'val_' + self.args.experiment, level=logging.INFO, screen=True, tofile=True)

        self.logger_train = logging.getLogger('train')
        self.logger_val = logging.getLogger('val')
        self.tb_logger = SummaryWriter(log_dir=self.tb_log_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def _setup_model(self):
        self.model = get_model(self.args.model_name).to(self.device)
        if self.args.pretrained:
            checkpoint = torch.load(self.args.pretrained)
            new_sd = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(new_sd)

        self._load_checkpoint()
        
        if self.ddp_enable:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])

    def _load_checkpoint(self):
        """加载模型检查点，支持恢复训练或仅加载模型权重"""
        checkpoint = torch.load(self.args.checkpoint)

        # 处理模型权重（移除 "module." 前缀以兼容单机和分布式训练）
        new_sd = {}
        for k, v in checkpoint['state_dict'].items():
            while k.startswith("module."):
                k = k[7:]  # 移除 "module." 前缀
            new_sd[k] = v
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


    def _setup_optimizers(self):
        self.optimizer, self.aux_optimizer = configure_optimizers(self.model, self.args)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 100], gamma=0.1)

    def _setup_criterion(self):
        self.criterion = RateDistortionLoss(lmbda=self.args.lmbda, metrics=self.args.metrics)

    def train_one_epoch(self, epoch):
        self.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad()
            self.aux_optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.args.amp):
                out_net = self.model(images)
                out_criterion = self.criterion(out_net, images)

            scaler.scale(out_criterion["loss"]).backward()
            if self.args.clip_max_norm > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm)
            scaler.step(self.optimizer)

            aux_loss  = self.model.module.aux_loss() if self.ddp_enable else self.model.aux_loss()
            scaler.scale(aux_loss).backward()
            scaler.step(self.aux_optimizer)
            scaler.update()

            self.current_step += 1

            if self.is_main_process() and batch_idx % self.args.log_freq == 0:
                log_message = self.train_log_format.format(
                    epoch=epoch,
                    processed=batch_idx * len(images),
                    total=len(self.train_loader.dataset),
                    percentage=100. * batch_idx / len(self.train_loader),
                    loss=out_criterion["loss"].item(),
                    bpp_loss=out_criterion["bpp_loss"].item(),
                    aux_loss=aux_loss.item()
                )
                self.logger_train.info(log_message)
                self.tb_logger.add_scalar(self.tb_train_loss_tag, out_criterion["loss"].item(), epoch * len(self.train_loader) + batch_idx)

    def validate(self, epoch):
        self.model.eval()
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
            for batch_idx, batch in enumerate(self.test_loader):
                images = batch["image"].to(self.device)
                B, C, H, W = images.shape

                pad_h = 0 if H % 64 == 0 else 64 * (H // 64 + 1) - H
                pad_w = 0 if W % 64 == 0 else 64 * (W // 64 + 1) - W
                images_pad = F.pad(images, (0, pad_w, 0, pad_h), mode='constant', value=0)

                out_net = self.model(images_pad)
                out_net['x_hat'] = out_net['x_hat'][:, :, :H, :W]
                out_criterion = self.criterion(out_net, images)

                aux_loss.update(self.model.module.aux_loss() if self.ddp_enable else self.model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                if out_criterion["mse_loss"] is not None:
                    mse_loss.update(out_criterion["mse_loss"])
                if out_criterion["ms_ssim_loss"] is not None:
                    ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

                rec = torch2img(out_net['x_hat'])
                img = torch2img(images)
                p, m, lpips_m, dists = compute_metrics(rec, img)
                psnr.update(p)
                ms_ssim.update(m)
                lpips_loss.update(lpips_m)
                avg_dists.update(dists)

                if not os.path.exists(os.path.join(self.val_images_dir, f'{epoch + 1:03d}')):
                    os.makedirs(os.path.join(self.val_images_dir, f'{epoch + 1:03d}'))

                rec.save(os.path.join(self.val_images_dir, f'{epoch + 1:03d}', f'{batch_idx:03d}_rec.png'))
                img.save(os.path.join(self.val_images_dir, f'{epoch + 1:03d}', f'{batch_idx:03d}_gt.png'))

        if self.is_main_process():
            log_message = self.val_log_format.format(
                epoch=epoch,
                loss=loss.avg,
                bpp=bpp_loss.avg,
                psnr=psnr.avg,
                ms_ssim=ms_ssim.avg
            )
            self.logger_val.info(log_message)
            self.tb_logger.add_scalar(self.tb_val_loss_tag, loss.avg, epoch + 1)
            self.tb_logger.add_scalar(self.tb_val_bpp_tag, bpp_loss.avg, epoch + 1)
            self.tb_logger.add_scalar(self.tb_val_psnr_tag, psnr.avg, epoch + 1)
            self.tb_logger.add_scalar(self.tb_val_ms_ssim_tag, ms_ssim.avg, epoch + 1)

        return loss.avg

    def fit(self):
        for epoch in range(self.args.epochs):
            if self.ddp_enable:
                self.train_loader.sampler.set_epoch(epoch)

            self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            if self.is_main_process() and val_loss < self.best_loss:
                self.best_loss = val_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch + 1:03d}.pth.tar")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.module.state_dict() if self.ddp_enable else self.model.state_dict(),
                        "loss": val_loss,
                        "optimizer": self.optimizer.state_dict(),
                        "aux_optimizer": self.aux_optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    },
                    True,
                    checkpoint_path
                )
                self.logger_val.info('Best checkpoint saved.')

            self.lr_scheduler.step()

        if self.ddp_enable:
            dist.destroy_process_group()


    def _get_default_args(self):
        """返回默认参数"""
        return SimpleNamespace(
            experiment="default_experiment",
            amp=False,
            resume=False,
            log_freq=20,
            model_name="MLICPP_L",
            dataset="/home/npr/dataset/",
            epochs=500,
            learning_rate=1e-4,
            num_workers=8,
            lmbda=0.045,
            metrics="mse",
            batch_size=8,
            test_batch_size=1,
            aux_learning_rate=1e-3,
            patch_size=(256, 256),
            gpu_id="0,1",
            cuda=True,
            save=True,
            seed=2025,
            clip_max_norm=1.0,
            checkpoint=None,
            pretrained=None,
            world_size=1,
            dist_url='env://'
        )

if __name__ == '__main__':
    args = train_options()
    trainer = Trainer(args)
    trainer.fit()
