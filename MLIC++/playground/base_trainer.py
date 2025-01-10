import os
import logging
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.logger import setup_logger
from utils.utils import save_checkpoint
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, args=None, **kwargs):
        """
        初始化 BaseTrainer。

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
        self.device = self._setup_device()
        self.ddp_enable = self.args.cuda and torch.cuda.device_count() > 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        torch.cuda.set_device(self.local_rank)
        # dist.init_process_group(backend='nccl')
        print(f"init at rank: {self.local_rank}")

        # 初始化分布式训练
        if self.ddp_enable and self.device == "cuda":
            dist.init_process_group(backend="nccl" if self.args.cuda else "gloo")
            self.device = torch.device("cuda", self.local_rank)
        
        # 提取路径常量
        self.experiment_dir = os.path.join('./experiments', self.args.experiment)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.val_images_dir = os.path.join(self.experiment_dir, 'val_images')
        self.tb_log_dir = os.path.join('./tb_logger', self.args.experiment)
        self.eval_first = True

        # 初始化日志记录
        self._setup_logging()

        # 初始化数据加载器
        self.__setup_data()

        # 初始化模型
        self.model = self._setup_model()
        assert self.model is not None, "Model should not None."

        # 初始化优化器
        self._setup_optimizers()

        # 初始化损失函数
        self._setup_criterion()

        # 初始化训练状态
        self.__rest()

        # 加载检查点（如果提供了 checkpoint 路径）
        if self.args.checkpoint is not None:
            self._load_checkpoint()

    def __rest(self):
        # 初始化训练状态
        self.best_loss = float('inf')
        self.current_step = 0
        self.start_epoch = 0

    def __setup_data(self):
        """初始化训练和测试数据集"""
        self.train_loader = self._setup_train_data()
        self.test_loader = self._setup_test_data()

    def _setup_train_data(self):
        raise NotImplementedError()
    
    def _setup_test_data(self):
        raise NotImplementedError()

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
        """初始化模型"""
        raise NotImplementedError("Subclasses should implement this method")

    def _setup_optimizers(self):
        """初始化优化器"""
        raise NotImplementedError("Subclasses should implement this method")

    def _setup_criterion(self):
        """初始化损失函数"""
        raise NotImplementedError("Subclasses should implement this method")
    
    def is_main_process(self):
        """判断当前进程是否是主进程（rank 0）"""
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True  # 单机训练时默认是主进程

    def train_stage(self, epoch):
        # 训练一个 epoch
        self.train_epoch(epoch)

    def eval_stage(self, epoch):
        # 验证
        if self.is_main_process():
            val_metrics = self.validate_epoch(epoch)
        
        # 保存最佳模型
        if self.is_main_process() and val_metrics["loss"] < self.best_loss:
            self.best_loss = val_metrics["loss"]
            self._save_checkpoint(epoch, val_metrics["loss"])

        dist.barrier()

    def fit(self):
        """训练循环"""
        for epoch in range(self.args.epochs):
            if self.ddp_enable:
                self.train_loader.sampler.set_epoch(epoch)

            if self.eval_first:
                self.eval_stage(epoch)
                self.train_stage(epoch)
            else:
                self.train_stage(epoch)
                self.eval_stage(epoch)
            
            # 更新学习率
            if hasattr(self, "lr_scheduler"):
                self.lr_scheduler.step()

        # 清理分布式训练
        if self.ddp_enable:
            dist.destroy_process_group()

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)

        # 初始化训练指标
        self._init_train_metrics()

        # 使用 tqdm 显示进度条
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            # 执行训练步骤
            step_metrics = self.train_step(batch, scaler)

            # 更新训练指标
            self._update_train_metrics(step_metrics)

            # 记录日志
            if self.is_main_process() and batch_idx % self.args.log_freq == 0:
                self._log_train(epoch, batch_idx)

            # 更新进度条描述
            progress_bar.set_postfix(loss=step_metrics.get("loss", "N/A"))

        # 关闭进度条
        progress_bar.close()

    def validate_epoch(self, epoch):
        """验证一个 epoch"""
        self.model.eval()

        # 初始化验证指标
        self._init_val_metrics()

        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.test_loader):
                # 执行验证步骤
                step_metrics = self.val_step(batch)

                # 更新验证指标
                self._update_val_metrics(step_metrics)

        # 记录日志
        if self.is_main_process():
            self._log_val(epoch)

        # 返回验证指标
        return self._get_val_metrics()

    def train_step(self, batch, scaler):
        """训练步骤，由子类实现"""
        raise NotImplementedError("Subclasses should implement this method")

    def val_step(self, batch):
        """验证步骤，由子类实现"""
        raise NotImplementedError("Subclasses should implement this method")

    def _init_train_metrics(self):
        """初始化训练指标，由子类实现"""
        raise NotImplementedError("Subclasses should implement this method")

    def _update_train_metrics(self, step_metrics):
        """更新训练指标，由子类实现"""
        raise NotImplementedError("Subclasses should implement this method")

    def _log_train(self, epoch, batch_idx):
        """记录训练日志，由子类实现"""
        raise NotImplementedError("Subclasses should implement this method")

    def _init_val_metrics(self):
        """初始化验证指标，由子类实现"""
        raise NotImplementedError("Subclasses should implement this method")

    def _update_val_metrics(self, step_metrics):
        """更新验证指标，由子类实现"""
        raise NotImplementedError("Subclasses should implement this method")

    def _log_val(self, epoch):
        """记录验证日志，由子类实现"""
        raise NotImplementedError("Subclasses should implement this method")

    def _get_val_metrics(self):
        """获取验证指标，由子类实现"""
        raise NotImplementedError("Subclasses should implement this method")

    def _load_checkpoint(self):
        pass

    def _save_checkpoint(self, epoch, val_loss):
        """保存检查点"""
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
