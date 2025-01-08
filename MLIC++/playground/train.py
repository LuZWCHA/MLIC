import os
import random
import logging
import time
from PIL import ImageFile, Image
import math
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from utils.logger import setup_logger
from utils.utils import get_system_info_str, save_checkpoint
from utils.optimizers import configure_optimizers
from utils.training import train_one_epoch
from utils.testing import test_one_epoch
from loss.rd_loss import RateDistortionLoss
from config.args import train_options
from config.config import model_config
from models import *
import random
from ddp import *
from playground.dataset import ImageFolder2

def parse_gpu_ids(gpu_ids_str):
    # 替换中文逗号为英文逗号
    gpu_ids_str = gpu_ids_str.replace('，', ',')
    # 按逗号分割
    gpu_ids_list = gpu_ids_str.split(',')
    # 去除每个元素的空白并转换为整数
    gpu_ids = []
    for gpu_id_str in gpu_ids_list:
        gpu_id = gpu_id_str.strip()
        if gpu_id:
            try:
                gpu_ids.append(int(gpu_id))
            except ValueError:
                print(f"Warning: '{gpu_id}' is not a valid integer and will be ignored.")
    return gpu_ids

def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()
    config = model_config(args.model_name)

    gpu_ids = parse_gpu_ids(args.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    amp = args.amp
    
    ddp_enable = args.cuda and torch.cuda.device_count() > 1 and len(gpu_ids) > 0
    
    if args.seed is not None:
        seed = 100 * random.random()
    torch.manual_seed(seed)
    random.seed(seed)
    
    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))

    setup_logger('train', os.path.join('./experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', os.path.join('./experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('./experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('./experiments', args.experiment, 'checkpoints'))

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder2(args.dataset, split="train/openimagev7/testset", transform=train_transforms)
    test_dataset = ImageFolder2(args.dataset, split="test", transform=test_transforms)

    
    train_sampler = test_sampler =  None
    local_rank = -1
    if ddp_enable:
        local_rank = int(os.environ["LOCAL_RANK"]) ## DDP
        logger_train.info(os.environ)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
        
        from torch.utils.data.distributed import DistributedSampler ## DDP
        train_sampler = DistributedSampler(train_dataset)
        
    # logger_train.info(f"Local Rank: {local_rank}")
    if local_rank <= 0:
        logger_train.info(get_system_info_str())
        logger_train.info(f"DDP: {ddp_enable}")

        # test_sampler = DistributedSampler(test_dataset)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        sampler=train_sampler
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        # sampler=test_sampler
    )
    
    from models.model_loader import get_model
    net = get_model(args.model_name)
    
    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)


    if args.pretrained != None:
        checkpoint = torch.load(args.pretrained)
        # new_ckpt = modify_checkpoint(checkpoint['state_dict'])
        new_sd = dict()
        for k, v in checkpoint['state_dict'].items():
            k: str
            while k.startswith("module."):
                k = k[7:]
            new_sd[k] = v
        net.load_pretrained(new_sd)


    start_epoch = 0
    best_loss = 1e10
    current_step = 0
    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        # new_ckpt = modify_checkpoint(checkpoint['state_dict'])
        new_sd = dict()
        for k, v in checkpoint['state_dict'].items():
            k: str
            while k.startswith("module."):
                k = k[7:]
            new_sd[k] = v
        net.load_state_dict(new_sd)
        # net.load_state_dict(checkpoint['state_dict'])
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450,550], gamma=0.1)
            # lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
            # lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
            # print(lr_scheduler.state_dict())
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
            current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
            
        checkpoint = None

    if ddp_enable:
        net = DDP(net, device_ids=[local_rank]).to(device)
        
        
    # start_epoch = 0
    # best_loss = 1e10
    # current_step = 0
    
        
    optimizer.param_groups[0]['lr'] = args.learning_rate
    local_rank = dist.get_rank()
    
    if local_rank <= 0:
        logger_train.info(args)
        logger_train.info(config)
        logger_train.info(net)
        logger_train.info(optimizer)
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(start_epoch, args.epochs):
        if ddp_enable:
            train_dataloader.sampler.set_epoch(epoch)
        current_step = 0
        current_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step,
            rank=local_rank,
            amp=amp
        )
        
        if local_rank <= 0:
            save_dir = os.path.join('./experiments', args.experiment, 'val_images', '%03d' % (epoch + 1))
            # Test on gpuid=0
            loss = test_one_epoch(epoch, test_dataloader, net.module if isinstance(net, DDP) else net, criterion, save_dir, logger_val, tb_logger)
            
            if isinstance(net, DDP):
                net.module.update(force=True)
            else:
                net.update(force=True)
            
            if args.save:
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": net.module.state_dict() if isinstance(net, DDP) else net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    os.path.join('./experiments', args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch + 1))
                )
                if is_best:
                    logger_val.info('best checkpoint saved.')
                    
        else:
            if isinstance(net, DDP):
                net.module.update(force=True)
            else:
                net.update(force=True)

        lr_scheduler.step()
        print(f"{dist.get_rank()}: Waiting ...")
        if ddp_enable:
            dist.barrier()

if __name__ == '__main__':
    main()
    # torch.multiprocessing.spawn(main, args=(), nprocs=2, join=True)
