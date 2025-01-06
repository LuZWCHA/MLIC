from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

from config.args import test_options
from config.config import model_config
from compressai.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import ImageFile, Image
from models import *
from utils.testing import test_model
from utils.logger import setup_logger


def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options()
    config = model_config(args.model_name)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    torch.backends.cudnn.deterministic = True

    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))
    setup_logger('test', os.path.join('./experiments', args.experiment), 'test_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = ImageFolder(args.dataset, split=".", transform=test_transforms)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    from models.model_loader import get_model
    net = get_model(args.model_name)

    net = net.to(device)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        new_sd = dict()
        for k, v in checkpoint['state_dict'].items():
            k: str
            while k.startswith("module."):
                k = k[7:]
            new_sd[k] = v
        net.load_state_dict(new_sd)
        epoch = checkpoint["epoch"] if "epoch" in checkpoint else 0
    else:
        epoch = 0
        print("Checkpoint not found.")
    
    logger_test.info(f"Start testing!" )
    save_dir = os.path.join('./experiments', args.experiment, 'codestream', '%02d' % (epoch + 1))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_model(net=net, test_dataloader=test_dataloader, logger_test=logger_test, save_dir=save_dir, epoch=epoch)


if __name__ == '__main__':
    main()

