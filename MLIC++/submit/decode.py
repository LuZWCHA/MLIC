import argparse
import glob
import os
from pathlib import Path
import sys
import numpy as np
import torch
from models import MLICPlusPlus
from config.config import model_config
from utils.utils import read_uints, read_body, write_uints, write_body
from models.model_loader import get_model
from PIL import ImageFile, Image
import PIL.Image as Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def load_model(model_name, path=None):
    net = get_model(model_name)
    net = net.cuda()
    net.eval()
    if path is not None:
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["state_dict"])
    return net

@torch.no_grad()
def decode(net, bits_path):
    net.update_resolutions(16, 16)
    with open(bits_path, 'rb') as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)
        out = net.decompress(strings, shape)
        x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    cost_time = out["cost_time"]
        
    return net, out['x_hat'], cost_time

def save_img(x_hat, save_path):
    x_hat = x_hat.squeeze(0)
    x_hat = x_hat.permute(1, 2, 0)
    x_hat = x_hat.cpu().numpy()
    x_hat = (x_hat * 255).astype(np.uint8)
    Image.fromarray(x_hat).save(save_path)
    
if __name__ == '__main__':
    
    paser = argparse.ArgumentParser()
    paser.add_argument('--path', type=str, default='/nasdata2/private/zwlu/compress/naic2024/MLIC/MLIC++/playground/experiments/chusai_exp_mlicpp_new_mse_q1/codestream/119')
    paser.add_argument('--checkpoint', type=str, default='/nasdata2/private/zwlu/compress/naic2024/MLIC/MLIC++/playground/experiments/chusai_exp_mlicpp_new_mse_q1/checkpoints/checkpoint_best_loss.pth.tar')
    paser.add_argument('--model_name', type=str, default='MLICPP_L')
    paser.add_argument('--output', type=str, default='output')
    args = paser.parse_args()
    net = args.model_name
    net = load_model(net, args.checkpoint)
    
    for i in glob.glob(os.path.join(args.path, '*')):
        if "." not in Path(i).name or Path(i).name.endswith(".bit"):
            net, img, cost_time = decode(net, i)
            image_name = Path(i).stem + ".png"
            save_img(img, os.path.join(args.output, image_name))
            print(f"decode {i} done, cost time: {cost_time}")
            
