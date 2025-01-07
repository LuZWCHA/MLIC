import PIL.Image as Image
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
from pathlib import Path
from torchvision.transforms import ToPILImage
import json


""" configuration json """
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace(filename.split('/')[-1], "checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, best_filename)


def split_data(source_path, destination_path, train_file):
    f = open(train_file, encoding='utf-8')
    while True:
        line = f.readline()
        if line:
            line = line.strip('\n')
            print(line)
            img_path = source_path + line
            shutil.move(img_path, destination_path + line)

import sys
import platform
import torch
import psutil

def get_system_info_str():
    def create_table(title, data, col1_width, col2_width):
        # 生成分隔线
        top_border = '+' + '-' * (col1_width + 2) + '+' + '-' * (col2_width + 2) + '+'
        header_sep = '+' + '=' * (col1_width + 2) + '+' + '=' * (col2_width + 2) + '+'
        # 生成标题行，居中
        title_line = '| {} |'.format(title.center(col1_width + col2_width + 3))
        # 生成列名
        header = '| {} | {} |'.format("Attribute".ljust(col1_width), "Value".ljust(col2_width))
        # 生成数据行
        rows = ['| {} | {} |'.format(str(key).ljust(col1_width), str(value).ljust(col2_width)) for key, value in data.items()]
        # 组合成一个字符串
        table_parts = [
            top_border,
            title_line,
            top_border,
            header,
            header_sep,
            '\n'.join(rows),
            top_border
        ]
        table_str = '\n'.join(table_parts)
        return table_str

    # 收集软件信息
    software_info = {
        "Operating System": f"{platform.system()} {platform.release()}",
        "Python Version": sys.version.split()[0],
        "PyTorch Version": torch.__version__
    }

    # 收集CPU信息
    cpu_info = {
        "Processor": platform.processor(),
        "Cores": psutil.cpu_count(logical=False),
        "Threads": psutil.cpu_count(logical=True)
    }

    # 收集GPU信息
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_info[f"GPU {i}"] = f"{name}, Memory: {mem:.2f} GB"
    else:
        gpu_info["GPU"] = "No GPU available"

    # 计算全局列宽
    all_keys = []
    all_values = []
    
    for info in [software_info, cpu_info, gpu_info]:
        all_keys.extend(info.keys())
        all_values.extend(info.values())
    
    global_col1_width = max(len(str(key)) for key in all_keys)
    global_col2_width = max(len(str(value)) for value in all_values)
    
    global_col1_width = max(global_col1_width, len("Attribute"))
    global_col2_width = max(global_col2_width, len("Value"))
    
    # 生成软件信息表格
    software_table = create_table("Software Information", software_info, global_col1_width, global_col2_width)
    
    # 生成CPU信息表格
    cpu_table = create_table("CPU Information", cpu_info, global_col1_width, global_col2_width)
    
    # 生成GPU信息表格
    gpu_table = create_table("GPU Information", gpu_info, global_col1_width, global_col2_width)
    
    # 拼接所有表格
    total_info_str = software_table + '\n\n' + cpu_table + '\n\n' + gpu_table

    return "\n" + total_info_str

if __name__ == "__main__":
    print(get_system_info_str())