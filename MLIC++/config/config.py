import torch.nn as nn
from utils.utils import Config

# 2088.96 GMacs
def model_config(model_name="MLICPP_S"):
    if model_name == "MLICPP_L":
        config = Config({
            "N": 192,
            "M": 320,
            "slice_num": 10,
            "context_window": 5,
            "act": nn.GELU,
        }) # 1620 GMacs normal conv2d
    elif model_name == "MLICPP_S":
        config = Config({
            "N": 16 * 6,
            "M": 32 * 5,
            "slice_num": 5,
            "context_window": 5,
            "act": nn.GELU,
        }) # 436.35 GMacs normal conv2d; 204.55 GMacs depthwise conv
    elif model_name == "MLICPP_M":
        config = Config({
            "N": 16 * 10,
            "M": 32 * 8,
            "slice_num": 8,
            "context_window": 5,
            "act": nn.GELU,
        }) # 524.85 GMACs depthwise conv; 1120 GMacs normal conv2d

    return config
