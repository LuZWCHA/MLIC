import torch.nn as nn
from utils.utils import Config
import modules.layers.conv as conv

def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_deep_wise_conv=True):
    if use_deep_wise_conv:
        return conv.DepthWiseConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
    else:
        return conv.conv3x3_old(in_channels, out_channels, stride=stride)


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
    elif model_name in ["MLICPP_S", "MLICPP_S_VBR"]:
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
    elif model_name == "MLICPP_S2":
        config = Config({
            "N": 16 * 8,
            "M": 32 * 4,
            "slice_num": 2,
            "context_window": 5,
            "act": nn.GELU,
        }) # 436.35 GMacs normal conv2d; 204.55 GMacs depthwise conv
    elif model_name in ["MLICPP_M_SMALL_DEC", "MLICPP_M_SMALL_DEC_VBR"]:
        config = Config({
            "N": 192,
            "M": 320,
            "slice_num": 10,
            "context_window": 5,
            "act": nn.GELU,
        }) 


    return config
