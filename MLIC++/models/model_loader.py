import  config.config as cf
from models import *

def get_model(model_name):
    model_config_ = cf.model_config(model_name)
    model_cls = MLICPlusPlus

    if model_name in ["MLICPP_L", "MLICPP_M", "MLICPP_S", "MLICPP_S2"]:
        model_cls = MLICPlusPlus
    elif model_name in ["MLICPP_M_SMALL_DEC"]:
        model_cls = MLICPlusPlusSD
    elif model_name in ["MLICPP_S_VBR"]:
        model_cls = MLICPlusPlusVbr


    return model_cls(config=model_config_)
