import torch
import torch.nn as nn
import torch.optim as optim


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = [
        p for n, p in net.named_parameters() if not n.endswith(".quantiles")
    ]
    aux_parameters = [
        p for n, p in net.named_parameters() if n.endswith(".quantiles")
    ]

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = set(parameters) & set(aux_parameters)
    union_params = set(parameters) | set(aux_parameters)
    
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    if hasattr(args, "optimizer"):
        if args.optimizer == "Adam":
            optim_cls = optim.Adam
        elif args.optimizer == "AdamW":
            optim_cls = optim.AdamW
        elif args.optimizer == "SGD":
            optim_cls = optim.SGD
        else:
            raise ValueError("Unknown optimizer: {}".format(args.optimizer))
    else:
        optim_cls = optim.Adam

    optimizer = optim_cls(
        (p for p in parameters if p.requires_grad),
        lr=args.learning_rate,
    )
    aux_optimizer = optim_cls(
        (p for p in aux_parameters if p.requires_grad),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def configure_optimizers_mmo(net, args, keys=["Gain", "gain"]):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = [
        p for n, p in net.named_parameters() if not n.endswith(".quantiles") and not any(key in n for key in keys)
    ]
    
    gain_parameters = [
        p for n, p in net.named_parameters() if any(key in n for key in keys)
    ]
    
    aux_parameters = [
        p for n, p in net.named_parameters() if n.endswith(".quantiles")
    ]

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = set(parameters) & set(aux_parameters)
    union_params = set(parameters) | set(aux_parameters)

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    if hasattr(args, "optimizer"):
        if args.optimizer == "Adam":
            optim_cls = optim.Adam
        elif args.optimizer == "AdamW":
            optim_cls = optim.AdamW
        elif args.optimizer == "SGD":
            optim_cls = optim.SGD
        else:
            raise ValueError("Unknown optimizer: {}".format(args.optimizer))
    else:
        optim_cls = optim.Adam

    optimizer = optim_cls(
        (p for p in parameters if p.requires_grad),
        lr=args.learning_rate,
    )
    aux_optimizer = optim_cls(
        (p for p in aux_parameters if p.requires_grad),
        lr=args.aux_learning_rate,
    )
    
    assert len(gain_parameters) == 1
    gain_parameter = gain_parameters[0]
    
    gain_optimizers = []
    for i in len(gain_parameter):
        gain_optimizers.append(optim_cls(
            (p for p in gain_parameter[i] if p.requires_grad),
            lr=args.learning_rate,
        ))
    
    return optimizer, aux_optimizer, gain_optimizers
