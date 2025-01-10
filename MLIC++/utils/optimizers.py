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

    optimizer = optim.Adam(
        (p for p in parameters if p.requires_grad),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (p for p in aux_parameters if p.requires_grad),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def configure_optimizers_mmo(net, args, key="Gain"):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = [
        p for n, p in net.named_parameters() if not n.endswith(".quantiles") and not key in n
    ]
    
    gain_parameters = [
        p for n, p in net.named_parameters() if key in n
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

    optimizer = optim.Adam(
        (p for p in parameters if p.requires_grad),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (p for p in aux_parameters if p.requires_grad),
        lr=args.aux_learning_rate,
    )
    
    assert len(gain_parameters) == 1
    gain_parameter = gain_parameters[0]
    
    gain_optimizers = []
    for i in len(gain_parameter):
        gain_optimizers.append(optim.Adam(
            (p for p in gain_parameter[i] if p.requires_grad),
            lr=args.learning_rate,
        ))
    
    return optimizer, aux_optimizer, gain_optimizers
