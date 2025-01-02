import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional, EntropyModel


def ckbd_split(y):
    """
    Split y to anchor and non-anchor
    anchor :
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
    non-anchor:
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
    """
    anchor = ckbd_anchor(y)
    nonanchor = ckbd_nonanchor(y)
    return anchor, nonanchor

def ckbd_merge(anchor, nonanchor):
    # out = torch.zeros_like(anchor).to(anchor.device)
    # out[:, :, 0::2, 0::2] = non_anchor[:, :, 0::2, 0::2]
    # out[:, :, 1::2, 1::2] = non_anchor[:, :, 1::2, 1::2]
    # out[:, :, 0::2, 1::2] = anchor[:, :, 0::2, 1::2]
    # out[:, :, 1::2, 0::2] = anchor[:, :, 1::2, 0::2]

    return anchor + nonanchor

def ckbd_anchor(y):
    anchor = torch.zeros_like(y).to(y.device)
    anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
    anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
    return anchor

def ckbd_nonanchor(y):
    nonanchor = torch.zeros_like(y).to(y.device)
    nonanchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
    nonanchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
    return nonanchor

def ckbd_anchor_sequeeze(y):
    B, C, H, W = y.shape
    anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
    anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
    anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
    return anchor

def ckbd_nonanchor_sequeeze(y):
    B, C, H, W = y.shape
    nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
    nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
    nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
    return nonanchor

def ckbd_anchor_unsequeeze(anchor):
    B, C, H, W = anchor.shape
    y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
    y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
    y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
    return y_anchor

def ckbd_nonanchor_unsequeeze(nonanchor):
    B, C, H, W = nonanchor.shape
    y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
    y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
    y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
    return y_nonanchor

from compressai.ops.ops import quantize_ste
def compress_anchor_vbr(gaussian_conditional:EntropyModel, QuantABCD, anchor, scales_anchor, means_anchor, symbols_list, indexes_list, scale, rescale, no_quantoffset, ):
    # squeeze anchor to avoid non-anchor symbols
    anchor_squeeze = ckbd_anchor_sequeeze(anchor)
    scales_anchor_squeeze = ckbd_anchor_sequeeze(scales_anchor)
    means_anchor_squeeze = ckbd_anchor_sequeeze(means_anchor)

    indexes = gaussian_conditional.build_indexes(scales_anchor_squeeze * scale)

    if no_quantoffset or (
                        no_quantoffset is False
                    ):
        anchor_hat = gaussian_conditional.quantize(anchor_squeeze - means_anchor_squeeze, "symbols", means_anchor_squeeze)
        symbols_list.extend(anchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        anchor_hat = ckbd_anchor_unsequeeze(anchor_hat * rescale + means_anchor_squeeze)

    else:
        y_zm = anchor_squeeze - means_anchor_squeeze.detach()
        y_zm_sc = y_zm * scale
        signs = torch.sign(y_zm_sc).detach()
        q_abs = quantize_ste(y_zm_sc)

        q_stdev = gaussian_conditional.lower_bound_scale(
            scales_anchor_squeeze * scale
        )

        stdev_and_gain = torch.cat(
            (
                q_stdev.unsqueeze(dim=2),
                scale.detach().expand(q_stdev.unsqueeze(dim=2).shape),
            ),
            dim=2,
        )
        q_offsets = (-1) * (
            QuantABCD.forward(stdev_and_gain)
        ).squeeze(dim=2)
        q_offsets[-0.0001 < q_abs < 0.0001] = (
            0  # must use zero offset for locations quantized to zero
        )

        anchor_hat = (signs * (q_abs + 0)).int()
        symbols_list.extend(anchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        anchor_hat = ckbd_anchor_unsequeeze((signs * (q_abs + q_offsets)) * rescale + means_anchor_squeeze)

    return anchor_hat

def compress_anchor(gaussian_conditional:EntropyModel, anchor, scales_anchor, means_anchor, symbols_list, indexes_list):
    # squeeze anchor to avoid non-anchor symbols
    anchor_squeeze = ckbd_anchor_sequeeze(anchor)
    scales_anchor_squeeze = ckbd_anchor_sequeeze(scales_anchor)
    means_anchor_squeeze = ckbd_anchor_sequeeze(means_anchor)
    indexes = gaussian_conditional.build_indexes(scales_anchor_squeeze)
    anchor_hat = gaussian_conditional.quantize(anchor_squeeze, "symbols", means_anchor_squeeze)
    symbols_list.extend(anchor_hat.reshape(-1).tolist())
    indexes_list.extend(indexes.reshape(-1).tolist())
    anchor_hat = ckbd_anchor_unsequeeze(anchor_hat + means_anchor_squeeze)
    return anchor_hat

def compress_nonanchor(gaussian_conditional:EntropyModel, nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list):
    nonanchor_squeeze = ckbd_nonanchor_sequeeze(nonanchor)
    scales_nonanchor_squeeze = ckbd_nonanchor_sequeeze(scales_nonanchor)
    means_nonanchor_squeeze = ckbd_nonanchor_sequeeze(means_nonanchor)
    indexes = gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
    nonanchor_hat = gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_nonanchor_squeeze)
    symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
    indexes_list.extend(indexes.reshape(-1).tolist())
    nonanchor_hat = ckbd_nonanchor_unsequeeze(nonanchor_hat + means_nonanchor_squeeze)
    return nonanchor_hat

def compress_nonanchor_vbr(gaussian_conditional:EntropyModel, QuantABCD, nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list, scale, rescale, no_quantoffset):
    nonanchor_squeeze = ckbd_nonanchor_sequeeze(nonanchor)
    scales_nonanchor_squeeze = ckbd_nonanchor_sequeeze(scales_nonanchor)
    means_nonanchor_squeeze = ckbd_nonanchor_sequeeze(means_nonanchor)

    indexes = gaussian_conditional.build_indexes(scales_nonanchor_squeeze * scale)
    if no_quantoffset or (
                        no_quantoffset is False
                    ):
        nonanchor_hat = gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_nonanchor_squeeze)
        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        nonanchor_hat = ckbd_nonanchor_unsequeeze(nonanchor_hat * rescale + means_nonanchor_squeeze)
    else:
        nonanchor_hat = gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_nonanchor_squeeze)
        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        nonanchor_hat = ckbd_nonanchor_unsequeeze(nonanchor_hat + means_nonanchor_squeeze)
        y_zm = nonanchor_squeeze - means_nonanchor_squeeze.detach()
        y_zm_sc = y_zm * scale
        signs = torch.sign(y_zm_sc).detach()
        q_abs = quantize_ste(y_zm_sc)

        q_stdev = gaussian_conditional.lower_bound_scale(
            scales_nonanchor_squeeze * scale
        )

        stdev_and_gain = torch.cat(
            (
                q_stdev.unsqueeze(dim=2),
                scale.detach().expand(q_stdev.unsqueeze(dim=2).shape),
            ),
            dim=2,
        )
        q_offsets = (-1) * (
            QuantABCD.forward(stdev_and_gain)
        ).squeeze(dim=2)
        q_offsets[-0.0001 < q_abs < 0.0001] = (
            0  # must use zero offset for locations quantized to zero
        )

        nonanchor_hat = (signs * (q_abs + 0)).int()
        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        nonanchor_hat = ckbd_anchor_unsequeeze((signs * (q_abs + q_offsets)) * rescale + means_nonanchor_squeeze)

    
    return nonanchor_hat

def decompress_anchor(gaussian_conditional:EntropyModel, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets):
    scales_anchor_squeeze = ckbd_anchor_sequeeze(scales_anchor)
    means_anchor_squeeze = ckbd_anchor_sequeeze(means_anchor)
    indexes = gaussian_conditional.build_indexes(scales_anchor_squeeze)
    anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
    anchor_hat = torch.Tensor(anchor_hat).reshape(scales_anchor_squeeze.shape).to(scales_anchor.device) + means_anchor_squeeze
    anchor_hat = ckbd_anchor_unsequeeze(anchor_hat)
    return anchor_hat

def decompress_anchor_vbr(gaussian_conditional:EntropyModel, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets, scale, rescale):
    scales_anchor_squeeze = ckbd_anchor_sequeeze(scales_anchor)
    means_anchor_squeeze = ckbd_anchor_sequeeze(means_anchor)
    indexes = gaussian_conditional.build_indexes(scales_anchor_squeeze * scale)
    anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
    anchor_hat = torch.Tensor(anchor_hat).reshape(scales_anchor_squeeze.shape).to(scales_anchor.device) * rescale + means_anchor_squeeze
    anchor_hat = ckbd_anchor_unsequeeze(anchor_hat)
    return anchor_hat

def decompress_nonanchor(gaussian_conditional:EntropyModel, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets):
    scales_nonanchor_squeeze = ckbd_nonanchor_sequeeze(scales_nonanchor)
    means_nonanchor_squeeze = ckbd_nonanchor_sequeeze(means_nonanchor)
    indexes = gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
    nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
    nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_nonanchor_squeeze.shape).to(scales_nonanchor.device) + means_nonanchor_squeeze
    nonanchor_hat = ckbd_nonanchor_unsequeeze(nonanchor_hat)
    return nonanchor_hat

def decompress_nonanchor_vbr(gaussian_conditional:EntropyModel, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets, scale, rescale):
    scales_nonanchor_squeeze = ckbd_nonanchor_sequeeze(scales_nonanchor)
    means_nonanchor_squeeze = ckbd_nonanchor_sequeeze(means_nonanchor)
    indexes = gaussian_conditional.build_indexes(scales_nonanchor_squeeze * scale)
    nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
    nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_nonanchor_squeeze.shape).to(scales_nonanchor.device) * rescale + means_nonanchor_squeeze
    nonanchor_hat = ckbd_nonanchor_unsequeeze(nonanchor_hat)
    return nonanchor_hat
