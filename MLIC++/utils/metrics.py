import torch
import numpy as np
import PIL.Image as Image
from typing import Dict, List, Optional, Tuple, Union
from pytorch_msssim import ms_ssim
from lpips import lpips
from DISTS_pytorch import DISTS

lp_fn = dists_fn = None
# lp_fn = lpips.LPIPS(net="vgg").eval().to("cpu")
# dists_fn = DISTS().eval().to("cpu")

def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    max_val: float = 255.0,
    device="cuda:0",
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    with torch.inference_mode():
        # MAX_PIXELS = 3000 * 2000
        # if a.shape[2] * a.shape[3] > MAX_PIXELS:
        #     lpips_m = lp_fn.to("cpu").forward(a.to("cpu") / max_val, b.to("cpu") / max_val, normalize=True).item()
        # else:
        try:
            if lp_fn is None:
                lp_fn = lpips.LPIPS(net="vgg").eval().to(device)
            lpips_m = lp_fn.to(device).forward(a.to(device) / max_val, b.to(device) / max_val, normalize=True).item()
        except:
            lpips_m = -1

        try:
            if dists is None:
                dists_fn = DISTS().eval().to(device)
            dists = dists_fn.to(device).forward(a.to(device) / max_val, b.to(device) / max_val).item()
        except:
            dists = -1
    return p, m, lpips_m, dists
