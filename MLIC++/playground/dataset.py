# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import glob
import math
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from compressai.registry import register_dataset


@register_dataset("ImageFolder2")
class ImageFolder2(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(glob.glob(str(splitdir/"**/*.jpg"), recursive=True) + glob.glob(str(splitdir/"**/*.png"), recursive=True))

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"image": img, "path": str(self.samples[index])}

    def __len__(self):
        return len(self.samples)
    
from torchvision.transforms import functional as F

class RandomResize:
    def __init__(self, c=3.2, interpolation=Image.BILINEAR):
        """
        Random area scaling transformation with scale factors in [0.2, 5].
        
        Parameters:
            c (float): Controls the range of scaling, s ∈ [exp(-c), exp(c)].
            interpolation: Interpolation method, default is Image.BILINEAR.
        """
        self.c = c
        self.interpolation = interpolation

    def __call__(self, img):
        # Sample ln(s) uniformly from [-c, c]
        ln_s = torch.empty(1).uniform_(-self.c, self.c).item()
        # Compute scaling factor s
        s = math.exp(ln_s)
        # Calculate scale factor for width and height
        scale_factor = math.sqrt(s)
        # Get original image dimensions
        width, height = img.size
        # Compute new dimensions
        new_width = max(1, int(width * scale_factor))
        new_height = max(1, int(height * scale_factor))
        # Resize the image
        return F.resize(img, (new_height, new_width), interpolation=self.interpolation)
