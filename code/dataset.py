"""
Dataset generators and masking loaders for EdgeVerify.

Two sources are provided:

  * ``StructuredJEPADataset`` -- procedurally generated 64x64 RGB images with
    genuine spatial structure (background gradients plus randomly placed
    rectangles and circles). Unlike i.i.d. uniform noise, these images contain
    predictable local structure, so a context encoder can learn a non-trivial
    representation and the collapse ablation is meaningful.

  * ``DigitsJEPADataset`` -- the scikit-learn handwritten-digits dataset
    (real 8x8 images, 1797 samples), upsampled to 64x64 and replicated to
    three channels. This is real (non-synthetic) data available offline, used
    as a sanity check.

Each sample is a triple ``(partial_image, full_image, spatial_action)`` matching
the signature of ``VisionJEPA.forward``. The partial (context) view is produced
by zeroing a rectangular region of the full image; the spatial action is the
normalized bounding box ``[x, y, w, h]`` of that masked region, so the predictor
is conditioned on the location it must reconstruct in latent space.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _apply_mask(img, rng, min_frac=0.35, max_frac=0.6):
    """Zero a random rectangular region; return (partial, action[x,y,w,h] normalized)."""
    _, H, W = img.shape
    w = int(rng.uniform(min_frac, max_frac) * W)
    h = int(rng.uniform(min_frac, max_frac) * H)
    x = int(rng.integers(0, W - w + 1))
    y = int(rng.integers(0, H - h + 1))
    partial = img.clone()
    partial[:, y:y + h, x:x + w] = 0.0
    action = torch.tensor([x / W, y / H, w / W, h / H], dtype=torch.float32)
    return partial, action


def _make_structured_image(rng, size=64):
    """Background gradient + a few random rectangles and circles, values in [0,1]."""
    img = np.zeros((3, size, size), dtype=np.float32)
    # Background gradient between two random colours.
    c0 = rng.uniform(0, 1, size=3)
    c1 = rng.uniform(0, 1, size=3)
    if rng.random() < 0.5:
        ramp = np.linspace(0, 1, size)[None, :]        # horizontal
    else:
        ramp = np.linspace(0, 1, size)[:, None]        # vertical
    for c in range(3):
        img[c] = c0[c] + (c1[c] - c0[c]) * ramp
    # Random rectangles.
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(int(rng.integers(1, 4))):
        rw, rh = rng.integers(8, 28), rng.integers(8, 28)
        rx, ry = rng.integers(0, size - rw), rng.integers(0, size - rh)
        col = rng.uniform(0, 1, size=3)
        img[:, ry:ry + rh, rx:rx + rw] = col[:, None, None]
    # Random filled circles.
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.integers(10, size - 10), rng.integers(10, size - 10)
        r = rng.integers(5, 14)
        col = rng.uniform(0, 1, size=3)
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        for c in range(3):
            img[c][disk] = col[c]
    return torch.from_numpy(np.clip(img, 0.0, 1.0))


class StructuredJEPADataset(Dataset):
    def __init__(self, n_samples=1024, size=64, seed=0):
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n_samples):
            full = _make_structured_image(rng, size)
            partial, action = _apply_mask(full, rng)
            self.samples.append((partial, full, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DigitsJEPADataset(Dataset):
    def __init__(self, size=64, seed=0):
        from sklearn.datasets import load_digits
        rng = np.random.default_rng(seed)
        digits = load_digits()
        imgs = digits.images.astype(np.float32) / 16.0          # (1797, 8, 8), scaled to [0,1]
        t = torch.from_numpy(imgs).unsqueeze(1)                   # (N, 1, 8, 8)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        t = t.repeat(1, 3, 1, 1)                                  # (N, 3, size, size)
        self.samples = []
        for i in range(t.size(0)):
            full = t[i].contiguous()
            partial, action = _apply_mask(full, rng)
            self.samples.append((partial, full, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CIFARJEPADataset(Dataset):
    """CIFAR-10 subset (requires network access to download via torchvision)."""
    def __init__(self, n_samples=2000, size=64, seed=0, root="./cifar"):
        import torchvision
        rng = np.random.default_rng(seed)
        tv = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        data = torch.from_numpy(tv.data[:n_samples]).float().permute(0, 3, 1, 2) / 255.0
        data = F.interpolate(data, size=(size, size), mode="bilinear", align_corners=False)
        self.samples = []
        for i in range(data.size(0)):
            full = data[i].contiguous()
            partial, action = _apply_mask(full, rng)
            self.samples.append((partial, full, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_dataset(name, **kwargs):
    name = name.lower()
    if name in ("structured", "shapes"):
        return StructuredJEPADataset(**kwargs)
    if name in ("digits", "sklearn"):
        return DigitsJEPADataset(**kwargs)
    if name in ("cifar", "cifar10", "cifar-10"):
        return CIFARJEPADataset(**kwargs)
    raise ValueError(f"Unknown dataset: {name!r}")
