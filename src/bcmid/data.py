"""In-memory dataset, leakage-free splits and GPU augmentation."""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

from . import paths

MODALITIES = ("MG", "US")
TRAIN_HW = {"MG": (448, 224), "US": (252, 336)}
_FILES = {"MG": "mg.npy", "US": "us.npy"}


def load_tables(cache=None):
    cache = cache or paths.CACHE
    kw = dict(dtype={"patient_id": str, "group_id": str}, keep_default_na=False)
    return pd.read_csv(cache / "manifest.csv", **kw), pd.read_csv(cache / "patients.csv", **kw)


def resize_u8(arr, hw, device, chunk=128):
    """uint8 (N,H,W) numpy -> uint8 (N,h,w) tensor on device, area-averaged."""
    out = []
    for i in range(0, len(arr), chunk):
        x = torch.from_numpy(np.array(arr[i:i + chunk])).to(device).float().unsqueeze(1)
        if tuple(x.shape[-2:]) != tuple(hw):
            x = F.interpolate(x, size=hw, mode="area")
        out.append(x.squeeze(1).round().clamp(0, 255).to(torch.uint8))
    return torch.cat(out)


class BCMID:
    """All images of both modalities as uint8 tensors on the training device."""

    def __init__(self, device="cuda", hw=None, cache=None):
        cache = cache or paths.CACHE
        self.device, self.hw = device, dict(hw or TRAIN_HW)
        self.manifest, self.patients = load_tables(cache)
        self.images, self.index = {}, {}
        for m in MODALITIES:
            self.images[m] = resize_u8(np.load(cache / _FILES[m], mmap_mode="r"), self.hw[m], device)
            sub = self.manifest[self.manifest.modality == m]
            groups = sub.groupby("patient_id").idx.apply(lambda s: np.sort(s.to_numpy()))
            self.index[m] = {pid: torch.as_tensor(v, device=device) for pid, v in groups.items()}
        self.label = dict(zip(self.patients.patient_id, self.patients.label))

    def batch(self, pids, k_max=None, gen=None):
        """Stack the images of the given studies. k_max caps images per study (training)."""
        images, bags = {}, {}
        for m in MODALITIES:
            idx, bag = [], []
            for b, pid in enumerate(pids):
                ii = self.index[m][pid]
                if k_max and len(ii) > k_max[m]:
                    ii = ii[torch.randperm(len(ii), generator=gen)[:k_max[m]].to(ii.device)]
                idx.append(ii)
                bag.append(torch.full((len(ii),), b, device=self.device, dtype=torch.long))
            idx = torch.cat(idx)
            images[m] = self.images[m][idx].unsqueeze(1).float() / 255.0
            bags[m] = torch.cat(bag)
        return images, bags


def make_folds(patients, n_splits=5, seed=0, val_splits=5):
    """Stratified splits grouped by woman; each training part is further split into train/val."""
    lab = patients[patients.label >= 0].reset_index(drop=True)
    ids, y, g = lab.patient_id.to_numpy(), lab.label.to_numpy(), lab.group_id.to_numpy()
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, te) in enumerate(outer.split(ids, y, g)):
        inner = StratifiedGroupKFold(n_splits=val_splits, shuffle=True, random_state=seed + 1 + k)
        tr2, va = next(inner.split(ids[tr], y[tr], g[tr]))
        fold = dict(train=ids[tr][tr2].tolist(), val=ids[tr][va].tolist(), test=ids[te].tolist())
        gs = {name: set(g[np.isin(ids, v)]) for name, v in fold.items()}
        assert not (gs["train"] & gs["test"]) and not (gs["val"] & gs["test"]) and not (gs["train"] & gs["val"])
        folds.append(fold)
    return folds


def augment(x, modality, gen):
    """Random affine + photometric jitter on a (N,1,H,W) batch in [0,1]."""
    n, _, h, w = x.shape
    dev = x.device

    def u(lo, hi):
        return torch.rand(n, generator=gen, device=dev) * (hi - lo) + lo

    ang = u(-10, 10) * math.pi / 180
    scale = 1.0 / u(0.85, 1.15)
    tx, ty = u(-0.06, 0.06), u(-0.06, 0.06)
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    theta = torch.stack([torch.stack([cos, -sin * h / w, tx], 1), torch.stack([sin * w / h, cos, ty], 1)], 1)
    sign = ((torch.rand(n, generator=gen, device=dev) < 0.5).float() * 2 - 1).unsqueeze(1)
    # ultrasound: left-right mirror; mammography: head-foot mirror (chest wall stays left)
    theta[:, :, 0 if modality == "US" else 1] *= sign
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gamma = torch.exp(u(-0.25, 0.25)).view(n, 1, 1, 1)
    x = x.clamp(1e-4, 1).pow(gamma)
    x = (x - 0.5) * u(0.85, 1.15).view(n, 1, 1, 1) + 0.5 + u(-0.08, 0.08).view(n, 1, 1, 1)
    return x.clamp(0, 1)


def random_block_mask(n, grid_hw, ratio, block, gen, device):
    """Binary mask over the token grid (1 = hidden), drawn in block x block token cells."""
    h, w = grid_hw
    gh, gw = h // block, w // block
    k = int(round(ratio * gh * gw))
    if k <= 0:
        return torch.zeros(n, 1, h, w, device=device)
    r = torch.rand(n, gh * gw, generator=gen, device=device)
    kth = r.kthvalue(k, dim=1, keepdim=True).values
    m = (r <= kth).float().view(n, 1, gh, gw)
    return F.interpolate(m, size=(h, w), mode="nearest")


def apply_mask(x, mask, patch):
    """Replace hidden patches by the image mean intensity."""
    mpx = F.interpolate(mask, scale_factor=patch, mode="nearest")
    return x * (1 - mpx) + x.mean((2, 3), keepdim=True) * mpx
