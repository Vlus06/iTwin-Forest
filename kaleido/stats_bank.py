"""Spectral statistics bank and restyling.

For every training tile we store eight numbers: the mean and standard deviation of R, G, B and NIR over its valid pixels (reflectance
scale 0-1). A tile is restyled by moving its own per-band statistics towards those of a tile drawn from the bank,
    mu_t = a * mu_b + (1 - a) * mu,   sigma_t = a * sigma_b + (1 - a) * sigma,   a ~ U(0, 1) per tile,
    x'_c = (x_c - mu_c) / (sigma_c + 1e-4) * sigma_t,c + mu_t,c ,
NDVI is recomputed from the new R and NIR and no-data pixels stay 0. Only statistics of source-domain training tiles enter the bank.
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

COLS = [f"m{i}" for i in range(4)] + [f"s{i}" for i in range(4)]


def valid_mask(x):
    """[B,1,H,W]: 1 where any of the four bands is non-zero (no-data pixels are zero in all bands)."""
    return (x[:, :4].sum(1, keepdim=True) > 0).float()


def with_ndvi(bands, valid):
    """bands [B,4,H,W] -> [B,5,H,W]: clamp to [0,1], zero the invalid pixels and append the recomputed NDVI (rescaled to [0,1])."""
    bands = bands.clamp(0, 1) * valid
    ndvi = (bands[:, 3] - bands[:, 0]) / (bands[:, 3] + bands[:, 0] + 1e-10)
    ndvi = ((ndvi.clamp(-1, 1) + 1) / 2).unsqueeze(1) * valid
    return torch.cat([bands, ndvi], 1)


def tile_stats(x, valid):
    """Per-tile, per-band mean and standard deviation over the valid pixels. x: [B,4,H,W]."""
    n = valid.sum((2, 3), keepdim=True).clamp_min(1)
    mu = (x * valid).sum((2, 3), keepdim=True) / n
    sd = ((((x - mu) ** 2) * valid).sum((2, 3), keepdim=True) / n).sqrt()
    return mu, sd


class StatBank:
    def __init__(self, table, device="cpu"):
        self.t = torch.as_tensor(np.asarray(table), dtype=torch.float32, device=device)     # [N, 8]: 4 means then 4 standard deviations

    def __len__(self):
        return len(self.t)

    def restyle(self, x):
        """x: [B,5,H,W] (R,G,B,NIR,NDVI) -> restyled batch with the same shape."""
        b, dev = x.shape[0], x.device
        valid = valid_mask(x)
        v = x[:, :4]
        mu, sd = tile_stats(v, valid)
        pick = self.t[torch.randint(0, len(self.t), (b,), device=dev)]
        mb, sb = pick[:, :4].reshape(b, 4, 1, 1), pick[:, 4:].reshape(b, 4, 1, 1)
        a = torch.rand(b, 1, 1, 1, device=dev)
        mt, st = a * mb + (1 - a) * mu, a * sb + (1 - a) * sd
        return with_ndvi((v - mu) / (sd + 1e-4) * st + mt, valid)


def _one(args):
    path, name = args
    import rasterio
    with rasterio.open(path) as r:
        a = r.read([1, 2, 3, 4]).astype(np.float32) / 255.0      # labels are never read
    valid = a.sum(0) > 0
    if valid.sum() < 50:
        return None
    v = a[:, valid]
    return [name] + v.mean(1).tolist() + v.std(1).tolist()


def compute_band_stats(tiles_dir, files, procs=6):
    """DataFrame with FileName and the eight statistics of every tile in `files` (tiles with < 50 valid pixels are skipped)."""
    jobs = [(os.path.join(tiles_dir, f), f) for f in files]
    with Pool(procs) as p:
        rows = [r for r in p.map(_one, jobs, chunksize=64) if r is not None]
    return pd.DataFrame(rows, columns=["FileName"] + COLS)


def load_bank(csv_path, train_files, device="cpu"):
    """Bank restricted to the training tiles of the current scenario."""
    df = pd.read_csv(csv_path)
    df = df[df["FileName"].isin(set(train_files))]
    return StatBank(df[COLS].values, device)
