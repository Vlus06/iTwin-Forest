"""TreeFinder tiles, the evaluation scenarios and their splits.

Tiles are 224x224 GeoTIFFs with bands R, G, B, NIR and a label band (0 background, 1 dead tree, 255 no data). The network input has five
channels R, G, B, NIR, NDVI, all in [0, 1]; pixels with label 255 are excluded from losses, statistics and metrics.

Scenarios
  random  tile-level random split (split seed fixed to 2025, as in the paper)
  co      train on Colorado only, test on all other states
  ew      train on the eastern states, test on the western states
  we      train on the western states, test on the eastern states
  site    site-disjoint split: all tiles cut from one source image fall in the same partition (partition seed 2025)
For co / ew / we the validation set is a random 10% of the training tiles (it depends on the training seed).
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WEST = "AR AZ CA CO IA ID KS LA MN MO MT ND NE NM NV OK OR SD TX UT WA WY".split()
EAST = "AL CT DE FL GA IL IN KY MA MD ME MI MS NC NH NJ NY OH PA RI SC TN VA VT WV WI".split()
SCENARIOS = ("random", "co", "ew", "we", "site")
INFO_CSV = "tile_info224_v3.csv"
TILE_DIR = "tiles224_v3"
NO_DATA = 255


def read_info(data_root):
    return pd.read_csv(os.path.join(data_root, INFO_CSV))


def _random_partition(info, seed, test_ratio=0.2, trainval_ratio=0.8, val_ratio=0.1):
    tiles = info["FileName"].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(tiles)
    n_test = int(test_ratio * len(tiles))
    n_trainval = int(trainval_ratio * len(tiles))
    test = tiles[:n_test]
    trainval = info[info["FileName"].isin(tiles[n_test:n_test + n_trainval])]["FileName"].unique().tolist()
    rng.shuffle(trainval)
    n_val = int(val_ratio * len(trainval))
    return {"train": trainval[n_val:], "val": trainval[:n_val], "test": test}


def _state_partition(info, train_states, seed):
    from sklearn.model_selection import train_test_split
    train_all = info[info["State"].isin(train_states)].reset_index(drop=True)
    train, val = train_test_split(train_all, test_size=0.1, random_state=seed)
    test = info[~info["State"].isin(train_states)]
    return {"train": train["FileName"].tolist(), "val": val["FileName"].tolist(), "test": test["FileName"].tolist()}


def site_partition(info, seed=2025, test_frac=0.2, val_frac=0.1):
    """Site-disjoint split (about 70/10/20 of the sites), stratified by state and by whether the site contains any dead tree."""
    sites = info.groupby("ImageRawPath", as_index=False).agg(State=("State", "first"), dead_pixels=("LabelSize", "sum"))
    sites["has_dead"] = sites["dead_pixels"].gt(0)
    rng = np.random.default_rng(seed)
    assignment = {}
    for _, group in sites.groupby(["State", "has_dead"], sort=True):
        names = group["ImageRawPath"].to_numpy().copy()
        rng.shuffle(names)
        n = len(names)
        n_test, n_val = round(test_frac * n), round(val_frac * n)
        if n >= 3:
            n_test, n_val = max(1, n_test), max(1, n_val)
        if n_test + n_val >= n:
            n_val = max(0, n - n_test - 1)
        for p in names[:n_test]:
            assignment[p] = "test"
        for p in names[n_test:n_test + n_val]:
            assignment[p] = "val"
        for p in names[n_test + n_val:]:
            assignment[p] = "train"
    split = info["ImageRawPath"].map(assignment)
    return {s: info.loc[split == s, "FileName"].tolist() for s in ("train", "val", "test")}


def partition(info, scenario, seed=2025, split_seed=None):
    """Return {'train': [...], 'val': [...], 'test': [...]} lists of tile file names.

    `seed` is the training seed; `split_seed` overrides the seed that selects the split (random and site always use 2025 by default,
    so that only the training noise changes between seeds)."""
    if scenario == "random":
        return _random_partition(info, 2025 if split_seed is None else split_seed)
    if scenario == "site":
        return site_partition(info, 2025 if split_seed is None else split_seed)
    states = {"co": ["CO"], "ew": EAST, "we": WEST}[scenario]
    return _state_partition(info, states, seed if split_seed is None else split_seed)


def load_tile(path):
    """-> image [5,H,W] float32 in [0,1] (R,G,B,NIR,NDVI), label [H,W] uint8 in {0,1}, no_data [H,W] bool."""
    import rasterio
    with rasterio.open(path) as r:
        arr = r.read()
    img = arr[:4].astype(np.float32)
    raw = arr[4].astype(np.uint8)
    no_data = raw == NO_DATA
    label = raw.copy()
    label[no_data] = 0
    nir, red = img[3], img[0]
    ndvi = np.clip((nir - red) / (nir + red + 1e-10), -1, 1)
    img = np.concatenate([img / 255.0, ((ndvi + 1) / 2.0)[None]], 0)
    return img, label, no_data


def _geometric(img, label, no_data):
    """Random horizontal / vertical flips and a random multiple of 90 degrees, applied to image, label and mask alike."""
    if random.random() < 0.5:
        img, label, no_data = img[:, :, ::-1], label[:, ::-1], no_data[:, ::-1]
    if random.random() < 0.5:
        img, label, no_data = img[:, ::-1, :], label[::-1, :], no_data[::-1, :]
    k = random.choice([0, 1, 2, 3])
    img, label, no_data = np.rot90(img, k, axes=(1, 2)), np.rot90(label, k), np.rot90(no_data, k)
    return img.copy(), label.copy(), no_data.copy()


class TileDataset(Dataset):
    def __init__(self, data_root, files, train=False):
        self.dir = os.path.join(data_root, TILE_DIR)
        self.files, self.train = list(files), train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img, label, no_data = load_tile(os.path.join(self.dir, self.files[i]))
        if self.train:
            img, label, no_data = _geometric(img, label, no_data)
        return {"image": torch.from_numpy(img).float(), "label": torch.from_numpy(label).long(),
                "no_data_mask": torch.from_numpy(no_data).bool()}
