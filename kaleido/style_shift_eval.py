"""Robustness to controlled spectral shifts, without retraining: a trained checkpoint is evaluated on the test split after the four bands of
every test tile are perturbed (brightness, gamma, colour cast, contrast, NIR gain; NDVI recomputed). Reports the pooled dead-tree F1 per shift.

    python -m kaleido.style_shift_eval --data_root datasets --scenario co --ckpt runs/co_kaleido_s2025/best.pth --out shift.json
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import SCENARIOS, TileDataset, partition, read_info
from .model import Dinov2Seg
from .stats_bank import tile_stats, valid_mask, with_ndvi

SHIFTS = ["identity", "bright0.70", "bright0.85", "bright1.15", "bright1.30", "gamma0.70", "gamma1.40", "castRB+", "castRB-",
          "contrast0.60", "contrast1.40", "nir0.70", "nir1.30"]


def shift(x, name):
    if name == "identity":
        return x
    valid = valid_mask(x)
    v = x[:, :4].clone()
    if name.startswith("bright"):
        v = v * float(name[6:])
    elif name.startswith("gamma"):
        v = v.clamp(0, 1) ** float(name[5:])
    elif name == "castRB+":
        v[:, 0] *= 1.2
        v[:, 2] *= 0.8
    elif name == "castRB-":
        v[:, 0] *= 0.8
        v[:, 2] *= 1.2
    elif name.startswith("contrast"):
        mu, _ = tile_stats(v, valid)
        v = (v - mu) * float(name[8:]) + mu
    elif name.startswith("nir"):
        v[:, 3] *= float(name[3:])
    return with_ndvi(v, valid)


@torch.no_grad()
def evaluate(model, loader, device):
    cnt = {s: [0, 0, 0] for s in SHIFTS}                      # tp, fp, fn of the dead-tree class
    for b in loader:
        x0 = b["image"].to(device)
        y = b["label"].to(device)
        valid = ~b["no_data_mask"].to(device)
        pos, neg = (y == 1) & valid, (y != 1) & valid
        for s in SHIFTS:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                o = model(shift(x0, s))
            p = o.float().squeeze(1) > 0
            c = cnt[s]
            c[0] += int((p & pos).sum())
            c[1] += int((p & neg).sum())
            c[2] += int((~p & pos).sum())
    return {s: 2 * c[0] / max(1, 2 * c[0] + c[1] + c[2]) for s, c in cnt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=2025, help="training seed of the checkpoint (selects the same split)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = partition(read_info(a.data_root), a.scenario, a.seed)["test"]
    loader = DataLoader(TileDataset(a.data_root, files), batch_size=32, shuffle=False, num_workers=a.workers)
    model = Dinov2Seg(5, pretrained=False).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    res = evaluate(model, loader, device)
    for s, v in res.items():
        print(f"{s:14s} F1 {100 * v:.2f}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
