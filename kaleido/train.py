"""Train and evaluate Kaleido (or its ablations) on one TreeFinder scenario.

    python -m kaleido.train --data_root datasets --scenario co --seed 2025                       # Kaleido
    python -m kaleido.train --data_root datasets --scenario co --seed 2025 --no_bank --cons 0    # DINOv2 + decoder only
    python -m kaleido.train --data_root datasets --scenario co --seed 2025 --cons 0              # + statistics bank, no consistency loss

Recipe: AdamW (weight decay 0.01), decoder lr 1e-4 and encoder lr = encoder_mult x 1e-4, ExponentialLR(0.95) per epoch, batch size 32,
at most 100 epochs, early stopping (patience 10) on the validation mask loss (BCE + Dice), bf16 autocast on GPU. The checkpoint with the lowest
validation mask loss is evaluated once on the test split in fp32 at threshold 0.5.
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SCENARIOS, TILE_DIR, TileDataset, partition, read_info
from .losses import consistency_loss, mask_loss
from .metrics import Pooled
from .model import Dinov2Seg, param_groups
from .stats_bank import compute_band_stats, load_bank

log = logging.getLogger("kaleido")


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, help="folder with tile_info224_v3.csv and tiles224_v3/")
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=2025, help="training seed")
    ap.add_argument("--split_seed", type=int, default=None, help="override the seed that selects the split")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--tag", default="kaleido")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder_mult", type=float, default=0.01, help="encoder lr = encoder_mult x lr (the paper reports 0.1 and 0.01)")
    ap.add_argument("--no_bank", action="store_true", help="no statistics-bank restyling")
    ap.add_argument("--cons", type=float, default=1.0, help="weight lambda of the consistency loss (0 disables it)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bank_csv", default=None, help="per-tile band statistics (default: <data_root>/tile_band_stats.csv, built if missing)")
    ap.add_argument("--max_tiles", type=int, default=0, help="debug: keep this many tiles per split")
    return ap.parse_args(argv)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def batch_to(b, device):
    x = b["image"].to(device, non_blocking=True)
    y = b["label"].unsqueeze(1).float().to(device)
    valid = (~b["no_data_mask"].unsqueeze(1)).float().to(device)
    return x, y, valid


@torch.no_grad()
def val_loss(model, loader, device):
    model.eval()
    tot = 0.0
    for b in loader:
        x, y, valid = batch_to(b, device)
        with autocast(device):
            o = model(x)
        tot += mask_loss(o.float(), y, valid).item()
    return tot / max(1, len(loader))


@torch.no_grad()
def test_metrics(model, loader, device):
    model.eval()
    m = Pooled()
    for b in loader:
        x, y, valid = batch_to(b, device)
        m.update(model(x).float(), y, valid)               # fp32
    return m.result()


def main(argv=None):
    a = get_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(a.seed)
    import random
    import numpy as np
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = read_info(a.data_root)
    parts = partition(info, a.scenario, a.seed, a.split_seed)
    if a.max_tiles:
        parts = {k: v[:a.max_tiles] for k, v in parts.items()}
    log.info("scenario %s: %s tiles", a.scenario, {k: len(v) for k, v in parts.items()})

    def loader(files, train):
        return DataLoader(TileDataset(a.data_root, files, train), batch_size=a.bs, shuffle=train, num_workers=a.workers, pin_memory=True,
                          persistent_workers=a.workers > 0, prefetch_factor=2 if a.workers > 0 else None)
    train_loader, val_loader, test_loader = loader(parts["train"], True), loader(parts["val"], False), loader(parts["test"], False)

    bank = None
    if not a.no_bank:
        csv = a.bank_csv or os.path.join(a.data_root, "tile_band_stats.csv")
        if not os.path.exists(csv):
            log.info("building %s", csv)
            compute_band_stats(os.path.join(a.data_root, TILE_DIR), info["FileName"].tolist()).to_csv(csv, index=False)
        bank = load_bank(csv, parts["train"], device)
        log.info("statistics bank: %d training tiles", len(bank))

    model = Dinov2Seg(in_channels=5).to(device)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.encoder_mult), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    use_cons = bank is not None and a.cons > 0

    name = f"{a.scenario}_{a.tag}_s{a.seed}"
    out = Path(a.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    best, bad, hist, t0 = float("inf"), 0, [], time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for b in train_loader:
            x, y, valid = batch_to(b, device)
            x_in = bank.restyle(x) if bank is not None else x
            opt.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(x_in)
            logits = logits.float()
            loss = mask_loss(logits, y, valid)
            if use_cons:                                    # reference: the clean tile, no gradient
                with torch.no_grad(), autocast(device):
                    p_clean = torch.sigmoid(model(x).float())
                loss = loss + a.cons * consistency_loss(torch.sigmoid(logits), p_clean, valid)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        vl = val_loss(model, val_loader, device)
        hist.append({"epoch": epoch, "train_loss": tot / max(1, nb), "val_mask_loss": vl, "minutes": (time.time() - t0) / 60})
        log.info("epoch %d train %.4f val_mask %.4f", epoch, hist[-1]["train_loss"], vl)
        if vl < best:
            best, bad = vl, 0
            torch.save(model.state_dict(), out / "best.pth")
        else:
            bad += 1
            if bad >= a.patience:
                log.info("early stopping at epoch %d", epoch)
                break

    model.load_state_dict(torch.load(out / "best.pth", map_location=device))
    res = test_metrics(model, test_loader, device)
    (out / "metrics.json").write_text(json.dumps({"args": vars(a), "best_val_mask_loss": best, "epochs": len(hist),
                                                  "metrics": res, "history": hist}, indent=2))
    log.info("test dead-tree F1 %.2f  P %.2f  R %.2f  IoU %.2f", *(100 * res["dead"][k] for k in ("f1", "precision", "recall", "iou")))


if __name__ == "__main__":
    main()
