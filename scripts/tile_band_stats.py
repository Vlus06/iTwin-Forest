"""Pre-compute the statistics bank table (per-tile mean/std of R,G,B,NIR) for every tile. Labels are never read.

    python scripts/tile_band_stats.py --data_root datasets
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kaleido.data import TILE_DIR, read_info   # noqa: E402
from kaleido.stats_bank import compute_band_stats   # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    df = compute_band_stats(os.path.join(a.data_root, TILE_DIR), read_info(a.data_root)["FileName"].tolist(), a.procs)
    out = os.path.join(a.data_root, "tile_band_stats.csv")
    df.to_csv(out, index=False)
    print(len(df), "tiles ->", out)
