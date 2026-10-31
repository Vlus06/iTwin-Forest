import numpy as np
import pandas as pd
import torch

from itwin_forest.data import EAST, WEST, partition
from itwin_forest.losses import consistency_loss, mask_loss
from itwin_forest.stats_bank import StatBank, tile_stats, valid_mask


def fake_info(n_sites=60, tiles=5):
    rows = []
    states = ["CO", "UT", "NY", "FL"]
    for s in range(n_sites):
        st = states[s % 4]
        for t in range(tiles):
            rows.append({"FileName": f"{st}{s:03d}_{t}.tif", "State": st, "ImageRawPath": f"img{s}.tif", "LabelSize": (s * t) % 3 * 10})
    return pd.DataFrame(rows)


def test_state_lists_cover_disjoint_states():
    assert not set(EAST) & set(WEST) and len(EAST) + len(WEST) == 48


def test_partitions_are_disjoint_and_complete():
    info = fake_info()
    for sc in ("random", "co", "site"):
        p = partition(info, sc, seed=1)
        sets = [set(p[k]) for k in ("train", "val", "test")]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    p = partition(info, "co", 1)
    assert all(f.startswith("CO") for f in p["train"] + p["val"]) and not any(f.startswith("CO") for f in p["test"])


def test_site_partition_has_no_site_in_two_splits():
    info = fake_info()
    p = partition(info, "site")
    site = dict(zip(info.FileName, info.ImageRawPath))
    s = [{site[f] for f in p[k]} for k in ("train", "val", "test")]
    assert not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2])


def test_restyle_identity_when_bank_is_the_tile_itself():
    torch.manual_seed(0)
    x = torch.rand(4, 5, 32, 32) * 0.6 + 0.1
    x[:, :, :4, :4] = 0                                                # no-data corner
    valid = valid_mask(x)
    mu, sd = tile_stats(x[:, :4], valid)
    bank = StatBank(torch.cat([mu.flatten(1), sd.flatten(1)], 1)[:1].repeat(3, 1))   # all entries identical to tile 0
    y = bank.restyle(x)
    assert y.shape == x.shape
    assert torch.all(y[:, :, :4, :4] == 0)                              # no-data stays 0
    assert torch.all((y >= 0) & (y <= 1))


def test_restyle_moves_statistics_towards_bank():
    torch.manual_seed(0)
    x = torch.rand(256, 5, 16, 16) * 0.2 + 0.4
    bank = StatBank(torch.tensor([[0.2] * 4 + [0.05] * 4]))
    y = bank.restyle(x)
    assert y[:, :4].mean() < x[:, :4].mean()


def test_losses():
    torch.manual_seed(0)
    y = (torch.rand(2, 1, 8, 8) > 0.8).float()
    valid = torch.ones_like(y)
    perfect = (y * 2 - 1) * 20
    # Dice without the factor 2: the loss of a perfect prediction is 0.5 (BCE ~ 0 plus Dice 1 - S/(2S))
    assert abs(mask_loss(perfect, y, valid).item() - 0.5) < 1e-2
    assert mask_loss(-perfect, y, valid) > 1
    p = torch.rand(2, 1, 8, 8, requires_grad=True)
    q = torch.rand(2, 1, 8, 8, requires_grad=True)
    consistency_loss(p, q, valid).backward()
    assert p.grad is not None and q.grad is None                       # stop-gradient on the reference
