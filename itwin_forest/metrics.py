"""Pooled pixel metrics at threshold 0.5. TP/FP/FN/TN are summed over all valid pixels of the test set, per class."""
import torch


class Pooled:
    def __init__(self):
        self.c = {k: torch.zeros(4, dtype=torch.float64) for k in (0, 1)}      # tp, fp, fn, tn

    @torch.no_grad()
    def update(self, logits, y, valid):
        pred, valid = (logits > 0).cpu(), valid.cpu().bool()
        y = y.cpu()
        for k in (0, 1):
            p, t = (pred == k) & valid, (y == k) & valid
            tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t & valid).sum()
            tn = (~p & ~t & valid).sum()
            self.c[k] += torch.tensor([tp, fp, fn, tn], dtype=torch.float64)

    def result(self):
        out = {}
        for k in (0, 1):
            tp, fp, fn, _ = self.c[k].tolist()
            pr, rc = tp / (tp + fp + 1e-6), tp / (tp + fn + 1e-6)
            out[f"class{k}"] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc + 1e-6), "iou": tp / (tp + fp + fn + 1e-6)}
        out["dead"] = out["class1"]                                            # dead-tree class
        out["all"] = {m: 0.5 * (out["class0"][m] + out["class1"][m]) for m in ("precision", "recall", "f1", "iou")}   # macro over both classes
        return out
