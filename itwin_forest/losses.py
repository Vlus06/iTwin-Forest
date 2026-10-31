import torch
import torch.nn.functional as F


def mask_loss(logits, y, valid, eps=1e-6):
    """BCE + Dice over valid pixels. Dice is pooled over the batch and has no factor 2 in the numerator:
    1 - (sum(p*y) + eps) / (sum(p) + sum(y) + eps)."""
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * valid).sum() / valid.sum()
    p = torch.sigmoid(logits) * valid
    dice = 1 - ((p * y).sum() + eps) / (p.sum() + y.sum() + eps)
    return bce + dice


def consistency_loss(p_restyled, p_clean, valid):
    """Mean squared difference of the probabilities over valid pixels; `p_clean` is the stop-gradient reference."""
    return (((p_restyled - p_clean.detach()) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)
