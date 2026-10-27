"""DINOv2 ViT-B/14 with a light multi-level convolutional decoder.

The 5-channel tile (R,G,B,NIR,NDVI) is upsampled x2 (224 -> 448), so that a ~9 px dead crown spans more than one 14 px patch. The
ImageNet mean/std are applied to RGB only. The patch embedding is widened to five channels (the two extra filters start as the mean of the
RGB filters). Hidden states of layers 3, 6, 9, 12 (32x32 token grids) go through 1x1 conv-BN-ReLU (768 -> 128), are concatenated (512), fused
by a 3x3 conv (192) and upsampled twice (x2, conv to 96 and 48 channels); a 1x1 conv gives the logit map, resized to the input size.
About 1.5 M parameters outside the encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation


def widen_first_conv(conv, in_ch):
    if conv.in_channels == in_ch:
        return conv
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3].copy_(conv.weight[:, :3])
        new.weight[:, 3:].copy_(conv.weight[:, :3].mean(1, keepdim=True).repeat(1, in_ch - 3, 1, 1))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class Dinov2Seg(nn.Module):
    def __init__(self, in_channels=5, pretrained=True, name="facebook/dinov2-base", layers=(3, 6, 9, 12), up=2, config=None):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        self.up, self.layers = up, tuple(layers)
        if config is not None:                                   # custom (e.g. tiny) config, for tests
            self.vit = Dinov2Model(config)
        elif pretrained:
            self.vit = Dinov2Model.from_pretrained(name)
        else:
            self.vit = Dinov2Model(Dinov2Config.from_pretrained(name))
        pe = self.vit.embeddings.patch_embeddings
        pe.projection = widen_first_conv(pe.projection, in_channels)
        pe.num_channels = in_channels
        d = self.vit.config.hidden_size
        self.proj = nn.ModuleList([Conv2dNormActivation(d, 128, kernel_size=1) for _ in range(4)])
        self.fuse = Conv2dNormActivation(512, 192, kernel_size=3)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 Conv2dNormActivation(192, 96, kernel_size=3))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 Conv2dNormActivation(96, 48, kernel_size=3))
        self.head = nn.Conv2d(48, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        size = x.shape[2:]
        x = torch.cat([(x[:, :3] - self.mean) / self.std, x[:, 3:]], 1)
        x = F.interpolate(x, scale_factor=self.up, mode="bilinear", align_corners=False)
        h, w = x.shape[2] // 14, x.shape[3] // 14
        x = x[:, :, : h * 14, : w * 14]
        hs = self.vit(pixel_values=x, output_hidden_states=True).hidden_states
        feats = []
        for proj, i in zip(self.proj, self.layers):
            t = hs[i][:, 1:]                                         # drop the CLS token
            feats.append(proj(t.transpose(1, 2).reshape(t.shape[0], -1, h, w)))
        y = self.up2(self.up1(self.fuse(torch.cat(feats, 1))))
        return F.interpolate(self.head(y), size=size, mode="bilinear", align_corners=False)


def param_groups(model, lr, encoder_mult):
    """Encoder (DINOv2) at lr * encoder_mult, everything else at lr."""
    enc = [p for n, p in model.named_parameters() if n.startswith("vit.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("vit.")]
    return [{"params": enc, "lr": lr * encoder_mult}, {"params": rest, "lr": lr}]
