"""Multimodal ViT-UNet with multiple-instance pooling.

A shared, frozen DINOv2 ViT trunk feeds two modality branches (mammography and
ultrasound). Each branch owns the last transformer blocks, a UNETR-style
convolutional decoder with skip connections from intermediate ViT layers, and a
gated-attention MIL pool over the images of one study. The decoder has two
heads: a reconstruction head (masked-image pretext task) and a saliency head
whose map weights the pooling of the final ViT tokens.
"""
import copy
from dataclasses import asdict, dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("MG", "US")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    backbone: str = "vit_small_patch14_reg4_dinov2.lvd142m"
    n_finetune_blocks: int = 2
    skip_layers: tuple = (4, 8)             # 1-based block indices feeding decoder skips
    use_decoder: bool = True
    decoder_channels: tuple = (256, 128, 64)
    embed_dim: int = 256
    dropout: float = 0.3
    pretrained: bool = True

    def to_dict(self):
        return asdict(self)


def _gn(ch):
    return nn.GroupNorm(8, ch)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True),
        )


class Up2(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.ConvTranspose2d(cin, cout, 2, stride=2, bias=False), _gn(cout), nn.ReLU(inplace=True))


class UNetDecoder(nn.Module):
    """Expanding path over ViT token maps (all at 1/patch resolution).

    Skips from shallower transformer layers are upsampled with transposed
    convolutions before concatenation, as in UNETR. Three levels: the token
    grid, 2x and 4x; the output is 4x the token grid.
    """

    SCALE = 4

    def __init__(self, dim, ch=(256, 128, 64)):
        super().__init__()
        c0, c1, c2 = ch
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.bottleneck = nn.Sequential(nn.Conv2d(dim, c0, 1, bias=False), _gn(c0), nn.ReLU(inplace=True))
        self.skip2 = Up2(dim, c1)
        self.skip1 = nn.Sequential(Up2(dim, c1), Up2(c1, c2))
        self.up2, self.conv2 = Up2(c0, c1), ConvBlock(2 * c1, c1)
        self.up1, self.conv1 = Up2(c1, c2), ConvBlock(2 * c2, c2)
        self.rec_head = nn.Conv2d(c2, 1, 1)
        self.sal_head = nn.Conv2d(c2, 1, 1)
        nn.init.zeros_(self.sal_head.weight)
        nn.init.zeros_(self.sal_head.bias)   # start as uniform (mean) pooling

    def forward(self, feats, hw):
        h, w = hw
        z1, z2, z3 = [n(f).transpose(1, 2).reshape(f.shape[0], -1, h, w) for n, f in zip(self.norms, feats)]
        d = self.bottleneck(z3)
        d = self.conv2(torch.cat([self.up2(d), self.skip2(z2)], 1))
        d = self.conv1(torch.cat([self.up1(d), self.skip1(z1)], 1))
        return self.rec_head(d), self.sal_head(d)


class GatedAttention(nn.Module):
    """Gated-attention MIL scores (Ilse et al., 2018)."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)

    def forward(self, h):
        return self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)


def bag_softmax(scores, bag, n_bags):
    mx = torch.full((n_bags,), -1e4, device=scores.device, dtype=scores.dtype)
    mx = mx.scatter_reduce(0, bag, scores, reduce="amax", include_self=True)
    e = torch.exp(scores - mx[bag])
    den = torch.zeros(n_bags, device=scores.device, dtype=scores.dtype).index_add_(0, bag, e)
    return e / den[bag]


class Branch(nn.Module):
    def __init__(self, vit, cfg):
        super().__init__()
        dim = vit.embed_dim
        n_ft = cfg.n_finetune_blocks
        self.blocks = nn.ModuleList(copy.deepcopy(b) for b in vit.blocks[len(vit.blocks) - n_ft:]) if n_ft else nn.ModuleList()
        self.norm = copy.deepcopy(vit.norm)
        self.norm.requires_grad_(n_ft > 0)
        self.blocks.requires_grad_(True)
        self.decoder = UNetDecoder(dim, cfg.decoder_channels) if cfg.use_decoder else None
        self.proj = nn.Sequential(nn.Linear(2 * dim, cfg.embed_dim), nn.LayerNorm(cfg.embed_dim), nn.GELU(),
                                  nn.Dropout(cfg.dropout))
        self.att = GatedAttention(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, 1)


class ViTUNetMIL(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        vit = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0, dynamic_img_size=True)
        self.patch = vit.patch_embed.patch_size[0]
        self.n_prefix = vit.num_prefix_tokens
        self.dim = vit.embed_dim
        self.branches = nn.ModuleDict({m: Branch(vit, cfg) for m in MODALITIES})
        self.n_frozen = len(vit.blocks) - cfg.n_finetune_blocks
        del vit.blocks[self.n_frozen:]
        vit.requires_grad_(False)
        self.trunk = vit
        self.fusion = nn.Sequential(nn.Linear(2 * cfg.embed_dim, 128), nn.GELU(), nn.Dropout(cfg.dropout),
                                    nn.Linear(128, 1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.trunk.eval()
        return self

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("trunk.")}

    def load_trainable_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("trunk.")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    def param_groups(self, lr, lr_backbone, weight_decay):
        bb, new = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_bb = ".blocks." in name or (name.startswith("branches.") and ".norm." in name)
            (bb if is_bb else new).append(p)
        groups = [{"params": new, "lr": lr, "weight_decay": weight_decay}]
        if bb:
            groups.append({"params": bb, "lr": lr_backbone, "weight_decay": weight_decay})
        return groups

    def encode_images(self, x, modality):
        """x: (N,1,H,W) in [0,1]. Returns per-image embedding plus dense outputs."""
        br = self.branches[modality]
        N, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        x3 = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        skips = []
        with torch.no_grad():
            t = self.trunk.patch_embed(x3)
            t = self.trunk._pos_embed(t)
            t = self.trunk.norm_pre(t)
            for i, blk in enumerate(self.trunk.blocks, 1):
                t = blk(t)
                if i in self.cfg.skip_layers:
                    skips.append(t[:, self.n_prefix:])
        for blk in br.blocks:
            t = blk(t)
        t = br.norm(t)
        cls, tok = t[:, 0], t[:, self.n_prefix:]
        out = {"hw": (h, w)}
        dec = br.decoder(skips + [tok], (h, w)) if br.decoder is not None else None
        # pooling and heads run in full precision
        with torch.autocast(device_type=x.device.type, enabled=False):
            cls, tok = cls.float(), tok.float()
            if dec is not None:
                a = torch.sigmoid(dec[1].float())
                wgt = F.adaptive_avg_pool2d(a, (h, w)).flatten(1)
                pooled = (wgt.unsqueeze(-1) * tok).sum(1) / (wgt.sum(1, keepdim=True) + 1e-6)
                out.update(rec=torch.sigmoid(dec[0].float()), sal=a)
            else:
                pooled = tok.mean(1)
            out["emb"] = br.proj(torch.cat([cls, pooled], 1))
        return out

    def pool_bags(self, emb, bag, n_bags, modality):
        br = self.branches[modality]
        att = bag_softmax(br.att(emb), bag, n_bags)
        pooled = torch.zeros(n_bags, emb.shape[1], device=emb.device, dtype=emb.dtype).index_add_(0, bag, att.unsqueeze(-1) * emb)
        return pooled, att, br.head(pooled).squeeze(-1)

    def forward(self, images, bags, n_bags):
        """images/bags: dicts keyed by modality; bags map each image to a study index.

        A modality may be absent (inference with a single modality): its
        unimodal logit is then the only prediction returned for that study set.
        """
        out, pooled = {}, {}
        for m in MODALITIES:
            if m not in images or images[m] is None or images[m].shape[0] == 0:
                continue
            enc = self.encode_images(images[m], m)
            with torch.autocast(device_type=images[m].device.type, enabled=False):
                p, att, logit = self.pool_bags(enc["emb"], bags[m], n_bags, m)
            pooled[m] = p
            out[m] = {**enc, "att": att, "logit": logit}
        if len(pooled) == len(MODALITIES):
            with torch.autocast(device_type=pooled["MG"].device.type, enabled=False):
                out["fused_logit"] = self.fusion(torch.cat([pooled[m] for m in MODALITIES], 1)).squeeze(-1)
        return out
