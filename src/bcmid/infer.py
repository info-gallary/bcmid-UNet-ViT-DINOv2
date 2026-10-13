"""Inference with the cross-validation ensemble (used by the dashboard and the figures)."""
import numpy as np
import torch
import torch.nn.functional as F

from . import paths
from . import preprocess as pp
from .metrics import sigmoid
from .model import MODALITIES, ModelConfig, ViTUNetMIL


def load_fold(variant, fold, device, runs=None):
    ckpt = torch.load((runs or paths.RUNS) / variant / f"fold{fold}.pt", map_location="cpu", weights_only=False)
    model = ViTUNetMIL(ModelConfig(**{**ckpt["model_cfg"], "skip_layers": tuple(ckpt["model_cfg"]["skip_layers"]),
                                      "decoder_channels": tuple(ckpt["model_cfg"]["decoder_channels"])}))
    model.load_trainable_state_dict(ckpt["state"])
    model.to(device).eval()
    return model, ckpt


class Ensemble:
    """The fold models of one variant. Predictions are averaged in probability space."""

    def __init__(self, variant="full", folds=None, device=None, runs=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        runs = runs or paths.RUNS
        found = sorted(int(p.stem[4:]) for p in (runs / variant).glob("fold[0-9].pt"))
        self.folds = found if folds is None else [f for f in folds if f in found]
        if not self.folds:
            raise FileNotFoundError(f"no checkpoints for variant '{variant}' in {runs / variant}")
        self.members = {k: load_fold(variant, k, self.device, runs) for k in self.folds}
        first = self.members[self.folds[0]][0]
        for k in self.folds[1:]:                 # the frozen trunk is identical in every member
            self.members[k][0].trunk = first.trunk
        self.hw = self.members[self.folds[0]][1]["hw"]
        self.has_decoder = first.cfg.use_decoder

    def prepare(self, cached_u8, modality):
        """uint8 cache-resolution images (N,H,W) -> float (N,1,h,w) at model resolution."""
        x = torch.from_numpy(np.asarray(cached_u8)).to(self.device).float().unsqueeze(1)
        hw = tuple(self.hw[modality])
        if tuple(x.shape[-2:]) != hw:
            x = F.interpolate(x, size=hw, mode="area")
        return x / 255.0

    @torch.no_grad()
    def predict(self, images, folds=None):
        """images: {"MG": uint8 (N,H,W) or None, "US": ...} for ONE study.

        Returns calibrated probabilities per head and per-image saliency maps,
        MIL attention weights and reconstructions, averaged over ensemble members.
        """
        folds = self.folds if folds is None else folds
        xs = {m: self.prepare(images[m], m) for m in MODALITIES if images.get(m) is not None and len(images[m])}
        bags = {m: torch.zeros(len(x), dtype=torch.long, device=self.device) for m, x in xs.items()}
        probs = {h: [] for h in ("fused", "MG", "US")}
        thr = {h: [] for h in probs}
        dense = {m: {"sal": [], "att": [], "rec": []} for m in xs}
        for k in folds:
            model, ckpt = self.members[k]
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                out = model(xs, bags, 1)
            logits = {m: out[m]["logit"] for m in xs}
            if "fused_logit" in out:
                logits["fused"] = out["fused_logit"]
            for h, z in logits.items():
                a, b = ckpt["platt"][h]
                probs[h].append(float(sigmoid(a * float(z.float().cpu()) + b)))
                thr[h].append(ckpt["thresholds"][h])
            for m in xs:
                dense[m]["att"].append(out[m]["att"].float().cpu().numpy())
                if "sal" in out[m]:
                    dense[m]["sal"].append(out[m]["sal"].float().cpu().numpy()[:, 0])
                    dense[m]["rec"].append(out[m]["rec"].float().cpu().numpy()[:, 0])
        res = {"prob": {h: float(np.mean(v)) for h, v in probs.items() if v},
               "prob_members": {h: v for h, v in probs.items() if v},
               "threshold": {h: float(np.mean(v)) for h, v in thr.items() if v},
               "folds": list(folds)}
        for m in xs:
            res[m] = {key: np.mean(v, 0) for key, v in dense[m].items() if v}
            res[m]["input"] = xs[m].cpu().numpy()[:, 0]
        return res


def preprocess_upload(data, modality):
    """Raw image bytes (as exported by the scanner) -> cache-resolution uint8 image."""
    return pp.preprocess_bytes(data, modality)[0]
