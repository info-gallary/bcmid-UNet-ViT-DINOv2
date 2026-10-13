"""Cross-validated training of the ViT-UNet MIL model and its ablations.

Protocol (fixed before any test fold was evaluated):
  * 5-fold stratified cross-validation grouped by woman;
  * within each training part, 20% of the studies form a validation set used
    for epoch selection, Platt calibration and the operating threshold;
  * the test fold is touched once, with the selected epoch.
"""
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import paths
from .data import BCMID, MODALITIES, apply_mask, augment, make_folds, random_block_mask
from .metrics import fit_platt, sigmoid, youden_threshold
from .model import ModelConfig, ViTUNetMIL

HEADS = ("fused", "MG", "US")


@dataclass
class TrainConfig:
    variant: str = "full"
    epochs: int = 24
    batch_studies: int = 8
    k_mg: int = 4
    k_us: int = 4
    lr: float = 5e-4
    lr_backbone: float = 2e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    lambda_rec: float = 1.0
    lambda_uni: float = 0.5
    mask_ratio: float = 0.3
    mask_block: int = 2
    n_splits: int = 5
    seed: int = 0


# variant -> (model overrides, training overrides)
VARIANTS = {
    "full": ({}, {}),
    "no_recon": ({}, {"lambda_rec": 0.0, "mask_ratio": 0.0}),
    "no_decoder": ({"use_decoder": False}, {"lambda_rec": 0.0, "mask_ratio": 0.0}),
    "frozen": ({"n_finetune_blocks": 0}, {}),
}


def configs(variant, **overrides):
    m_over, t_over = VARIANTS[variant]
    return ModelConfig(**m_over), TrainConfig(variant=variant, **{**t_over, **overrides})


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def predict(model, ds, pids, batch=8):
    """Study-level logits of the fused head and both unimodal heads (all images, no augmentation)."""
    model.eval()
    out = {h: [] for h in HEADS}
    for i in range(0, len(pids), batch):
        chunk = pids[i:i + batch]
        images, bags = ds.batch(chunk)
        with _autocast(torch.device(ds.device)):
            o = model(images, bags, len(chunk))
        out["fused"].append(o["fused_logit"].float().cpu())
        for m in MODALITIES:
            out[m].append(o[m]["logit"].float().cpu())
    return {h: torch.cat(v).numpy().astype(np.float64) for h, v in out.items()}


def train_fold(ds, fold, k, mcfg, tcfg, unlabeled=(), log=print):
    dev = torch.device(ds.device)
    seed = tcfg.seed * 1000 + k
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen_cpu = torch.Generator().manual_seed(seed)
    gen_dev = torch.Generator(device=dev).manual_seed(seed)

    model = ViTUNetMIL(mcfg).to(dev)
    opt = torch.optim.AdamW(model.param_groups(tcfg.lr, tcfg.lr_backbone, tcfg.weight_decay))
    train, B = fold["train"], tcfg.batch_studies
    steps = math.ceil(len(train) / B)
    total, warm = tcfg.epochs * steps, max(1, tcfg.warmup_epochs * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")

    y_tr = np.array([ds.label[p] for p in train])
    pos_weight = torch.tensor(float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), device=dev)
    y_val = np.array([ds.label[p] for p in fold["val"]])
    k_max = {"MG": tcfg.k_mg, "US": tcfg.k_us}
    use_rec = tcfg.lambda_rec > 0 and mcfg.use_decoder

    def bce(logit, y):
        return F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_weight)

    best_auc, best_state, best_epoch, history = -1.0, None, -1, []
    for ep in range(tcfg.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train), generator=gen_cpu).tolist()
        agg = {"cls": 0.0, "rec": 0.0}
        for s in range(0, len(perm), B):
            pids = [train[j] for j in perm[s:s + B]]
            images, bags = ds.batch(pids, k_max, gen_cpu)
            y = torch.tensor([ds.label[p] for p in pids], device=dev, dtype=torch.float32)
            clean, masks = {}, {}
            for m in MODALITIES:
                x = augment(images[m], m, gen_dev)
                clean[m] = x
                if tcfg.mask_ratio > 0:
                    grid = (x.shape[-2] // model.patch, x.shape[-1] // model.patch)
                    masks[m] = random_block_mask(x.shape[0], grid, tcfg.mask_ratio, tcfg.mask_block, gen_dev, dev)
                    x = apply_mask(x, masks[m], model.patch)
                images[m] = x
            with _autocast(dev):
                out = model(images, bags, len(pids))
            loss_cls = bce(out["fused_logit"], y) + tcfg.lambda_uni * sum(bce(out[m]["logit"], y) for m in MODALITIES)
            loss_rec = torch.zeros((), device=dev)
            if use_rec:
                for m in MODALITIES:
                    rec = out[m]["rec"]
                    tgt = F.interpolate(clean[m], size=rec.shape[-2:], mode="area")
                    mm = F.interpolate(masks[m], size=rec.shape[-2:], mode="nearest")
                    loss_rec = loss_rec + ((rec - tgt).abs() * mm).sum() / mm.sum().clamp_min(1.0)
            loss = loss_cls + tcfg.lambda_rec * loss_rec
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg["cls"] += loss_cls.item() / steps
            agg["rec"] += float(loss_rec) / steps

        val = predict(model, ds, fold["val"])
        auc = roc_auc_score(y_val, val["fused"])
        history.append(dict(epoch=ep + 1, loss_cls=agg["cls"], loss_rec=agg["rec"], val_auroc=auc,
                            seconds=time.time() - t0))
        if ep >= tcfg.warmup_epochs and auc >= best_auc:
            best_auc, best_epoch = auc, ep + 1
            best_state = {n: v.detach().cpu().clone() for n, v in model.trainable_state_dict().items()}
        log(f"  fold {k} epoch {ep + 1:02d}  cls {agg['cls']:.3f}  rec {agg['rec']:.4f}  "
            f"val AUROC {auc:.3f}  ({time.time() - t0:.0f}s)")

    model.load_trainable_state_dict(best_state)
    val = predict(model, ds, fold["val"])
    test = predict(model, ds, fold["test"])
    extra = predict(model, ds, list(unlabeled)) if len(unlabeled) else None
    platt, thr = {}, {}
    for h in HEADS:
        platt[h] = fit_platt(val[h], y_val)
        thr[h] = youden_threshold(sigmoid(platt[h][0] * val[h] + platt[h][1]), y_val)

    def rows(pids, logits, split):
        df = pd.DataFrame({"patient_id": pids, "fold": k, "split": split})
        for h in HEADS:
            a, b = platt[h]
            df[f"logit_{h}"] = logits[h]
            df[f"prob_{h}"] = sigmoid(a * logits[h] + b)
            df[f"thr_{h}"] = thr[h]
        return df

    preds = [rows(fold["test"], test, "test"), rows(fold["val"], val, "val")]
    if extra is not None:
        preds.append(rows(list(unlabeled), extra, "unlabeled"))
    ckpt = dict(model_cfg=mcfg.to_dict(), train_cfg=asdict(tcfg), hw=ds.hw, fold=k, state=best_state,
                platt=platt, thresholds=thr, best_epoch=best_epoch, best_val_auroc=best_auc)
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return pd.concat(preds, ignore_index=True), ckpt, history


def run_cv(variant, folds=None, device=None, log=print, **overrides):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mcfg, tcfg = configs(variant, **overrides)
    torch.backends.cudnn.benchmark = True
    ds = BCMID(device)
    splits = make_folds(ds.patients, tcfg.n_splits, tcfg.seed)
    unlabeled = ds.patients[ds.patients.label < 0].patient_id.tolist()
    run_dir = paths.RUNS / variant
    run_dir.mkdir(parents=True, exist_ok=True)

    for k in (folds if folds is not None else range(tcfg.n_splits)):
        pred_file, ckpt_file = run_dir / f"fold{k}_preds.csv", run_dir / f"fold{k}.pt"
        if pred_file.exists() and ckpt_file.exists():
            log(f"[{variant}] fold {k} already done, skipping")
            continue
        f = splits[k]
        log(f"[{variant}] fold {k}: train {len(f['train'])}  val {len(f['val'])}  test {len(f['test'])}")
        t0 = time.time()
        preds, ckpt, history = train_fold(ds, f, k, mcfg, tcfg, unlabeled, log)
        torch.save(ckpt, ckpt_file)
        json.dump(history, open(run_dir / f"fold{k}_history.json", "w"), indent=1)
        preds.to_csv(pred_file, index=False)
        te = preds[preds.split == "test"]
        y = np.array([ds.label[p] for p in te.patient_id])
        log(f"[{variant}] fold {k} done in {(time.time() - t0) / 60:.1f} min  best epoch {ckpt['best_epoch']}  "
            f"val AUROC {ckpt['best_val_auroc']:.3f}  test AUROC fused {roc_auc_score(y, te.prob_fused):.3f}  "
            f"MG {roc_auc_score(y, te.prob_MG):.3f}  US {roc_auc_score(y, te.prob_US):.3f}")

    collect(variant, splits)
    return run_dir


def collect(variant, splits=None):
    """Merge per-fold outputs into results/<variant>/ (small, human-readable files)."""
    run_dir, out = paths.RUNS / variant, paths.RESULTS / variant
    files = sorted(run_dir.glob("fold*_preds.csv"))
    if not files:
        return None
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in files], ignore_index=True)
    _, patients = __import__("bcmid.data", fromlist=["load_tables"]).load_tables()
    preds = preds.merge(patients[["patient_id", "group_id", "birads", "category", "label"]], on="patient_id")
    preds[preds.split == "test"].drop(columns="split").to_csv(out / "oof_predictions.csv", index=False)
    preds[preds.split == "val"].drop(columns="split").to_csv(out / "val_predictions.csv", index=False)
    unl = preds[preds.split == "unlabeled"]
    if len(unl):
        cols = [c for c in unl.columns if c.startswith("prob_")]
        unl.groupby(["patient_id", "birads"], as_index=False)[cols].mean().to_csv(out / "unlabeled_predictions.csv", index=False)
    hist = {f.stem.split("_")[0]: json.load(open(f)) for f in sorted(run_dir.glob("fold*_history.json"))}
    json.dump(hist, open(out / "history.json", "w"), indent=1)
    meta = {}
    for f in sorted(run_dir.glob("fold*.pt")):
        c = torch.load(f, map_location="cpu", weights_only=False)
        meta[f.stem] = {k: c[k] for k in ("platt", "thresholds", "best_epoch", "best_val_auroc")}
        cfg = {"model": c["model_cfg"], "train": c["train_cfg"], "input_hw": c["hw"]}
    json.dump({"config": cfg, "folds": meta}, open(out / "run.json", "w"), indent=1)
    if splits is not None:
        json.dump(splits, open(paths.RESULTS / "splits.json", "w"))
    return preds
