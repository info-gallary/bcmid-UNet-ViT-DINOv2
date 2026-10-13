"""Frozen-feature baselines: pretrained encoder -> study-level mean pooling -> logistic regression.

They share the folds, the validation-based calibration and the thresholding
protocol of the deep model, so their numbers are directly comparable.
"""
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import paths
from .data import BCMID, MODALITIES, load_tables, make_folds
from .metrics import fit_platt, sigmoid, youden_threshold
from .model import IMAGENET_MEAN, IMAGENET_STD

ENCODERS = {
    "dinov2_lr": "vit_small_patch14_reg4_dinov2.lvd142m",
    "resnet50_lr": "resnet50.a1_in1k",
}
KEYS = (*ENCODERS, "metadata_lr")


@torch.no_grad()
def image_features(ds, name, batch=64):
    """One feature vector per image for both modalities."""
    dev = torch.device(ds.device)
    is_vit = name.startswith("vit")
    kw = dict(dynamic_img_size=True) if is_vit else {}
    net = timm.create_model(name, pretrained=True, num_classes=0, **kw).to(dev).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    feats = {}
    for m in MODALITIES:
        out = []
        for i in range(0, len(ds.images[m]), batch):
            x = ds.images[m][i:i + batch].unsqueeze(1).float() / 255.0
            x = (x.expand(-1, 3, -1, -1) - mean) / std
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                if is_vit:
                    t = net.forward_features(x)
                    f = torch.cat([t[:, 0], t[:, net.num_prefix_tokens:].mean(1)], 1)
                else:
                    f = net(x)
            out.append(f.float().cpu())
        feats[m] = torch.cat(out).numpy()
    del net
    return feats


def study_features(ds, feats):
    """Mean over the images of a study, per modality."""
    out = {}
    for m in MODALITIES:
        out[m] = {pid: feats[m][idx.cpu().numpy()].mean(0) for pid, idx in ds.index[m].items()}
    return out


def _fit_predict(Xtr, ytr, Xs, seed):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=5, scoring="roc_auc", class_weight="balanced",
                                             max_iter=2000, random_state=seed))
    clf.fit(Xtr, ytr)
    return [clf.decision_function(X) for X in Xs]


def metadata_features():
    """Pixel-free study descriptors: how many images were acquired and on which mammography system.

    A classifier on these alone measures how much of the label is predictable
    from acquisition habits rather than from image content.
    """
    manifest, patients = load_tables()
    mg, us = manifest[manifest.modality == "MG"], manifest[manifest.modality == "US"]
    digital = (~mg.inverted.astype(str).str.lower().eq("true")).groupby(mg.patient_id).any()
    untagged = us.laterality.eq("UNK").groupby(us.patient_id).any()        # file name carries no L/R tag
    both = us[us.laterality != "UNK"].groupby("patient_id").laterality.nunique().gt(1)
    sf = {"MG": {}, "US": {}}
    for r in patients.itertuples():
        p = r.patient_id
        sf["MG"][p] = np.array([float(r.n_mg), float(digital.get(p, False))])
        sf["US"][p] = np.array([float(r.n_us), float(untagged.get(p, False)), float(both.get(p, False))])
    return patients, sf


def run(key, device=None, seed=0, log=print):
    if key == "metadata_lr":
        patients, sf = metadata_features()
        label = dict(zip(patients.patient_id, patients.label))
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ds = BCMID(device)
        patients, label = ds.patients, ds.label
        sf = study_features(ds, image_features(ds, ENCODERS[key]))
    folds = make_folds(patients, seed=seed)

    def X(pids, head):
        mods = MODALITIES if head == "fused" else (head,)
        return np.stack([np.concatenate([sf[m][p] for m in mods]) for p in pids])

    def y(pids):
        return np.array([label[p] for p in pids])

    frames = []
    for k, f in enumerate(folds):
        dfs = {s: pd.DataFrame({"patient_id": f[s], "fold": k, "split": s}) for s in ("test", "val")}
        for head in ("fused", "MG", "US"):
            z_val, z_test = _fit_predict(X(f["train"], head), y(f["train"]), [X(f["val"], head), X(f["test"], head)], seed)
            a, b = fit_platt(z_val, y(f["val"]))
            thr = youden_threshold(sigmoid(a * z_val + b), y(f["val"]))
            for s, z in (("val", z_val), ("test", z_test)):
                dfs[s][f"logit_{head}"], dfs[s][f"prob_{head}"], dfs[s][f"thr_{head}"] = z, sigmoid(a * z + b), thr
        frames += list(dfs.values())
        log(f"[{key}] fold {k} done")
    preds = pd.concat(frames, ignore_index=True)
    _, patients = load_tables()
    preds = preds.merge(patients[["patient_id", "group_id", "birads", "category", "label"]], on="patient_id")
    out = paths.RESULTS / key
    out.mkdir(parents=True, exist_ok=True)
    preds[preds.split == "test"].drop(columns="split").to_csv(out / "oof_predictions.csv", index=False)
    preds[preds.split == "val"].drop(columns="split").to_csv(out / "val_predictions.csv", index=False)
    return preds
