"""Turn cross-validation outputs into metrics, figures and LaTeX fragments.

Every number that appears in the paper is written by this module from the
out-of-fold prediction files; nothing is typed by hand.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from . import paths
from .data import load_tables
from .metrics import bootstrap_ci, paired_bootstrap_delta

MODELS = {                       # fixed order: colour and row position follow the model, not its rank
    "full": "ViT-UNet MIL (proposed)",
    "no_recon": "w/o masked reconstruction",
    "no_decoder": "w/o U-Net decoder (ViT-MIL)",
    "frozen": "w/ fully frozen encoder",
    "dinov2_lr": "DINOv2 ViT-S/14 features + LR",
    "resnet50_lr": "ResNet-50 features + LR",
    "metadata_lr": "Metadata only (no pixels) + LR",
}
HEADS = {"fused": "Fusion (MG+US)", "MG": "Mammography only", "US": "Ultrasound only"}
COLORS = {"fused": "#2a78d6", "MG": "#eb6834", "US": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
N_BOOT = 2000


def load_oof(key):
    f = paths.RESULTS / key / "oof_predictions.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, dtype={"patient_id": str, "group_id": str}).sort_values("patient_id").reset_index(drop=True)


def head_metrics(df, head):
    y, p = df.label.to_numpy(), df[f"prob_{head}"].to_numpy()
    pred = (p >= df[f"thr_{head}"].to_numpy()).astype(int)
    m = bootstrap_ci(y, p, pred, n_boot=N_BOOT)
    fold_auc = [roc_auc_score(g.label, g[f"prob_{head}"]) for _, g in df.groupby("fold")]
    m["fold_auroc"] = [float(a) for a in fold_auc]
    m["fold_auroc_mean"], m["fold_auroc_sd"] = float(np.mean(fold_auc)), float(np.std(fold_auc, ddof=1))
    return m


CONFOUNDERS = {                      # study-level acquisition characteristics, all pixel-free
    "bilateral_mg": "Bilateral mammography (4 views)",
    "digital_mg": "Digital mammography system",
    "untagged_us": "Ultrasound files without side tag",
    "many_us": "More than two ultrasound frames",
}
SUBSETS = {
    "all": "All labelled studies",
    "unilateral": "Unilateral mammography only",
    "film": "Film mammography only",
    "tagged_us": "Side-tagged ultrasound only",
    "restricted": "All three restrictions",
}


def study_flags(manifest, studies):
    """One boolean column per acquisition characteristic, indexed by study."""
    import pandas as pd

    mg, us = manifest[manifest.modality == "MG"], manifest[manifest.modality == "US"]
    lab = studies.set_index("patient_id")
    digital = ~mg.inverted.astype(str).str.lower().eq("true")
    flags = pd.DataFrame(index=lab.index)
    flags["bilateral_mg"] = (lab.n_mg > 2).to_numpy()
    flags["digital_mg"] = digital.groupby(mg.patient_id).any().reindex(lab.index).fillna(False).astype(bool).to_numpy()
    flags["untagged_us"] = us.laterality.eq("UNK").groupby(us.patient_id).any().reindex(lab.index).fillna(False).astype(bool).to_numpy()
    flags["many_us"] = (lab.n_us > 2).to_numpy()
    return flags


def confounder_audit(manifest, labelled, oof):
    """How much of the label is tied to the way a study was acquired?

    Reports the association of each acquisition characteristic with the label
    and the AUROC of every model inside subsets where the characteristic is
    held constant. The subsets were fixed before any model was trained.
    """
    from scipy.stats import fisher_exact

    flags = study_flags(manifest, labelled)
    y = labelled.set_index("patient_id").label.astype(int)
    mg = manifest[manifest.modality == "MG"]
    film = mg.inverted.astype(str).str.lower().eq("true")
    out = {"n_mg_film": int(film.sum()), "n_mg_digital": int((~film).sum()), "associations": [], "subgroups": []}
    for key, name in CONFOUNDERS.items():
        f = flags[key]
        table = [[int((f & (y == 1)).sum()), int((f & (y == 0)).sum())],
                 [int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())]]
        out["associations"].append({"key": key, "name": name, "n_with": int(f.sum()), "pos_with": table[0][0],
                                    "n_without": int((~f).sum()), "pos_without": table[1][0],
                                    "fisher_p": float(fisher_exact(table)[1])})
    keep = {"all": flags.bilateral_mg | True, "unilateral": ~flags.bilateral_mg, "film": ~flags.digital_mg,
            "tagged_us": ~flags.untagged_us,
            "restricted": ~flags.bilateral_mg & ~flags.digital_mg & ~flags.untagged_us}
    for key, name in SUBSETS.items():
        ids = set(keep[key][keep[key]].index)
        row = {"key": key, "name": name, "n": len(ids), "n_pos": int(y[list(ids)].sum()), "auroc": {}, "delta": {}}
        for model, df in oof.items():
            sub = df[df.patient_id.isin(ids)]
            heads = HEADS if model == "full" else ("fused",)
            for h in heads:
                if sub.label.nunique() == 2:
                    ci = bootstrap_ci(sub.label.to_numpy(), sub[f"prob_{h}"].to_numpy(), n_boot=N_BOOT)["auroc"]
                    row["auroc"][f"{model}:{h}"] = list(ci)
        if "full" in oof:                     # proposed model minus each reference, paired within the subset
            ref = oof["full"][oof["full"].patient_id.isin(ids)]
            for other in ("metadata_lr", "dinov2_lr"):
                if other in oof:
                    cmp_ = oof[other][oof[other].patient_id.isin(ids)]
                    assert (cmp_.patient_id.values == ref.patient_id.values).all()
                    row["delta"][other] = paired_bootstrap_delta(ref.label, ref.prob_fused, cmp_.prob_fused, n_boot=N_BOOT)
        out["subgroups"].append(row)
    return out



def summarise():
    manifest, patients = load_tables()
    labelled = patients[patients.label >= 0]
    summary = {
        "dataset": {
            "n_entries": int(len(patients)), "n_women": int(patients.group_id.nunique()),
            "n_labelled": int(len(labelled)), "n_pos": int((labelled.label == 1).sum()),
            "n_neg": int((labelled.label == 0).sum()), "n_unlabelled": int((patients.label < 0).sum()),
            "n_mg": int((manifest.modality == "MG").sum()), "n_us": int((manifest.modality == "US").sum()),
            "birads_counts": {str(k): int(v) for k, v in patients.birads.value_counts().sort_index().items()},
            "mg_per_study_median": float(patients.n_mg.median()), "us_per_study_median": float(patients.n_us.median()),
            "us_per_study_max": int(patients.n_us.max()),
        },
        "models": {}, "comparisons": {}, "modality_comparisons": {}, "birads": [],
    }
    oof = {k: load_oof(k) for k in MODELS}
    oof = {k: v for k, v in oof.items() if v is not None}
    if "inverted" in manifest.columns:
        summary["confounders"] = confounder_audit(manifest, labelled, oof)
    for key, df in oof.items():
        summary["models"][key] = {"name": MODELS[key], "heads": {h: head_metrics(df, h) for h in HEADS}}
    if "full" in oof:
        ref = oof["full"]
        for key, df in oof.items():
            if key == "full":
                continue
            assert (df.patient_id.values == ref.patient_id.values).all()
            summary["comparisons"][key] = paired_bootstrap_delta(ref.label, ref.prob_fused, df.prob_fused, n_boot=N_BOOT)
        for h in ("MG", "US"):
            summary["modality_comparisons"][h] = paired_bootstrap_delta(ref.label, ref.prob_fused, ref[f"prob_{h}"], n_boot=N_BOOT)
        flag = ref.prob_fused >= ref.thr_fused
        for b, g in ref.groupby("birads"):
            summary["birads"].append({"birads": int(b), "n": int(len(g)), "label": int(g.label.iloc[0]),
                                      "median_prob": float(g.prob_fused.median()),
                                      "mean_prob": float(g.prob_fused.mean()),
                                      "flagged": int(flag[g.index].sum())})
        run = json.load(open(paths.RESULTS / "full" / "run.json"))
        summary["config"] = run["config"]
        summary["best_epochs"] = [v["best_epoch"] for v in run["folds"].values()]
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(paths.RESULTS / "summary.json", "w"), indent=1)

    rows = []
    for key, m in summary["models"].items():
        for h, v in m["heads"].items():
            row = {"model": key, "name": m["name"], "head": h}
            for met in ("auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "f1", "brier", "ece"):
                row[met], row[met + "_lo"], row[met + "_hi"] = v[met]
            row["fold_auroc_mean"], row["fold_auroc_sd"] = v["fold_auroc_mean"], v["fold_auroc_sd"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(paths.RESULTS / "metrics_table.csv", index=False)
    return summary, oof


# ---------------------------------------------------------------- figures

def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": "#52514e", "ytick.labelcolor": "#52514e", "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.axisbelow": True,
    })


def _save(fig, name):
    import matplotlib.pyplot as plt
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / f"{name}.pdf")
    fig.savefig(paths.FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_roc_pr(summary, oof):
    import matplotlib.pyplot as plt
    df = oof["full"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.7))
    a.plot([0, 1], [0, 1], color="#c3c2b7", lw=0.8)
    prev = df.label.mean()
    b.axhline(prev, color="#c3c2b7", lw=0.8)
    b.text(0.02, prev + 0.015, f"prevalence {prev:.2f}", ha="left", va="bottom", color=MUTED, fontsize=6.5)
    below = dict(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)     # legends never sit on the curves
    for h, name in HEADS.items():
        m = summary["models"]["full"]["heads"][h]
        fpr, tpr, _ = roc_curve(df.label, df[f"prob_{h}"])
        a.plot(fpr, tpr, color=COLORS[h], lw=1.6 if h == "fused" else 1.2,
               label=f"{name}: {m['auroc'][0]:.3f} [{m['auroc'][1]:.3f}, {m['auroc'][2]:.3f}]")
        pr, rc, _ = precision_recall_curve(df.label, df[f"prob_{h}"])
        b.plot(rc, pr, color=COLORS[h], lw=1.6 if h == "fused" else 1.2, drawstyle="steps-post",
               label=f"{name}: {m['auprc'][0]:.3f} [{m['auprc'][1]:.3f}, {m['auprc'][2]:.3f}]")
    a.set(xlabel="1 - specificity", ylabel="Sensitivity", xlim=(-0.01, 1), ylim=(0, 1.01), title="(a) ROC, AUROC [95% CI]")
    b.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", xlim=(0, 1), ylim=(0, 1.02), title="(b) Precision-recall, AUPRC [95% CI]")
    a.legend(**below)
    b.legend(**below)
    fig.tight_layout()
    _save(fig, "fig_roc_pr")


def fig_calibration_birads(summary, oof):
    import matplotlib.pyplot as plt
    df = oof["full"]
    y, p = df.label.to_numpy(), df.prob_fused.to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    edges = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 4)
    a.plot([0, 1], [0, 1], color="#c3c2b7", lw=0.8)
    xs, ys, ns = [], [], []
    for k in range(5):
        m = idx == k
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    a.plot(xs, ys, color=COLORS["fused"], lw=1.4, marker="o", ms=5, mec="white", mew=1)
    for x_, y_, n_ in zip(xs, ys, ns):
        a.annotate(f"n={n_}", (x_, y_), textcoords="offset points", xytext=(5, -9), fontsize=6.5, color="#52514e")
    m = summary["models"]["full"]["heads"]["fused"]
    a.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", xlim=(0, 1), ylim=(-0.02, 1.02),
          title=f"(a) Calibration (ECE {m['ece'][0]:.3f}, Brier {m['brier'][0]:.3f})")
    a.set_aspect("equal")

    rng = np.random.default_rng(0)
    cats = sorted(df.birads.unique())
    for c in cats:
        g = df[df.birads == c]
        col = "#eb6834" if g.label.iloc[0] == 1 else "#2a78d6"
        b.scatter(c + rng.uniform(-0.22, 0.22, len(g)), g.prob_fused, s=9, color=col, alpha=0.55, lw=0)
        b.plot([c - 0.3, c + 0.3], [g.prob_fused.median()] * 2, color=INK, lw=1.3)
    b.scatter([], [], s=12, color="#2a78d6", label="negative (BI-RADS 1-3)")
    b.scatter([], [], s=12, color="#eb6834", label="positive (BI-RADS 4-6)")
    b.plot([], [], color=INK, lw=1.3, label="median")
    b.set(xlabel="BI-RADS category assigned by the radiologist", ylabel="Predicted probability (out-of-fold)",
          xticks=cats, ylim=(-0.02, 1.02), title="(b) Prediction by BI-RADS category")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    _save(fig, "fig_calibration_birads")


def fig_models(summary):
    import matplotlib.pyplot as plt
    keys = [k for k in MODELS if k in summary["models"]]
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(keys) + 0.9))
    for i, k in enumerate(keys):
        pt, lo, hi = summary["models"][k]["heads"]["fused"]["auroc"]
        col = COLORS["fused"] if k == "full" else "#898781"
        ax.plot([lo, hi], [i, i], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(pt, i, "o", color=col, ms=6, mec="white", mew=1)
        ax.text(hi + 0.006, i, f"{pt:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(keys)), [MODELS[k] for k in keys])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="AUROC (fused head, pooled out-of-fold, 95% bootstrap CI)")
    lo_all = min(summary["models"][k]["heads"]["fused"]["auroc"][1] for k in keys)
    ax.set_xlim(max(0.4, lo_all - 0.04), 1.0)
    fig.tight_layout()
    _save(fig, "fig_models")


def fig_audit(summary):
    """AUROC of the image models and of the pixel-free control inside acquisition-homogeneous subsets."""
    import matplotlib.pyplot as plt
    rows = summary["confounders"]["subgroups"]
    series = [("full:fused", "ViT-UNet MIL (proposed)", "#2a78d6", "o"),
              ("dinov2_lr:fused", "DINOv2 features + LR", "#1baf7a", "s"),
              ("metadata_lr:fused", "Metadata only (no pixels)", "#eb6834", "D")]
    series = [s for s in series if any(s[0] in r["auroc"] for r in rows)]
    fig, ax = plt.subplots(figsize=(6.6, 0.62 * len(rows) + 1.0))
    off = np.linspace(-0.24, 0.24, len(series)) if len(series) > 1 else [0.0]
    ax.axvline(0.5, color="#c3c2b7", lw=0.8)
    for (key, name, col, mk), dy in zip(series, off):
        for i, r in enumerate(rows):
            if key not in r["auroc"]:
                continue
            pt, lo, hi = r["auroc"][key]
            ax.plot([lo, hi], [i + dy] * 2, color=col, lw=1.4, solid_capstyle="round")
            ax.plot(pt, i + dy, mk, color=col, ms=5.5, mec="white", mew=0.9, label=name if i == 0 else None)
            ax.text(hi + 0.008, i + dy, f"{pt:.2f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(rows)), [f"{r['name']}\n(n={r['n']}, {r['n_pos']} positive)" for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="AUROC on the fused input (pooled out-of-fold, 95% bootstrap CI)", xlim=(0.42, 1.02))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(series), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    _save(fig, "fig_audit")


def make_figures(summary, oof):
    import matplotlib
    matplotlib.use("Agg")
    _style()
    if "full" in oof:
        fig_roc_pr(summary, oof)
        fig_calibration_birads(summary, oof)
    if summary["models"]:
        fig_models(summary)
    if summary.get("confounders", {}).get("subgroups"):
        fig_audit(summary)


# ---------------------------------------------------------------- LaTeX

def _ci(v, d=3):
    return f"{v[0]:.{d}f} ({v[1]:.{d}f}--{v[2]:.{d}f})"


def _s(x):
    """Signed number in math mode, so that a negative value gets a real minus sign."""
    return f"${x:+.3f}$"


def _p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def make_latex(summary):
    paths.GENERATED.mkdir(parents=True, exist_ok=True)
    ds, models = summary["dataset"], summary["models"]
    mac = {
        "NEntries": ds["n_entries"], "NWomen": ds["n_women"], "NLabelled": ds["n_labelled"], "NPos": ds["n_pos"],
        "NNeg": ds["n_neg"], "NUnlabelled": ds["n_unlabelled"], "NMG": ds["n_mg"], "NUS": ds["n_us"],
        "NImages": ds["n_mg"] + ds["n_us"], "PrevalencePct": f"{100 * ds['n_pos'] / ds['n_labelled']:.1f}",
        "USMax": ds["us_per_study_max"],
    }
    words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six"]     # TeX macro names cannot hold digits
    for b, n in ds["birads_counts"].items():
        mac["NBirads" + words[int(b)]] = n
    short = {"full": "Full", "no_recon": "NoRecon", "no_decoder": "NoDecoder", "frozen": "Frozen",
             "dinov2_lr": "DinoLR", "resnet50_lr": "ResNetLR", "metadata_lr": "MetaLR"}
    hshort = {"fused": "", "MG": "MG", "US": "US"}
    for k, m in models.items():
        for h, v in m["heads"].items():
            pre = short[k] + hshort[h]
            mac[pre + "Auroc"] = f"{v['auroc'][0]:.3f}"
            mac[pre + "AurocCI"] = f"{v['auroc'][1]:.3f}--{v['auroc'][2]:.3f}"
            mac[pre + "Auprc"] = f"{v['auprc'][0]:.3f}"
            mac[pre + "AuprcCI"] = f"{v['auprc'][1]:.3f}--{v['auprc'][2]:.3f}"
            if k == "full":
                for met, nm in (("sensitivity", "Sens"), ("specificity", "Spec"), ("balanced_accuracy", "BalAcc"),
                                ("f1", "Fone"), ("ppv", "PPV"), ("npv", "NPV")):
                    mac[pre + nm] = f"{100 * v[met][0]:.1f}"
                    mac[pre + nm + "CI"] = f"{100 * v[met][1]:.1f}--{100 * v[met][2]:.1f}"
                mac[pre + "Brier"], mac[pre + "ECE"] = f"{v['brier'][0]:.3f}", f"{v['ece'][0]:.3f}"
                mac[pre + "FoldAuroc"] = f"{v['fold_auroc_mean']:.3f} $\\pm$ {v['fold_auroc_sd']:.3f}"
                mac[pre + "FoldMin"], mac[pre + "FoldMax"] = f"{min(v['fold_auroc']):.2f}", f"{max(v['fold_auroc']):.2f}"
                for c in ("tp", "tn", "fp", "fn"):
                    mac[pre + c.upper()] = v[c]
    for k, c in summary["comparisons"].items():
        mac["Delta" + short[k]] = _s(c["delta"])
        mac["Delta" + short[k] + "CI"] = f"{_s(c['lo'])} to {_s(c['hi'])}"
        mac["P" + short[k]] = _p(c["p"])
    for h, c in summary["modality_comparisons"].items():
        mac["DeltaFusion" + h] = _s(c["delta"])
        mac["DeltaFusion" + h + "CI"] = f"{_s(c['lo'])} to {_s(c['hi'])}"
        mac["PFusion" + h] = _p(c["p"])
    conf = summary.get("confounders")
    if conf:
        mac.update(NMGFilm=conf["n_mg_film"], NMGDigital=conf["n_mg_digital"])
        cshort = {"bilateral_mg": "Bilateral", "digital_mg": "Digital", "untagged_us": "Untagged", "many_us": "ManyUS"}
        for a in conf["associations"]:
            c = cshort[a["key"]]
            mac[c + "N"], mac[c + "Pos"] = a["n_with"], a["pos_with"]
            mac[c + "PosPct"] = f"{100 * a['pos_with'] / max(a['n_with'], 1):.0f}"
            mac[c + "RestN"], mac[c + "RestPos"] = a["n_without"], a["pos_without"]
            mac[c + "RestPosPct"] = f"{100 * a['pos_without'] / max(a['n_without'], 1):.0f}"
            mac[c + "P"] = _p(a["fisher_p"])
        sshort = {"unilateral": "Unilateral", "film": "FilmOnly", "tagged_us": "TaggedUS", "restricted": "Restricted"}
        for r in conf["subgroups"]:
            if r["key"] not in sshort:
                continue
            c = sshort[r["key"]]
            mac[c + "N"], mac[c + "Pos"] = r["n"], r["n_pos"]
            for mk, suffix in (("full:fused", ""), ("full:MG", "MG"), ("full:US", "US"), ("dinov2_lr:fused", "Dino"),
                               ("metadata_lr:fused", "Meta")):
                if mk in r["auroc"]:
                    v = r["auroc"][mk]
                    mac[c + suffix + "Auroc"], mac[c + suffix + "AurocCI"] = f"{v[0]:.3f}", f"{v[1]:.3f}--{v[2]:.3f}"
            for other, suffix in (("metadata_lr", "Meta"), ("dinov2_lr", "Dino")):
                d = r.get("delta", {}).get(other)
                if d:
                    mac[c + "Delta" + suffix] = _s(d["delta"])
                    mac[c + "Delta" + suffix + "CI"] = f"{_s(d['lo'])} to {_s(d['hi'])}"
                    mac[c + "P" + suffix] = _p(d["p"])
    if "config" in summary:
        t = summary["config"]["train"]
        mac.update(Epochs=t["epochs"], BatchStudies=t["batch_studies"], MaskRatioPct=int(round(100 * t["mask_ratio"])),
                   BestEpochs=", ".join(str(e) for e in summary["best_epochs"]))
    with open(paths.GENERATED / "macros.tex", "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by scripts/04_report.py -- do not edit.\n")
        for k, v in mac.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # main table: heads of the proposed model + baselines
    lines = []
    if "full" in models:
        for h, name in HEADS.items():
            v = models["full"]["heads"][h]
            lines.append(f"ViT-UNet MIL & {name} & {_ci(v['auroc'])} & {_ci(v['auprc'])} & "
                         f"{100 * v['sensitivity'][0]:.1f} & {100 * v['specificity'][0]:.1f} & {v['f1'][0]:.3f} \\\\")
    for k in ("dinov2_lr", "resnet50_lr"):
        if k in models:
            lines.append("\\midrule")
            for h, name in HEADS.items():
                v = models[k]["heads"][h]
                lines.append(f"{MODELS[k].replace(' features + LR', ' + LR')} & {name} & {_ci(v['auroc'])} & {_ci(v['auprc'])} & "
                             f"{100 * v['sensitivity'][0]:.1f} & {100 * v['specificity'][0]:.1f} & {v['f1'][0]:.3f} \\\\")
    # \bottomrule lives in the generated file: \input directly before a rule breaks tabular parsing
    open(paths.GENERATED / "table_main.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n\\bottomrule\n")

    lines = []
    groups = (("full",), ("no_recon", "no_decoder", "frozen"), ("dinov2_lr", "resnet50_lr", "metadata_lr"))
    for group in groups:
        rows = []
        for k in group:
            if k not in models:
                continue
            v = models[k]["heads"]["fused"]
            c = summary["comparisons"].get(k)
            delta = "reference & --" if c is None else f"{_s(c['delta'])} ({_s(c['lo'])} to {_s(c['hi'])}) & {_p(c['p'])}"
            rows.append(f"{MODELS[k]} & {_ci(v['auroc'])} & {_ci(v['auprc'])} & "
                        f"{v['fold_auroc_mean']:.3f} $\\pm$ {v['fold_auroc_sd']:.3f} & {delta} \\\\")
        if rows:
            lines += (["\\midrule"] if lines else []) + rows
    open(paths.GENERATED / "table_ablation.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n\\bottomrule\n")

    lines = []
    for r in summary["birads"]:
        ref = "positive" if r["label"] == 1 else "negative"
        lines.append(f"{r['birads']} & {ref} & {r['n']} & {r['median_prob']:.2f} & {r['flagged']}/{r['n']} "
                     f"({100 * r['flagged'] / r['n']:.0f}\\%) \\\\")
    open(paths.GENERATED / "table_birads.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n\\bottomrule\n")

    if conf:
        lines = []
        for a in conf["associations"]:
            lines.append(f"{a['name']} & {a['pos_with']}/{a['n_with']} ({100 * a['pos_with'] / max(a['n_with'], 1):.0f}\\%) & "
                         f"{a['pos_without']}/{a['n_without']} ({100 * a['pos_without'] / max(a['n_without'], 1):.0f}\\%) & "
                         f"{_p(a['fisher_p'])} \\\\")
        open(paths.GENERATED / "table_confounders.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n\\bottomrule\n")

        cols = [k for k in ("full:fused", "full:MG", "full:US", "dinov2_lr:fused", "metadata_lr:fused")
                if any(k in r["auroc"] for r in conf["subgroups"])]
        lines = []
        for r in conf["subgroups"]:
            cells = [(_ci(r["auroc"][k], 2) if k in r["auroc"] else "--") for k in cols]
            lines.append(f"{r['name']} & {r['n']} ({r['n_pos']}) & " + " & ".join(cells) + " \\\\")
            if r["key"] == "all":
                lines.append("\\midrule")
        open(paths.GENERATED / "table_subgroups.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n\\bottomrule\n")


def run():
    summary, oof = summarise()
    make_figures(summary, oof)
    make_latex(summary)
    return summary
