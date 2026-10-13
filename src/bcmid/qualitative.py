"""Qualitative figure: decoder saliency on held-out studies, chosen by a fixed rule."""
import json

import cv2
import numpy as np
import pandas as pd

from . import paths
from .data import load_tables
from .infer import Ensemble


def tint(gray, sal, strength=0.7):
    s = cv2.resize(sal.astype(np.float32), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(s, 5), s.max()
    s = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) ** 1.5
    rgb = np.repeat(gray[..., None], 3, 2)
    a = (s * strength)[..., None]
    return np.clip(rgb * (1 - a) + np.array([235, 104, 52]) / 255.0 * a, 0, 1)


def pick_cases(df):
    """Fixed selection rule (stated in the caption) so the examples are not hand-picked."""
    pred = df.prob_fused >= df.thr_fused
    tp = df[pred & (df.label == 1)].sort_values("prob_fused", ascending=False)
    fn = df[~pred & (df.label == 1)].sort_values("prob_fused")
    fp = df[pred & (df.label == 0)].sort_values("prob_fused", ascending=False)
    cases = []
    if len(tp):
        cases.append(("True positive (highest score)", tp.iloc[0]))
        cases.append(("True positive (median score)", tp.iloc[len(tp) // 2]))
    if len(fn):
        cases.append(("False negative (lowest score)", fn.iloc[0]))
    if len(fp):
        cases.append(("False positive (highest score)", fp.iloc[0]))
    return cases


def saliency_concentration(ens, df, manifest, arrays):
    """How far are the saliency maps and MIL weights from uniform, on held-out studies?

    For every study, scored by the fold model that did not see it: the share of
    the saliency mass held by the 10% most salient pixels (0.10 = uniform), the
    mean saliency on tissue versus background, and the largest MIL weight
    relative to a uniform weight.
    """
    rows = []
    for r in df.itertuples():
        imgs = {}
        for m in ("MG", "US"):
            sub = manifest[(manifest.patient_id == r.patient_id) & (manifest.modality == m)].sort_values("idx")
            imgs[m] = np.stack([np.asarray(arrays[m][i]) for i in sub.idx])
        res = ens.predict(imgs, folds=[int(r.fold)])
        for m in ("MG", "US"):
            n = len(imgs[m])
            for j in range(n):
                s = res[m]["sal"][j]
                fg = cv2.resize(res[m]["input"][j], (s.shape[1], s.shape[0]), interpolation=cv2.INTER_AREA) > 0.02
                flat = np.sort(s.ravel())[::-1]
                rows.append({"mod": m, "top10": flat[: max(1, len(flat) // 10)].sum() / flat.sum(),
                             "fg": s[fg].mean() if fg.any() else np.nan, "bg": s[~fg].mean() if (~fg).any() else np.nan,
                             "att_ratio": res[m]["att"][j] * n if n > 1 else np.nan})
    d = pd.DataFrame(rows)
    return {m: {"images": int(len(g)), "top10_share": float(g.top10.median()), "foreground": float(g.fg.median()),
                "background": float(g.bg.median()), "att_ratio_median": float(g.att_ratio.median()),
                "att_ratio_p95": float(g.att_ratio.quantile(0.95))} for m, g in d.groupby("mod")}


def run(variant="full"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest, patients = load_tables()
    df = pd.read_csv(paths.RESULTS / variant / "oof_predictions.csv", dtype={"patient_id": str})
    arrays = {m: np.load(paths.CACHE / f, mmap_mode="r") for m, f in (("MG", "mg.npy"), ("US", "us.npy"))}
    ens = Ensemble(variant)
    cases = pick_cases(df)
    sal = saliency_concentration(ens, df, manifest, arrays) if ens.has_decoder else {}
    json.dump(sal, open(paths.RESULTS / "saliency_stats.json", "w"), indent=1)

    fig, axes = plt.subplots(len(cases), 4, figsize=(6.9, 2.05 * len(cases)),
                             gridspec_kw={"width_ratios": [1, 1, 2.67, 2.67], "wspace": 0.04, "hspace": 0.16})
    picked = []
    for r, (title, row) in enumerate(cases):
        imgs = {}
        for m in ("MG", "US"):
            sub = manifest[(manifest.patient_id == row.patient_id) & (manifest.modality == m)].sort_values("idx")
            imgs[m] = np.stack([np.asarray(arrays[m][i]) for i in sub.idx])
        res = ens.predict(imgs, folds=[int(row.fold)])
        for c, m in enumerate(("MG", "US")):
            j = int(np.argmax(res[m]["att"]))
            g = imgs[m][j].astype(np.float32) / 255.0
            for ax, im in ((axes[r, 2 * c], np.repeat(g[..., None], 3, 2)), (axes[r, 2 * c + 1], tint(g, res[m]["sal"][j]))):
                ax.imshow(im, aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
        axes[r, 0].set_ylabel(f"BI-RADS {int(row.birads)}\np = {row.prob_fused:.2f}", fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=6)
        axes[r, 2].set_title(title, fontsize=7.5, loc="left", pad=3)
        picked.append({"case": title, "patient_id": row.patient_id, "birads": int(row.birads),
                       "prob_fused": float(row.prob_fused), "fold": int(row.fold)})
    for c, name in enumerate(("Mammogram", "Saliency", "Ultrasound frame", "Saliency")):
        axes[-1, c].set_xlabel(name, fontsize=7)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / "fig_qualitative.pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(paths.FIGURES / "fig_qualitative.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    json.dump(picked, open(paths.RESULTS / "qualitative_cases.json", "w"), indent=1)

    # preprocessing statistics quoted in the paper
    mg = manifest[manifest.modality == "MG"]
    us = manifest[manifest.modality == "US"]
    ok = lambda s: s.astype(str).str.lower().eq("true")
    hw = ens.hw
    crop_h = pd.to_numeric(mg.y1) - pd.to_numeric(mg.y0)
    crop_w = pd.to_numeric(mg.x1) - pd.to_numeric(mg.x0)
    factor = np.maximum(crop_h / hw["MG"][0], crop_w / hw["MG"][1])       # letterbox scale
    stats = {"mg_seg_fallback": int((~ok(mg.seg_ok)).sum()), "us_seg_fallback": int((~ok(us.seg_ok)).sum()),
             "mg_downsample": float(factor.median()),
             "mg_orig_h": int(pd.to_numeric(mg.orig_h).median()), "mg_orig_w": int(pd.to_numeric(mg.orig_w).median()),
             "us_orig_h": int(pd.to_numeric(us.orig_h).median()), "us_orig_w": int(pd.to_numeric(us.orig_w).median()),
             "us_laterality_unknown": int((us.laterality == "UNK").sum())}
    sizes = {}
    if "inverted" in mg.columns:
        for name, sub in (("film", mg[ok(mg.inverted)]), ("digital", mg[~ok(mg.inverted)])):
            if len(sub):
                top = (sub.orig_w.astype(str) + "x" + sub.orig_h.astype(str)).value_counts()
                sizes[name] = {"size": top.index[0], "share": float(top.iloc[0] / len(sub)), "n": int(len(sub))}
        stats["mg_frame_trimmed"] = int(pd.to_numeric(mg.frame).sum())
    stats["mg_sizes"] = sizes
    stats["us_size_share"] = float(((pd.to_numeric(us.orig_w) == stats["us_orig_w"]) & (pd.to_numeric(us.orig_h) == stats["us_orig_h"])).mean())
    json.dump(stats, open(paths.RESULTS / "preprocessing_stats.json", "w"), indent=1)
    paths.GENERATED.mkdir(parents=True, exist_ok=True)
    with open(paths.GENERATED / "macros_preproc.tex", "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by scripts/05_qualitative.py -- do not edit.\n")
        for name, macro in (("film", "MGFilmSize"), ("digital", "MGDigitalSize")):
            if name in sizes:
                w, h = sizes[name]["size"].split("x")
                fh.write(f"\\newcommand{{\\{macro}}}{{{w}$\\times${h}}}\n")
        fh.write(f"\\newcommand{{\\MGFrameTrimmed}}{{{stats.get('mg_frame_trimmed', 0)}}}\n")
        for m, v in sal.items():
            fh.write(f"\\newcommand{{\\SalTop{m}}}{{{100 * v['top10_share']:.1f}}}\n")
            fh.write(f"\\newcommand{{\\SalFg{m}}}{{{v['foreground']:.2f}}}\n")
            fh.write(f"\\newcommand{{\\SalBg{m}}}{{{v['background']:.2f}}}\n")
            fh.write(f"\\newcommand{{\\AttMedian{m}}}{{{v['att_ratio_median']:.2f}}}\n")
            fh.write(f"\\newcommand{{\\AttHigh{m}}}{{{v['att_ratio_p95']:.2f}}}\n")
        fh.write(f"\\newcommand{{\\MGFallback}}{{{stats['mg_seg_fallback']}}}\n")
        fh.write(f"\\newcommand{{\\USFallback}}{{{stats['us_seg_fallback']}}}\n")
        fh.write(f"\\newcommand{{\\MGDownsample}}{{{stats['mg_downsample']:.0f}}}\n")
        fh.write(f"\\newcommand{{\\MGOrigSize}}{{{stats['mg_orig_w']}$\\times${stats['mg_orig_h']}}}\n")
        fh.write(f"\\newcommand{{\\USOrigSize}}{{{stats['us_orig_w']}$\\times${stats['us_orig_h']}}}\n")
        fh.write(f"\\newcommand{{\\NQualCases}}{{{len(cases)}}}\n")
    print("qualitative figure:", [p["case"] for p in picked])
    print("preprocessing stats:", stats)
