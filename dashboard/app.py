"""BCMID ViT-UNet dashboard.

Run from the project root:
    streamlit run dashboard/app.py

Reads the cross-validation outputs in results/ and, when the work directory with
the image cache and checkpoints is present, runs the trained models live.
"""
import os
import site
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if site.getusersitepackages() not in sys.path:
    site.addsitedir(site.getusersitepackages())
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from bcmid import paths
from bcmid.metrics import point_metrics
from bcmid.report import HEADS, MODELS

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
HEAD_COLOR = {"fused": BLUE, "MG": ORANGE, "US": AQUA}
SALIENCY_NOTE = ("The orange tint is the U-Net decoder's saliency map, stretched per image for display. On held-out "
                 "studies these maps are nearly uniform over the tissue (the 10% most salient pixels carry about 12% "
                 "of the weight), so they show where the model pools from. They are not lesion maps, and BCMID has "
                 "no lesion annotations to validate them against.")

st.set_page_config(page_title="BCMID ViT-UNet", page_icon=":material/monitoring:", layout="wide")


# ------------------------------------------------------------------ loading

@st.cache_data(show_spinner=False)
def load_summary():
    f = paths.RESULTS / "summary.json"
    return json.load(open(f)) if f.exists() else None


@st.cache_data(show_spinner=False)
def load_oof(key):
    f = paths.RESULTS / key / "oof_predictions.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, dtype={"patient_id": str, "group_id": str})


@st.cache_data(show_spinner=False)
def load_unlabelled(key):
    f = paths.RESULTS / key / "unlabeled_predictions.csv"
    return pd.read_csv(f, dtype={"patient_id": str}) if f.exists() else None


@st.cache_data(show_spinner=False)
def load_tables():
    if not (paths.CACHE / "manifest.csv").exists():
        return None, None
    from bcmid.data import load_tables as lt
    return lt()


@st.cache_resource(show_spinner=False)
def load_arrays():
    return {m: np.load(paths.CACHE / f, mmap_mode="r") for m, f in (("MG", "mg.npy"), ("US", "us.npy"))}


@st.cache_resource(show_spinner="Loading the model ensemble ...")
def load_ensemble(variant):
    from bcmid.infer import Ensemble
    return Ensemble(variant)


def have_checkpoints(variant="full"):
    return any((paths.RUNS / variant).glob("fold[0-9].pt"))


# ------------------------------------------------------------------ drawing

def base_layout(fig, height=360, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.22),
                      hoverlabel=dict(font_size=12), **kw)
    fig.update_xaxes(showline=True, linewidth=1, gridwidth=1, zeroline=False)
    fig.update_yaxes(showline=False, gridwidth=1, zeroline=False)
    return fig


def fmt_ci(v, pct=False):
    s = 100 if pct else 1
    d = 1 if pct else 3
    return f"{v[0] * s:.{d}f} ({v[1] * s:.{d}f} to {v[2] * s:.{d}f})"


def overlay(gray, sal, strength=0.65):
    """Tint a grayscale image (H,W in [0,1]) with a single-hue saliency map."""
    import cv2
    s = cv2.resize(sal.astype(np.float32), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(s, 5), s.max()
    s = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) ** 1.5
    rgb = np.repeat(gray[..., None], 3, 2)
    tint = np.array([235, 104, 52]) / 255.0
    a = (s * strength)[..., None]
    return np.clip(rgb * (1 - a) + tint * a, 0, 1)


def show_images(result, images_u8, labels, show_saliency):
    for m, title in (("MG", "Mammography"), ("US", "Ultrasound")):
        if m not in result:
            continue
        r = result[m]
        st.markdown(f"**{title}** · {len(images_u8[m])} image(s) · unimodal probability "
                    f"{result['prob'][m]:.2f}")
        per_row = 6 if m == "MG" else 4
        order = np.argsort(-r["att"])
        for start in range(0, len(order), per_row):
            cols = st.columns(per_row)
            for col, j in zip(cols, order[start:start + per_row]):
                g = np.asarray(images_u8[m][j]).astype(np.float32) / 255.0
                img = overlay(g, r["sal"][j]) if (show_saliency and "sal" in r) else g
                col.image(img, clamp=True, width="stretch")
                col.caption(f"{labels[m][j]} · attention {r['att'][j]:.2f}")


def probability_bar(prob, thr, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[prob], y=[""], orientation="h", marker_color=BLUE, width=0.45,
                         hovertemplate="probability %{x:.3f}<extra></extra>"))
    fig.add_vline(x=thr, line_width=2, line_color=MUTED)
    fig.add_annotation(x=thr, y=0.5, yref="paper", text=f"threshold {thr:.2f}", showarrow=False, yshift=28,
                       font=dict(size=11, color=MUTED))
    base_layout(fig, height=120, title=title, showlegend=False)
    fig.update_xaxes(range=[0, 1], title=None)
    fig.update_yaxes(showticklabels=False)
    return fig


# ------------------------------------------------------------------ pages

def page_overview(summary):
    st.title("Mammography + ultrasound study classification with a ViT-UNet")
    st.caption("Research prototype trained on the public BCMID dataset (Zenodo 10.5281/zenodo.19402690). "
               "Not a medical device; not for clinical use.")
    if summary is None:
        st.warning("No results found. Run scripts 01-04 first (see README).")
        return
    ds = summary["dataset"]
    c = st.columns(5)
    c[0].metric("Study entries", ds["n_entries"], help=f"{ds['n_women']} distinct women")
    c[1].metric("Mammograms", ds["n_mg"])
    c[2].metric("Ultrasound frames", ds["n_us"])
    c[3].metric("Reference positive", ds["n_pos"], help="BI-RADS 4-6")
    c[4].metric("Reference negative", ds["n_neg"], help="BI-RADS 1-3")

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Reference standard")
        counts = ds["birads_counts"]
        cats = sorted(counts, key=int)
        group = ["Excluded (incomplete)" if int(b) == 0 else "Negative" if int(b) <= 3 else "Positive" for b in cats]
        col = {"Excluded (incomplete)": MUTED, "Negative": BLUE, "Positive": ORANGE}
        fig = go.Figure()
        for g in ("Negative", "Positive", "Excluded (incomplete)"):
            xs = [b for b, gg in zip(cats, group) if gg == g]
            fig.add_trace(go.Bar(x=[f"BI-RADS {b}" for b in xs], y=[counts[b] for b in xs], name=g, marker_color=col[g],
                                 text=[counts[b] for b in xs], textposition="outside",
                                 hovertemplate="%{x}: %{y} studies<extra>" + g + "</extra>"))
        base_layout(fig, height=330, title="Studies per BI-RADS category", bargap=0.25)
        fig.update_yaxes(title="studies")
        st.plotly_chart(fig, width="stretch")
        st.caption("The binary label is derived from the radiologist's BI-RADS category (1-3 negative, 4-6 positive); "
                   "it is not an independent histopathology result. BI-RADS 0 studies carry no binary label and are "
                   "excluded from training and evaluation.")
    with right:
        st.subheader("Headline result")
        if "full" in summary["models"]:
            m = summary["models"]["full"]["heads"]["fused"]
            a, b = st.columns(2)
            a.metric("AUROC (fusion)", f"{m['auroc'][0]:.3f}", help=f"95% CI {m['auroc'][1]:.3f} to {m['auroc'][2]:.3f}")
            b.metric("AUPRC (fusion)", f"{m['auprc'][0]:.3f}", help=f"95% CI {m['auprc'][1]:.3f} to {m['auprc'][2]:.3f}")
            a.metric("Sensitivity", f"{100 * m['sensitivity'][0]:.1f}%",
                     help=f"95% CI {100 * m['sensitivity'][1]:.1f} to {100 * m['sensitivity'][2]:.1f}")
            b.metric("Specificity", f"{100 * m['specificity'][0]:.1f}%",
                     help=f"95% CI {100 * m['specificity'][1]:.1f} to {100 * m['specificity'][2]:.1f}")
            rows = [{"Input": HEADS[h], "AUROC (95% CI)": fmt_ci(v["auroc"]), "AUPRC (95% CI)": fmt_ci(v["auprc"])}
                    for h, v in summary["models"]["full"]["heads"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"Pooled out-of-fold predictions from 5-fold cross-validation grouped by woman "
                       f"(n = {ds['n_labelled']}); percentile bootstrap, 2,000 resamples. Operating point: Youden "
                       "threshold chosen on each fold's validation split.")
            meta = summary["models"].get("metadata_lr")
            if meta:
                v = meta["heads"]["fused"]["auroc"]
                st.warning(f"Read these numbers against the shortcut audit: a model that sees **no pixels**, only how "
                           f"each study was acquired, reaches AUROC {v[0]:.3f} ({v[1]:.3f} to {v[2]:.3f}) on the same "
                           "folds. See *Model performance*.", icon=":material/warning:")
        else:
            st.info("The proposed model has not been trained yet.")


def page_performance(summary):
    st.title("Model performance")
    avail = [k for k in MODELS if load_oof(k) is not None]
    if not avail:
        st.warning("No out-of-fold predictions found in results/.")
        return
    f1, f2, f3 = st.columns([1.4, 1, 1.4])
    key = f1.selectbox("Model", avail, format_func=lambda k: MODELS[k])
    head = f2.selectbox("Input", list(HEADS), format_func=lambda h: HEADS[h])
    mode = f3.radio("Operating point", ["Pre-specified protocol (per-fold validation threshold)", "Custom global threshold"],
                    horizontal=False)
    df = load_oof(key)
    y, p = df.label.to_numpy(), df[f"prob_{head}"].to_numpy()
    if mode.startswith("Custom"):
        thr = st.slider("Decision threshold on the calibrated probability", 0.01, 0.99, 0.5, 0.01)
        pred = (p >= thr).astype(int)
        st.caption("A threshold picked here is chosen on the test predictions themselves, so the resulting numbers are "
                   "exploratory, not an unbiased estimate.")
    else:
        pred = (p >= df[f"thr_{head}"].to_numpy()).astype(int)
    m = point_metrics(y, p, pred)

    c = st.columns(6)
    c[0].metric("AUROC", f"{m['auroc']:.3f}")
    c[1].metric("AUPRC", f"{m['auprc']:.3f}")
    c[2].metric("Sensitivity", f"{100 * m['sensitivity']:.1f}%")
    c[3].metric("Specificity", f"{100 * m['specificity']:.1f}%")
    c[4].metric("PPV", f"{100 * m['ppv']:.1f}%")
    c[5].metric("NPV", f"{100 * m['npv']:.1f}%")

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1), name="chance",
                                 hoverinfo="skip"))
        for h in HEADS:
            fpr, tpr, th = roc_curve(y, df[f"prob_{h}"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=HEADS[h], customdata=np.clip(th, 0, 1),
                                     line=dict(color=HEAD_COLOR[h], width=3 if h == head else 1.5),
                                     opacity=1 if h == head else 0.55,
                                     hovertemplate="FPR %{x:.2f} · TPR %{y:.2f} · threshold %{customdata:.2f}<extra>" + HEADS[h] + "</extra>"))
        fig.add_trace(go.Scatter(x=[1 - m["specificity"]], y=[m["sensitivity"]], mode="markers", name="operating point",
                                 marker=dict(size=11, color=HEAD_COLOR[head], line=dict(color="white", width=2))))
        base_layout(fig, title="ROC curve")
        fig.update_xaxes(title="1 - specificity", range=[-0.01, 1])
        fig.update_yaxes(title="sensitivity", range=[0, 1.01])
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = go.Figure()
        for h in HEADS:
            pr, rc, _ = precision_recall_curve(y, df[f"prob_{h}"])
            fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=HEADS[h], line_shape="vh",
                                     line=dict(color=HEAD_COLOR[h], width=3 if h == head else 1.5),
                                     opacity=1 if h == head else 0.55,
                                     hovertemplate="recall %{x:.2f} · precision %{y:.2f}<extra>" + HEADS[h] + "</extra>"))
        fig.add_hline(y=y.mean(), line_width=1, line_color=MUTED, annotation_text=f"prevalence {y.mean():.2f}",
                      annotation_position="top right")
        base_layout(fig, title="Precision-recall curve")
        fig.update_xaxes(title="recall", range=[0, 1])
        fig.update_yaxes(title="precision", range=[0, 1.02])
        st.plotly_chart(fig, width="stretch")

    a, b, c2 = st.columns([0.8, 1, 1.2])
    with a:
        st.markdown("**Confusion matrix**")
        cm = pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                          index=["Reference negative", "Reference positive"], columns=["Predicted negative", "Predicted positive"])
        st.dataframe(cm, width="stretch")
        st.caption(f"Balanced accuracy {100 * m['balanced_accuracy']:.1f}% · F1 {m['f1']:.3f} · "
                   f"Brier {m['brier']:.3f} · ECE {m['ece']:.3f}")
    with b:
        edges = np.linspace(0, 1, 6)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 4)
        xs = [p[idx == k].mean() for k in range(5) if (idx == k).any()]
        ys = [y[idx == k].mean() for k in range(5) if (idx == k).any()]
        ns = [int((idx == k).sum()) for k in range(5) if (idx == k).any()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1), hoverinfo="skip",
                                 name="perfect"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=HEADS[head], customdata=ns,
                                 line=dict(color=HEAD_COLOR[head], width=2),
                                 marker=dict(size=9, line=dict(color="white", width=2)),
                                 hovertemplate="predicted %{x:.2f} · observed %{y:.2f} · n=%{customdata}<extra></extra>"))
        base_layout(fig, height=330, title="Calibration (5 equal-width bins)", showlegend=False)
        fig.update_xaxes(title="mean predicted probability", range=[0, 1])
        fig.update_yaxes(title="observed fraction positive", range=[-0.02, 1.02])
        st.plotly_chart(fig, width="stretch")
    with c2:
        rng = np.random.default_rng(0)
        fig = go.Figure()
        for lab, name, col in ((0, "Reference negative", BLUE), (1, "Reference positive", ORANGE)):
            g = df[df.label == lab]
            fig.add_trace(go.Scatter(x=g.birads + rng.uniform(-0.22, 0.22, len(g)), y=g[f"prob_{head}"], mode="markers",
                                     name=name, marker=dict(size=6, color=col, opacity=0.6), customdata=g.patient_id,
                                     hovertemplate="study %{customdata} · probability %{y:.2f}<extra>" + name + "</extra>"))
        base_layout(fig, height=330, title="Predicted probability by BI-RADS category")
        fig.update_xaxes(title="BI-RADS category", tickvals=sorted(df.birads.unique()))
        fig.update_yaxes(title="predicted probability", range=[-0.02, 1.02])
        st.plotly_chart(fig, width="stretch")

    if summary:
        st.subheader("All models (fusion input, pre-specified protocol)")
        rows = []
        for k, mm in summary["models"].items():
            v, cmp_ = mm["heads"]["fused"], summary["comparisons"].get(k)
            rows.append({"Model": mm["name"], "AUROC (95% CI)": fmt_ci(v["auroc"]), "AUPRC (95% CI)": fmt_ci(v["auprc"]),
                         "Fold AUROC mean ± SD": f"{v['fold_auroc_mean']:.3f} ± {v['fold_auroc_sd']:.3f}",
                         "Sens. %": round(100 * v["sensitivity"][0], 1), "Spec. %": round(100 * v["specificity"][0], 1),
                         "ΔAUROC vs proposed (95% CI)": "reference" if cmp_ is None else
                         f"{-cmp_['delta']:+.3f} ({-cmp_['hi']:+.3f} to {-cmp_['lo']:+.3f})",
                         "p": "" if cmp_ is None else f"{cmp_['p']:.3f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        keys = list(summary["models"])
        fig = go.Figure()
        for k in keys:
            pt, lo, hi = summary["models"][k]["heads"]["fused"]["auroc"]
            fig.add_trace(go.Scatter(x=[pt], y=[MODELS[k]], mode="markers", showlegend=False,
                                     marker=dict(size=11, color=BLUE if k == "full" else MUTED, line=dict(color="white", width=2)),
                                     error_x=dict(type="data", symmetric=False, array=[hi - pt], arrayminus=[pt - lo],
                                                  color=BLUE if k == "full" else MUTED, thickness=2, width=0),
                                     hovertemplate=f"{MODELS[k]}<br>AUROC {pt:.3f} (95% CI {lo:.3f} to {hi:.3f})<extra></extra>"))
        base_layout(fig, height=70 + 42 * len(keys), title="AUROC with 95% bootstrap confidence interval")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="AUROC")
        st.plotly_chart(fig, width="stretch")
        page_audit(summary)
        with st.expander("Table view of out-of-fold predictions"):
            st.dataframe(df[["patient_id", "fold", "birads", "label", "prob_fused", "prob_MG", "prob_US", "thr_fused"]],
                         hide_index=True, width="stretch")


def page_audit(summary):
    """Shortcut audit: how much of the label follows from the way a study was acquired."""
    conf = summary.get("confounders")
    if not conf:
        return
    st.subheader("Shortcut audit")
    st.markdown("In BCMID the label is partly predictable from **how a study was acquired**, without looking at any "
                "pixel. The *metadata only* model below uses just the number of images per modality, the mammography "
                "system and whether ultrasound files carry a side tag. An image model has to beat it, and has to keep "
                "working inside subsets where these characteristics are constant.")
    a, b = st.columns([1, 1.5])
    with a:
        rows = [{"Characteristic": x["name"],
                 "Positive if present": f"{x['pos_with']}/{x['n_with']} ({100 * x['pos_with'] / max(x['n_with'], 1):.0f}%)",
                 "Positive if absent": f"{x['pos_without']}/{x['n_without']} ({100 * x['pos_without'] / max(x['n_without'], 1):.0f}%)",
                 "Fisher p": "<0.001" if x["fisher_p"] < 0.001 else f"{x['fisher_p']:.3f}"} for x in conf["associations"]]
        st.markdown("**Association with the label**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with b:
        cols = {"full:fused": "ViT-UNet MIL", "dinov2_lr:fused": "DINOv2 + LR", "metadata_lr:fused": "Metadata only"}
        cols = {k: v for k, v in cols.items() if any(k in r["auroc"] for r in conf["subgroups"])}
        rows = []
        for r in conf["subgroups"]:
            row = {"Subset": r["name"], "Studies (positive)": f"{r['n']} ({r['n_pos']})"}
            for k, name in cols.items():
                v = r["auroc"].get(k)
                row[name] = "-" if v is None else f"{v[0]:.2f} ({v[1]:.2f} to {v[2]:.2f})"
            rows.append(row)
        st.markdown("**AUROC (95% CI) inside acquisition-homogeneous subsets, fusion input**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    palette = {"full:fused": BLUE, "dinov2_lr:fused": AQUA, "metadata_lr:fused": ORANGE}
    fig = go.Figure()
    names = [r["name"] for r in conf["subgroups"]]
    for k, name in cols.items():
        pts = [r["auroc"].get(k) for r in conf["subgroups"]]
        keep = [i for i, v in enumerate(pts) if v is not None]
        fig.add_trace(go.Scatter(
            x=[pts[i][0] for i in keep], y=[names[i] for i in keep], mode="markers", name=name,
            marker=dict(size=10, color=palette[k], line=dict(color="white", width=2)),
            error_x=dict(type="data", symmetric=False, array=[pts[i][2] - pts[i][0] for i in keep],
                         arrayminus=[pts[i][0] - pts[i][1] for i in keep], color=palette[k], thickness=1.5, width=0),
            hovertemplate="%{y}<br>AUROC %{x:.3f}<extra>" + name + "</extra>"))
    fig.add_vline(x=0.5, line_width=1, line_color=MUTED)
    base_layout(fig, height=110 + 46 * len(names), title="AUROC by subset (0.5 = chance)", scattermode="group",
                scattergap=0.45)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="AUROC", range=[0.4, 1.0])
    st.plotly_chart(fig, width="stretch")


def study_images(pid, manifest, arrays):
    images, labels = {}, {}
    for m in ("MG", "US"):
        sub = manifest[(manifest.patient_id == pid) & (manifest.modality == m)].sort_values("idx")
        images[m] = np.stack([np.asarray(arrays[m][i]) for i in sub.idx]) if len(sub) else None
        labels[m] = [f"{r.laterality} {r.view}" if m == "MG" else f"{r.laterality}" for r in sub.itertuples()]
    return images, labels


def page_cases(summary):
    st.title("Case explorer")
    manifest, patients = load_tables()
    df = load_oof("full")
    if manifest is None or df is None or not have_checkpoints():
        st.warning(f"This page needs the image cache and checkpoints in `{paths.WORK}` and the results of the proposed model.")
        return
    df = df.copy()
    df["pred"] = (df.prob_fused >= df.thr_fused).astype(int)
    df["outcome"] = np.select([(df.pred == 1) & (df.label == 1), (df.pred == 0) & (df.label == 0),
                               (df.pred == 1) & (df.label == 0)], ["True positive", "True negative", "False positive"],
                              "False negative")
    unl = load_unlabelled("full")
    f1, f2, f3 = st.columns([1, 1, 1.4])
    outcome = f1.selectbox("Outcome", ["All", "True positive", "False negative", "False positive", "True negative",
                                       "No reference label (BI-RADS 0)"])
    birads = f2.selectbox("BI-RADS", ["All"] + sorted(df.birads.unique().tolist()))
    if outcome.startswith("No reference"):
        pool = unl.patient_id.tolist() if unl is not None else []
    else:
        sel = df if outcome == "All" else df[df.outcome == outcome]
        sel = sel if birads == "All" else sel[sel.birads == birads]
        pool = sel.sort_values("prob_fused", ascending=False).patient_id.tolist()
    if not pool:
        st.info("No study matches these filters.")
        return
    pid = f3.selectbox(f"Study ({len(pool)} match, sorted by predicted probability)", pool)
    show_sal = st.toggle("Overlay decoder saliency map", value=True)

    images, labels = study_images(pid, manifest, load_arrays())
    ens = load_ensemble("full")
    row = df[df.patient_id == pid]
    if len(row):
        r = row.iloc[0]
        res = ens.predict(images, folds=[int(r.fold)])
        thr = float(r.thr_fused)
        st.markdown(f"**Study {pid}** · reference BI-RADS {int(r.birads)} "
                    f"({'positive' if r.label == 1 else 'negative'}) · {r.outcome.lower()} · "
                    f"scored by the fold-{int(r.fold)} model, which never saw this woman during training")
    else:
        res = ens.predict(images)
        thr = res["threshold"]["fused"]
        st.markdown(f"**Study {pid}** · BI-RADS 0 (incomplete, no reference label) · mean of {len(res['folds'])} fold models")
    st.plotly_chart(probability_bar(res["prob"]["fused"], thr, "Probability of a BI-RADS 4-6 assessment (fusion)"),
                    width="stretch")
    show_images(res, images, labels, show_sal)
    st.caption("Images are ordered by MIL attention weight (their share in the study-level decision). " + SALIENCY_NOTE)


def page_predict():
    st.title("Predict a new study")
    st.warning("Research use only. The model was trained on 327 studies from a single centre and has not been "
               "externally validated. It expects images exported like those in BCMID: inverted-greyscale mammography "
               "films and ultrasound scanner screenshots.", icon=":material/warning:")
    if not have_checkpoints():
        st.info(f"No checkpoints found in `{paths.RUNS / 'full'}`. Train the model first (scripts/02_train_cv.py).")
        return
    from bcmid.infer import preprocess_upload
    a, b = st.columns(2)
    up = {"MG": a.file_uploader("Mammogram view(s)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], accept_multiple_files=True),
          "US": b.file_uploader("Ultrasound frame(s)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], accept_multiple_files=True)}
    if not (up["MG"] or up["US"]):
        st.info("Upload at least one image. With both modalities the fusion head is used; with one, its unimodal head.")
        return
    images, labels = {}, {}
    for m in ("MG", "US"):
        ok, names = [], []
        for f in up[m] or []:
            try:
                ok.append(preprocess_upload(f.getvalue(), m))
                names.append(f.name[:22])
            except Exception as exc:
                st.error(f"Could not read {f.name}: {exc}")
        images[m] = np.stack(ok) if ok else None
        labels[m] = names
    if images["MG"] is None and images["US"] is None:
        return
    ens = load_ensemble("full")
    with st.spinner("Running the 5-model ensemble ..."):
        res = ens.predict(images)
    head = "fused" if "fused" in res["prob"] else next(iter(res["prob"]))
    prob, thr = res["prob"][head], res["threshold"][head]
    members = res["prob_members"][head]
    c = st.columns(3)
    c[0].metric(f"Probability ({HEADS[head]})", f"{prob:.2f}")
    c[1].metric("Ensemble range", f"{min(members):.2f} to {max(members):.2f}", help="Lowest and highest of the fold models")
    c[2].metric("Above mean decision threshold", "Yes" if prob >= thr else "No", help=f"Mean validation-selected threshold {thr:.2f}")
    st.plotly_chart(probability_bar(prob, thr, "Probability of a BI-RADS 4-6 assessment"), width="stretch")
    show_images(res, images, labels, st.toggle("Overlay decoder saliency map", value=True))
    st.caption("Shown after the same preprocessing used in training (breast or scan-area crop, intensity "
               "normalisation). " + SALIENCY_NOTE)


def page_methods(summary):
    st.title("Method and limitations")
    cfg = (summary or {}).get("config", {})
    hw = cfg.get("input_hw", {})
    st.markdown(f"""
**Task.** Study-level binary classification: BI-RADS 4-6 (positive) versus BI-RADS 1-3 (negative), from the
mammography views and ultrasound frames of one breast study.

**Model.** A frozen DINOv2 ViT-S/14 trunk is shared by two modality branches. Each branch fine-tunes the last
transformer blocks and adds a UNETR-style U-Net decoder that takes skip connections from intermediate ViT layers.
The decoder has two heads: one reconstructs randomly hidden image patches (a self-supervised auxiliary task), the
other produces a saliency map that weights the pooling of the ViT tokens. Image embeddings are combined per study
with gated-attention multiple-instance pooling, and the two modality embeddings are fused by a small MLP.
Input sizes: mammography {hw.get('MG', '-')}, ultrasound {hw.get('US', '-')}.

**Evaluation.** Five-fold cross-validation, stratified by label and grouped by woman (a few women have one entry
per breast). In every fold, 20% of the training studies are held out for epoch selection, Platt calibration and
the decision threshold; the test fold is used once. Confidence intervals are percentile bootstraps over studies.

**Limitations.**
- Much of the label is predictable from how a study was acquired and filed, without any pixel (number of images,
  mammography system, file-name tags). A metadata-only model matches the image model on the pooled data; the image
  model is clearly better only inside acquisition-homogeneous subsets. See *Model performance → Shortcut audit*.
- The reference label is the radiologist's BI-RADS category, not histopathology.
- One centre, 327 labelled studies, 63 positives: confidence intervals are wide and there is no external test set.
- Ultrasound frames contain burned-in calipers and annotations placed by the sonographer, which can act as a
  shortcut for the classifier.
- Mammograms are heavily downsampled (to 448 x 224 pixels); microcalcifications are not resolvable.
- The decoder's saliency maps are nearly uniform over the tissue. They are not lesion maps and have not been
  validated against lesion annotations, which BCMID does not provide.
""")
    if cfg:
        with st.expander("Training configuration"):
            st.json(cfg)


# ------------------------------------------------------------------ app

summary = load_summary()
PAGES = {"Overview": lambda: page_overview(summary), "Model performance": lambda: page_performance(summary),
         "Case explorer": lambda: page_cases(summary), "Predict a new study": page_predict,
         "Method and limitations": lambda: page_methods(summary)}
with st.sidebar:
    st.markdown("### BCMID ViT-UNet")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Dataset: BCMID, Seddik Tawfik et al., Zenodo 2026, CC BY 4.0.")
    st.caption("Research prototype. Not for clinical use.")
PAGES[page]()
