"""Calibration, thresholds and evaluation metrics with bootstrap confidence intervals."""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit_platt(logit, y):
    """Platt scaling with the original smoothed targets; returns (a, b) for p = sigmoid(a*z + b)."""
    logit, y = np.asarray(logit, float), np.asarray(y, float)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    t = np.where(y > 0.5, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def nll(p):
        q = np.clip(sigmoid(p[0] * logit + p[1]), 1e-9, 1 - 1e-9)
        return -(t * np.log(q) + (1 - t) * np.log(1 - q)).sum()

    b0 = np.log((n_pos + 1) / (n_neg + 1))
    res = minimize(nll, x0=[1.0, b0], method="L-BFGS-B", bounds=[(1e-3, 50), (-50, 50)])
    return float(res.x[0]), float(res.x[1])


def youden_threshold(prob, y):
    fpr, tpr, thr = roc_curve(y, prob)
    k = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[k], 0.0, 1.0))


def ece(prob, y, bins=10):
    prob, y = np.asarray(prob), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(out)


def point_metrics(y, prob, pred=None):
    """Threshold-free metrics, plus operating-point metrics when hard predictions are given."""
    y, prob = np.asarray(y).astype(int), np.asarray(prob, float)
    out = {"auroc": roc_auc_score(y, prob), "auprc": average_precision_score(y, prob),
           "brier": float(np.mean((prob - y) ** 2)), "ece": ece(prob, y)}
    if pred is not None:
        pred = np.asarray(pred).astype(int)
        tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        out.update(sensitivity=sens, specificity=spec, balanced_accuracy=(sens + spec) / 2,
                   accuracy=(tp + tn) / len(y), ppv=ppv, npv=npv,
                   f1=2 * tp / max(2 * tp + fp + fn, 1), tp=tp, tn=tn, fp=fp, fn=fn)
    return out


def bootstrap_ci(y, prob, pred=None, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over studies. Returns {metric: (point, lo, hi)}."""
    y, prob = np.asarray(y).astype(int), np.asarray(prob, float)
    pred = None if pred is None else np.asarray(pred).astype(int)
    point = point_metrics(y, prob, pred)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point if k not in ("tp", "tn", "fp", "fn")}
    n = len(y)
    done = 0
    while done < n_boot:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        m = point_metrics(y[i], prob[i], None if pred is None else pred[i])
        for k in draws:
            draws[k].append(m[k])
        done += 1
    out = {}
    for k, v in draws.items():
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = (float(point[k]), float(lo), float(hi))
    for k in ("tp", "tn", "fp", "fn"):
        if k in point:
            out[k] = point[k]
    return out


def paired_bootstrap_delta(y, prob_a, prob_b, metric="auroc", n_boot=2000, seed=0):
    """Difference metric(a) - metric(b) on the same resampled studies; two-sided bootstrap p-value."""
    y, pa, pb = np.asarray(y).astype(int), np.asarray(prob_a, float), np.asarray(prob_b, float)
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    point = fn(y, pa) - fn(y, pb)
    rng = np.random.default_rng(seed)
    d, n = [], len(y)
    while len(d) < n_boot:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d.append(fn(y[i], pa[i]) - fn(y[i], pb[i]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": float(point), "lo": float(lo), "hi": float(hi), "p": float(min(1.0, p))}
