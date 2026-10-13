"""Image preprocessing for BCMID.

Mammograms are stored as inverted 8-bit films (white background) with burned-in
view markers and a black frame; ultrasound frames are scanner screenshots with
UI text around the scan area. Both are reduced to the anatomical region and
letterboxed to a fixed cache resolution.
"""
import re

import cv2
import numpy as np

MG_CACHE_HW = (896, 448)
US_CACHE_HW = (504, 672)

_MG_RE = re.compile(r"_(L|R)_(CC|MLO|UNK)(?:_|$)")
_US_RE = re.compile(r"_US_(L|R|UNK)(?:_|$)")
_ID_RE = re.compile(r"^(\d+y\d{2})1$")


def group_id(patient_id):
    """Map a study entry to the woman it belongs to.

    A few women have a second entry for the contralateral breast whose ID is the
    base ID with an extra trailing '1' (e.g. 01566y21 / 01566y211).
    """
    m = _ID_RE.match(patient_id)
    return m.group(1) if m else patient_id


def parse_name(member):
    """<patient>/<Modality>/<file>.png -> (patient, modality, laterality, view)."""
    parts = member.split("/")
    patient, folder, stem = parts[0], parts[1], parts[-1].rsplit(".", 1)[0]
    if folder.lower().startswith("mammo"):
        m = _MG_RE.search(stem)
        return patient, "MG", (m.group(1) if m else "UNK"), (m.group(2) if m else "UNK")
    m = _US_RE.search(stem)
    return patient, "US", (m.group(1) if m else "UNK"), "US"


def _largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == k


def _letterbox(img, out_hw, align_left=False):
    oh, ow = out_hw
    h, w = img.shape[:2]
    s = min(oh / h, ow / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    res = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((oh, ow), np.uint8)
    y0 = (oh - nh) // 2
    x0 = 0 if align_left else (ow - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = res
    return out


def is_inverted(small):
    """BCMID mixes inverted films (white background) with standard-polarity digital mammograms."""
    return bool((small > 200).mean() > (small < 40).mean())


def background_threshold(work, valid):
    """Tissue/background cut placed above the background peak of the grey-level histogram.

    The background is the largest uniform region, so it is the mode of the dark
    half of the histogram; the lower half-width of that peak (which tissue does
    not contaminate) gives its spread. Otsu's threshold is not used because
    films with a greyish, shaded background push it into the background.
    """
    vals = work[valid]
    if vals.size == 0:
        return 12.0
    hist = np.convolve(np.bincount(vals, minlength=256).astype(np.float64), np.ones(5) / 5, mode="same")
    mode = int(np.argmax(hist[:128]))
    below = vals[vals <= mode]
    sigma = float(mode - np.percentile(below, 16)) if below.size else 0.0
    return float(np.clip(mode + 3.0 * sigma + 10.0, 12, 110))


def frame_margins(work, thr, max_frac=0.04):
    """Width of the film-frame band on each side (top, bottom, left, right).

    Scanned films carry a grey frame along two edges that touches the breast at
    the chest wall. A border line belongs to the frame if it is mostly
    near-saturated, or if almost none of it is at background level.
    """
    hot, nonbg = work > 190, work > thr

    def run(h, n):
        k, n_max = 0, int(len(h) * max_frac)
        while k < n_max and (h[k] > 0.5 or n[k] > 0.95):
            k += 1
        return k + 1 if k else 0           # one extra line for the blurred transition

    hr, nr, hc, nc = hot.mean(1), nonbg.mean(1), hot.mean(0), nonbg.mean(0)
    return run(hr, nr), run(hr[::-1], nr[::-1]), run(hc, nc), run(hc[::-1], nc[::-1])


def preprocess_mammogram(gray, out_hw=MG_CACHE_HW):
    """Mammogram of either polarity -> breast-only image, bright tissue on black, chest wall left."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray[..., :3], cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    f = 8
    small = cv2.resize(gray, (W // f, H // f), interpolation=cv2.INTER_AREA)
    inverted = is_inverted(small)
    work = 255 - small if inverted else small
    valid = work < 247                     # black frame, marker plates and burned-in text are not tissue
    thr = background_threshold(work, valid)
    top = bottom = left = right = 0
    if inverted:                           # only scanned films have a frame
        top, bottom, left, right = frame_margins(work, thr)
        h_s, w_s = work.shape
        valid[:top] = False
        valid[h_s - bottom:] = False
        valid[:, :left] = False
        valid[:, w_s - right:] = False
    tissue = ((work > thr) & valid).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, k)
    comp = _largest_component(tissue)
    frac = 0.0 if comp is None else float(comp.mean())
    ok = comp is not None and 0.02 < frac < 0.92
    if ok:
        comp = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
        ys, xs = np.where(filled > 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mask = cv2.dilate(filled, k)
    else:
        y0, y1, x0, x1 = 0, small.shape[0], 0, small.shape[1]
        mask = valid.astype(np.uint8)

    Y0, Y1, X0, X1 = y0 * f, min(H, y1 * f), x0 * f, min(W, x1 * f)
    crop = gray[Y0:Y1, X0:X1]
    if inverted:
        crop = 255 - crop
    m = cv2.resize(mask[y0:y1, x0:x1], (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    inside = crop[m > 0]
    lo, hi = np.percentile(inside, [0.5, 99.5]) if inside.size else (0, 255)
    crop = np.clip((crop.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    crop = (crop * 255 * (m > 0)).astype(np.uint8)

    # chest wall = the side whose border columns carry more tissue
    edge = max(1, m.shape[1] // 8)
    flipped = bool(m[:, -edge:].sum() > m[:, :edge].sum())
    if flipped:
        crop = crop[:, ::-1]
    out = _letterbox(np.ascontiguousarray(crop), out_hw, align_left=True)
    meta = dict(orig_h=H, orig_w=W, y0=int(Y0), y1=int(Y1), x0=int(X0), x1=int(X1),
                flipped=flipped, inverted=inverted, seg_ok=bool(ok), fg_frac=round(frac, 4),
                frame=int(max(top, bottom, left, right) > 0))
    return out, meta


def preprocess_ultrasound(img, out_hw=US_CACHE_HW):
    """Scanner screenshot -> scan area only (UI text, scale bar and footer removed)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    H, W = gray.shape
    content = (gray > 12).astype(np.uint8)
    ks = max(9, int(round(min(H, W) * 0.04)) | 1)
    opened = cv2.morphologyEx(content, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
    comp = _largest_component(opened)
    frac = 0.0 if comp is None else float(comp.mean())
    ok = comp is not None and frac > 0.04
    if ok:
        ys, xs = np.where(comp)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    else:
        y0, y1, x0, x1 = 0, H, 0, W
    crop = np.ascontiguousarray(gray[y0:y1, x0:x1])
    out = _letterbox(crop, out_hw)
    meta = dict(orig_h=H, orig_w=W, y0=int(y0), y1=int(y1), x0=int(x0), x1=int(x1),
                flipped=False, inverted=False, seg_ok=bool(ok), fg_frac=round(frac, 4), frame=0)
    return out, meta


def decode(data):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("could not decode image")
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return arr


def preprocess_bytes(data, modality):
    arr = decode(data)
    return preprocess_mammogram(arr) if modality == "MG" else preprocess_ultrasound(arr)
