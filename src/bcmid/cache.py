"""Build the preprocessed image cache straight from BCMID.zip (no extraction)."""
import csv
import zipfile
from multiprocessing import Pool

import numpy as np

from . import paths
from . import preprocess as pp

_ZIP = None


def _work(member):
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(paths.RAW_ZIP)
    patient, modality, lat, view = pp.parse_name(member)
    try:
        img, meta = pp.preprocess_bytes(_ZIP.read(member), modality)
        err = ""
    except Exception as exc:                     # keep going; failures are reported
        hw = pp.MG_CACHE_HW if modality == "MG" else pp.US_CACHE_HW
        img, meta, err = np.zeros(hw, np.uint8), {}, repr(exc)
    row = dict(member=member, patient_id=patient, group_id=pp.group_id(patient), modality=modality,
               laterality=lat, view=view, error=err, **meta)
    return row, img


def read_annotations(tsv=None):
    rows = {}
    with open(tsv or paths.RAW_TSV, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            pid = r["Patient ID"].strip()
            binary = r["Binary Label"].strip()
            rows[pid] = dict(patient_id=pid, group_id=pp.group_id(pid), birads=int(r["BI-RADS Label"]),
                             category=r["Pathology Label"].strip() or "Incomplete",
                             label={"Positive": 1, "Negative": 0}.get(binary, -1))
    return rows


def build(workers=6, limit=None):
    import pandas as pd

    ann = read_annotations()
    with zipfile.ZipFile(paths.RAW_ZIP) as zf:
        members = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))
    if limit:
        members = members[:limit]
    paths.CACHE.mkdir(parents=True, exist_ok=True)

    rows, unreadable, imgs = [], [], {"MG": [], "US": []}
    with Pool(workers) as pool:
        for k, (row, img) in enumerate(pool.imap(_work, members, chunksize=4), 1):
            if row["error"]:                     # e.g. a zero-byte file in the archive: leave it out
                unreadable.append(row)
            else:
                row["idx"] = len(imgs[row["modality"]])
                imgs[row["modality"]].append(img)
                rows.append(row)
            if k % 200 == 0 or k == len(members):
                print(f"preprocessed {k}/{len(members)}", flush=True)

    man = pd.DataFrame(rows)
    man.to_csv(paths.CACHE / "manifest.csv", index=False)
    pd.DataFrame(unreadable, columns=["member", "patient_id", "modality", "error"]).to_csv(
        paths.CACHE / "unreadable.csv", index=False)
    np.save(paths.CACHE / "mg.npy", np.stack(imgs["MG"]))
    np.save(paths.CACHE / "us.npy", np.stack(imgs["US"]))

    counts = man.groupby(["patient_id", "modality"]).size().unstack(fill_value=0)
    pat = pd.DataFrame(ann.values())
    pat["n_mg"] = pat.patient_id.map(counts.get("MG", {})).fillna(0).astype(int)
    pat["n_us"] = pat.patient_id.map(counts.get("US", {})).fillna(0).astype(int)
    pat = pat.sort_values("patient_id").reset_index(drop=True)
    pat.to_csv(paths.CACHE / "patients.csv", index=False)

    missing = sorted(set(man.patient_id) - set(ann))
    print(f"images: MG={len(imgs['MG'])} US={len(imgs['US'])}  patients={len(pat)}  "
          f"women={pat.group_id.nunique()}  unreadable={[r['member'] for r in unreadable]}  "
          f"seg_fallbacks={(~man.seg_ok.astype(bool)).sum()}  inverted_mg={int(man.inverted.astype(bool).sum())}  "
          f"unannotated_folders={missing}")
    return man, pat


def contact_sheets(n_per_sheet=40, seed=0):
    """Thumbnail grids for visual QA of the preprocessing."""
    import cv2
    import pandas as pd

    man = pd.read_csv(paths.CACHE / "manifest.csv", dtype={"patient_id": str, "group_id": str})
    out = paths.CACHE / "qa"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for mod, fname, thumb, cols in (("MG", "mg.npy", (224, 112), 10), ("US", "us.npy", (126, 168), 8)):
        arr = np.load(paths.CACHE / fname, mmap_mode="r")
        sub = man[man.modality == mod]
        flagged = sub[~sub.seg_ok.fillna(False).astype(bool)].idx.tolist()
        pick = rng.choice(sub.idx.to_numpy(), size=min(n_per_sheet, len(sub)), replace=False).tolist()
        for tag, ids in (("random", pick), ("fallback", flagged[:n_per_sheet])):
            if not ids:
                continue
            tiles = [cv2.resize(np.asarray(arr[i]), (thumb[1], thumb[0]), interpolation=cv2.INTER_AREA) for i in ids]
            while len(tiles) % cols:
                tiles.append(np.zeros(thumb, np.uint8))
            grid = np.concatenate([np.concatenate(tiles[r:r + cols], 1) for r in range(0, len(tiles), cols)], 0)
            cv2.imwrite(str(out / f"{mod.lower()}_{tag}.png"), grid)
    print("QA sheets written to", out)
