"""Resumable multi-connection download of BCMID.zip from Zenodo (stdlib only).

Usage:
    python -I tools/download_bcmid.py <out_dir> [--workers 8]

The archive is fetched in fixed-size byte ranges that are written straight into
their final offsets, so an interrupted run resumes without re-fetching finished
chunks. The MD5 published by Zenodo is verified at the end.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

RECORD = "19402690"
URL = f"https://zenodo.org/api/records/{RECORD}/files/BCMID.zip/content"
SIZE = 12941046854
MD5 = "7b7799c3534050b9336dec773edeee3a"
CHUNK = 48 * 1024 * 1024


def fetch_chunk(idx, path, lock, done_bytes):
    start = idx * CHUNK
    end = min(start + CHUNK, SIZE) - 1
    want = end - start + 1
    for attempt in range(12):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            buf = bytearray()
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise IOError(f"expected 206, got {r.status}")
                while True:
                    part = r.read(1 << 18)
                    if not part:
                        break
                    buf += part
                    with lock:
                        done_bytes[0] += len(part)
            if len(buf) != want:
                with lock:
                    done_bytes[0] -= len(buf)
                raise IOError(f"short read {len(buf)}/{want}")
            with lock:
                with open(path, "r+b") as fh:
                    fh.seek(start)
                    fh.write(buf)
            return idx
        except (urllib.error.URLError, IOError, OSError, TimeoutError) as exc:
            wait = min(90, 3 * 2 ** attempt)
            code = getattr(exc, "code", None)
            if code == 429:
                wait = max(wait, 60)
            print(f"[chunk {idx}] attempt {attempt + 1} failed: {exc!r}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"chunk {idx} failed permanently")


TSV_URL = f"https://zenodo.org/api/records/{RECORD}/files/BCMID_annotations.tsv/content"
TSV_MD5 = "db9702b4578ee3dd92b0c06515941137"


def fetch_annotations(out_dir):
    path = os.path.join(out_dir, "BCMID_annotations.tsv")
    if not (os.path.exists(path) and hashlib.md5(open(path, "rb").read()).hexdigest() == TSV_MD5):
        with urllib.request.urlopen(TSV_URL, timeout=60) as r:
            data = r.read()
        if hashlib.md5(data).hexdigest() != TSV_MD5:
            raise IOError("annotation file checksum mismatch")
        with open(path, "wb") as fh:
            fh.write(data)
    print("annotations OK:", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fetch_annotations(args.out_dir)
    path = os.path.join(args.out_dir, "BCMID.zip")
    state_path = path + ".state.json"
    n_chunks = (SIZE + CHUNK - 1) // CHUNK

    done = set()
    if os.path.exists(state_path) and os.path.exists(path):
        done = set(json.load(open(state_path))["done"])
    else:
        with open(path, "wb") as fh:
            fh.truncate(SIZE)
    # Fetch the tail first: it holds the zip central directory.
    todo = [i for i in range(n_chunks) if i not in done]
    todo.sort(key=lambda i: (i != n_chunks - 1, i))
    print(f"{len(done)}/{n_chunks} chunks already done; fetching {len(todo)} with {args.workers} workers", flush=True)

    lock = threading.Lock()
    done_bytes = [0]
    t0 = time.time()
    last = [t0, 0]
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_chunk, i, path, lock, done_bytes) for i in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            done.add(fut.result())
            with lock:
                json.dump({"done": sorted(done)}, open(state_path, "w"))
            now = time.time()
            if now - last[0] > 60 or k == len(futs):
                rate = (done_bytes[0] - last[1]) / max(now - last[0], 1e-6) / 1e6
                remaining = (len(futs) - k) * CHUNK / 1e6
                eta = remaining / max(rate, 1e-6) / 60
                print(f"{len(done)}/{n_chunks} chunks  {rate:.2f} MB/s  ETA {eta:.0f} min", flush=True)
                last[0], last[1] = now, done_bytes[0]

    print("verifying md5 ...", flush=True)
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    ok = h.hexdigest() == MD5
    print(f"md5 {h.hexdigest()} {'OK' if ok else 'MISMATCH (expected ' + MD5 + ')'}", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
