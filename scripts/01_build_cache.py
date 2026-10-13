"""Step 1: build the preprocessed image cache from BCMID.zip."""
import os
import site
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if site.getusersitepackages() not in sys.path:      # only needed when run isolated (python -I)
    site.addsitedir(site.getusersitepackages())
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import argparse

from bcmid import cache

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    args = ap.parse_args()
    cache.build(args.workers, args.limit)
    cache.contact_sheets()
