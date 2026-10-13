"""Step 2: cross-validated training of one model variant."""
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

from bcmid import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=sorted(train.VARIANTS))
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    over = {} if args.epochs is None else {"epochs": args.epochs}
    train.run_cv(args.variant, args.folds, log=lambda s: print(s, flush=True), **over)
