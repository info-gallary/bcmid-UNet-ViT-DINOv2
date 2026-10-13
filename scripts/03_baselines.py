"""Step 3: frozen-feature baselines (pretrained encoder + logistic regression)."""
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

from bcmid import baselines

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(baselines.KEYS), default=None)
    args = ap.parse_args()
    for key in ([args.only] if args.only else baselines.KEYS):
        baselines.run(key, log=lambda s: print(s, flush=True))
