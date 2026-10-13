"""Step 4: metrics with confidence intervals, figures and LaTeX fragments."""
import os
import site
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if site.getusersitepackages() not in sys.path:      # only needed when run isolated (python -I)
    site.addsitedir(site.getusersitepackages())
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from bcmid import report

if __name__ == "__main__":
    s = report.run()
    for key, m in s["models"].items():
        for head, v in m["heads"].items():
            print(f"{key:12s} {head:6s} AUROC {v['auroc'][0]:.3f} [{v['auroc'][1]:.3f}, {v['auroc'][2]:.3f}]  "
                  f"AUPRC {v['auprc'][0]:.3f}  sens {v['sensitivity'][0]:.3f}  spec {v['specificity'][0]:.3f}  "
                  f"fold AUROC {v['fold_auroc_mean']:.3f}+-{v['fold_auroc_sd']:.3f}")
    for key, c in s["comparisons"].items():
        print(f"full - {key:12s} dAUROC {c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] p={c['p']:.3f}")
