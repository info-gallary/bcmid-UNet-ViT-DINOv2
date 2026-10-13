"""Step 5: qualitative figure (decoder saliency on held-out studies) and preprocessing statistics."""
import os
import site
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
if site.getusersitepackages() not in sys.path:      # only needed when run isolated (python -I)
    site.addsitedir(site.getusersitepackages())
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from bcmid import qualitative

if __name__ == "__main__":
    qualitative.run()
