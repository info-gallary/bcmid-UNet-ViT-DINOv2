"""Filesystem layout.

Large artefacts (raw archive, image cache, checkpoints) live in a work directory
outside the project so they are not picked up by cloud-sync clients. Override
the location with the BCMID_WORK_DIR environment variable.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("BCMID_WORK_DIR", Path.home() / "bcmid_work"))

RAW_DIR = WORK / "raw"
RAW_ZIP = RAW_DIR / "BCMID.zip"
RAW_TSV = RAW_DIR / "BCMID_annotations.tsv"
CACHE = WORK / "processed"
RUNS = WORK / "runs"

RESULTS = Path(os.environ.get("BCMID_RESULTS_DIR", ROOT / "results"))
PAPER = Path(os.environ.get("BCMID_PAPER_DIR", ROOT / "paper"))
FIGURES = PAPER / "figures"
GENERATED = PAPER / "generated"
