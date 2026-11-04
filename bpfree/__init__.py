"""Backpropagation-free learning rules (DFA, DRTP, NoProp) for univariate time series classification."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BPFREE_DATA", ROOT / "data"))
SPLITS_DIR = DATA_DIR / "raw_splits"
SUMMARY_CSV = DATA_DIR / "DataSummary.csv"
OUTPUT_DIR = Path(os.environ.get("BPFREE_OUTPUT", ROOT / "outputs"))
