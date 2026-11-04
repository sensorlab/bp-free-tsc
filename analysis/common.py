"""Loading helpers for the published result tables."""

from pathlib import Path

import pandas as pd

from bpfree import ROOT

RESULTS_DIR = ROOT / "results"
METHODS = ["DFA", "DRTP", "NoProp"]

# configuration -> (file name pattern, column prefix)
FILES = {
    "original": ("{m}_original.csv", "{m}"),
    "native":   ("{m}.csv",          "{m}"),
    "bp":       ("{m}_BP.csv",       "BP"),
    "gap":      ("{m}_GAP.csv",      "{m}_GAP"),
    "gap_bp":   ("{m}_GAP_BP.csv",   "BP_GAP"),
}


def load(method: str, config: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Per-dataset rows of one result file, indexed by dataset (summary rows dropped)."""
    name, _ = FILES[config]
    df = pd.read_csv(results_dir / method / name.format(m=method))
    return df[~df["Dataset"].isin(["AVERAGE", "STD"])].set_index("Dataset")


def column(method: str, config: str, metric: str, stat: str = "mean", results_dir: Path = RESULTS_DIR) -> pd.Series:
    """Per-dataset mean (or std) over seeds of one metric, e.g. column("DFA", "native", "F1")."""
    df = load(method, config, results_dir)
    _, prefix = FILES[config]
    prefix = prefix.format(m=method)
    if method == "NoProp" and config == "bp":
        prefix = "NoProp"  # the published NoProp backpropagation file keeps the runner's column prefix
    return df[f"{prefix}_{metric}_{stat}"].astype(float)
