"""Converts raw runner outputs (outputs/*.csv) into the format of the published results/ files.

For every metric kept, the per-seed columns are averaged per dataset (mean, std with ddof=1).
Two summary rows are appended: AVERAGE (mean over datasets of every column) and
STD (standard deviation over datasets of the per-dataset means).

Usage:
    python -m experiments.summarize                     # all files found in outputs/
    python -m experiments.summarize --compute           # also keep FLOP/energy/memory columns
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bpfree import OUTPUT_DIR

BASE_METRICS = ["Accuracy", "Precision", "Recall", "F1", "TimeSeconds"]
COMPUTE_METRICS = ["TrainFlops", "TestFlops", "gpuEnergy(kwh)", "EnergyConsumed(kwh)", "Emissions(kg)", "PeakMemory"]
SEED_COL = re.compile(r"^(?P<prefix>.+?)_(?P<metric>" + "|".join(re.escape(m) for m in BASE_METRICS + COMPUTE_METRICS)
                      + r")_seed(?P<seed>\d+)$")

# raw output file -> (published file, raw column prefix, published column prefix, metrics)
F1_ONLY = ["F1"]
LAYOUT = {
    "DFA.csv":              ("DFA/DFA.csv",               "DFA",    "DFA",        BASE_METRICS),
    "DFA_BP.csv":           ("DFA/DFA_BP.csv",            "BP",     "BP",         BASE_METRICS),
    "DFA_original.csv":     ("DFA/DFA_original.csv",      "DFA",    "DFA",        BASE_METRICS),
    "DFA_GAP.csv":          ("DFA/DFA_GAP.csv",           "DFA_GAP", "DFA_GAP",  F1_ONLY),
    "DFA_GAP_BP.csv":       ("DFA/DFA_GAP_BP.csv",        "BP_GAP", "BP_GAP",     F1_ONLY),
    "DRTP.csv":             ("DRTP/DRTP.csv",             "DRTP",   "DRTP",       BASE_METRICS),
    "DRTP_BP.csv":          ("DRTP/DRTP_BP.csv",          "BP",     "BP",         BASE_METRICS),
    "DRTP_original.csv":    ("DRTP/DRTP_original.csv",    "DRTP",   "DRTP",       BASE_METRICS),
    "DRTP_GAP.csv":         ("DRTP/DRTP_GAP.csv",         "DRTP_GAP", "DRTP_GAP", F1_ONLY),
    "DRTP_GAP_BP.csv":      ("DRTP/DRTP_GAP_BP.csv",      "BP_GAP", "BP_GAP",     F1_ONLY),
    "NoProp.csv":           ("NoProp/NoProp.csv",         "NoProp", "NoProp",     BASE_METRICS),
    "NoProp_BP.csv":        ("NoProp/NoProp_BP.csv",      "NoProp", "NoProp",     BASE_METRICS),
    "NoProp_original.csv":  ("NoProp/NoProp_original.csv", "NoProp", "NoProp",    BASE_METRICS),
    "NoProp_GAP.csv":       ("NoProp/NoProp_GAP.csv",     "NoProp", "NoProp_GAP", F1_ONLY),
    "NoProp_GAP_BP.csv":    ("NoProp/NoProp_GAP_BP.csv",  "NoProp", "BP_GAP",     F1_ONLY),
}


def summarize(raw: pd.DataFrame, raw_prefix: str, out_prefix: str, metrics: list) -> pd.DataFrame:
    raw = raw[~raw["Dataset"].isin(["Average", "AVERAGE", "STD"])].set_index("Dataset")
    cols = {}
    for c in raw.columns:
        m = SEED_COL.match(c)
        if m and m["metric"] in metrics:
            if m["prefix"] != raw_prefix:
                raise ValueError(f"unexpected column prefix in {c!r} (expected {raw_prefix!r})")
            cols[(int(m["seed"]), m["metric"])] = c
    if not cols:
        raise ValueError("no per-seed metric columns found")
    seeds = sorted({s for s, _ in cols})
    metrics = [m for m in metrics if any((s, m) in cols for s in seeds)]

    out = pd.DataFrame(index=raw.index)
    for s in seeds:
        for m in metrics:
            if (s, m) in cols:
                out[f"{out_prefix}_{m}_seed{s}"] = pd.to_numeric(raw[cols[(s, m)]], errors="coerce")
    for m in metrics:
        per_seed = out[[f"{out_prefix}_{m}_seed{s}" for s in seeds if (s, m) in cols]]
        out[f"{out_prefix}_{m}_mean"] = per_seed.mean(axis=1)
        out[f"{out_prefix}_{m}_std"] = per_seed.std(axis=1, ddof=1)

    n_missing = int(out.filter(like="_seed").isna().sum().sum())
    if n_missing:
        print(f"  warning: {n_missing} missing per-seed values (failed runs); means use the remaining seeds")

    average = out.mean(axis=0)
    std = pd.Series(np.nan, index=out.columns)
    mean_cols = [c for c in out.columns if c.endswith("_mean")]
    std[mean_cols] = out[mean_cols].std(axis=0, ddof=1)
    out.loc["AVERAGE"] = average
    out.loc["STD"] = std
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "results")
    parser.add_argument("--compute", action="store_true", help="keep FLOP, energy and peak-memory columns")
    args = parser.parse_args()

    found = False
    for raw_name, (dst, raw_prefix, out_prefix, metrics) in LAYOUT.items():
        src = args.input_dir / raw_name
        if not src.exists():
            continue
        found = True
        if args.compute and metrics == BASE_METRICS:
            metrics = BASE_METRICS + COMPUTE_METRICS
        print(f"{src} -> {args.output_dir / dst}")
        table = summarize(pd.read_csv(src), raw_prefix, out_prefix, metrics)
        (args.output_dir / dst).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output_dir / dst, index=False)
    if not found:
        raise SystemExit(f"no runner outputs found in {args.input_dir}")


if __name__ == "__main__":
    main()
