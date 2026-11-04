"""Readout intervention: flatten vs global average pooling, difference-in-differences.

Per dataset: Delta_m = F1(m, GAP) - F1(m, flatten) for the native rule and its
backpropagation control, DD = Delta_native - Delta_BP. Each quantity is tested against
zero with the Wilcoxon signed-rank test, Holm-corrected across the three methods.

Usage: python -m analysis.readout_dd
"""

import numpy as np
from scipy.stats import wilcoxon

from analysis.common import METHODS, column


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def main():
    effects = {}
    for method in METHODS:
        d_native = column(method, "gap", "F1") - column(method, "native", "F1")
        d_bp = column(method, "gap_bp", "F1") - column(method, "bp", "F1")
        effects[method] = {"native": d_native, "bp": d_bp, "dd": d_native - d_bp}

    p_holm = {key: dict(zip(METHODS, holm([wilcoxon(effects[m][key]).pvalue for m in METHODS])))
              for key in ["native", "bp", "dd"]}

    print(f"{'Method':8s} {'mean D_native':>14s} {'deg.':>8s} {'p_Holm':>9s} "
          f"{'mean D_BP':>10s} {'deg.':>8s} {'p_Holm':>9s} {'DD':>8s} {'median DD':>10s} {'p_Holm':>9s}")
    for method in METHODS:
        e = effects[method]
        n = len(e["dd"])
        print(f"{method:8s} {e['native'].mean():+14.3f} {int((e['native'] < 0).sum()):>4d}/{n:<3d} {p_holm['native'][method]:9.1e} "
              f"{e['bp'].mean():+10.3f} {int((e['bp'] < 0).sum()):>4d}/{n:<3d} {p_holm['bp'][method]:9.1e} "
              f"{e['dd'].mean():+8.3f} {e['dd'].median():+10.3f} {p_holm['dd'][method]:9.1e}")


if __name__ == "__main__":
    main()
