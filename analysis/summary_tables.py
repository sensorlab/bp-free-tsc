"""Mean and standard deviation across datasets of the per-dataset (seed-averaged) metrics.

Reproduces the BP-free rows of the reference-architecture and optimized-model tables,
and the backpropagation-control training times of the computational-cost table.

Usage: python -m analysis.summary_tables
"""

from analysis.common import METHODS, column

METRICS = ["Accuracy", "Precision", "Recall", "F1", "TimeSeconds"]


def main():
    for config, title in [("original", "Reference architectures"),
                          ("native", "Optimized architectures, native rule"),
                          ("bp", "Optimized architectures, backpropagation control")]:
        print(f"\n{title} (mean / std across datasets)")
        print(f"{'Method':8s}" + "".join(f"{m:>22s}" for m in METRICS))
        for method in METHODS:
            cells = []
            for metric in METRICS:
                x = column(method, config, metric)
                fmt = "{:.2f} / {:.2f}" if metric == "TimeSeconds" else "{:.3f} / {:.3f}"
                cells.append(fmt.format(x.mean(), x.std(ddof=1)))
            print(f"{method:8s}" + "".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
