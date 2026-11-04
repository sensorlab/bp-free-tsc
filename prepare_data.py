"""Converts the UCR Time Series Classification Archive (2018) into the .npy splits read by bpfree.

Download UCRArchive_2018.zip from https://www.cs.ucr.edu/~eamonn/time_series_data_2018/, then run
either of

    python prepare_data.py /path/to/UCRArchive_2018.zip
    python prepare_data.py /path/to/UCRArchive_2018        # already unpacked

A zip archive is extracted to a temporary directory, with the archive's published password
if it is encrypted, and a single top-level folder is flattened (as in the original pipeline).

For each of the 112 datasets, the official train/test split is written to data/raw_splits/ as
{Name}_X_train.npy, {Name}_y_train.npy, {Name}_X_test.npy and {Name}_y_test.npy
(X: float32 of shape (N, L); y: int64 labels). The conversion follows the preprocessing of the
paper (fix_labels_and_drop_nans): rows with missing values are dropped, label -1 is mapped to 0,
and otherwise, if no test label is 0, the minimum test label is subtracted. The script stops if
the resulting labels are not exactly 0..C-1. The dataset metadata are read from
data/DataSummary.csv, which is part of the repository.
"""

import argparse
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from bpfree import SPLITS_DIR
from experiments.common import DATASETS


def fix_labels_and_drop_nans(train_path: Path, test_path: Path):
    """Port of the original preprocessing step (ucr_benchmark_template fix_labels_and_drop_nans)."""
    df_train = pd.read_csv(train_path, sep="\t", header=None).dropna()
    df_test = pd.read_csv(test_path, sep="\t", header=None).dropna()
    if df_train.empty or df_test.empty:
        raise ValueError(f"after dropping NaNs, {train_path.parent.name} train or test is empty")

    y_train = df_train.iloc[:, 0]
    y_test = df_test.iloc[:, 0]
    unique = y_test.unique()
    if -1 in unique:
        y_train = y_train.replace({-1: 0})
        y_test = y_test.replace({-1: 0})
    elif 0 not in unique:
        min_label = min(unique)
        y_train = y_train - min_label
        y_test = y_test - min_label

    return {
        "X_train": df_train.iloc[:, 1:].to_numpy(dtype=np.float32),
        "y_train": y_train.to_numpy(dtype=np.int64),
        "X_test": df_test.iloc[:, 1:].to_numpy(dtype=np.float32),
        "y_test": y_test.to_numpy(dtype=np.int64),
    }


UCR_ZIP_PASSWORD = "someone"  # published with the archive


@contextmanager
def archive_dir(archive: Path):
    """Yield the directory that holds one folder per dataset."""
    if archive.is_dir():
        yield archive
        return
    if not zipfile.is_zipfile(archive):
        raise SystemExit(f"{archive} is neither a directory nor a zip archive")
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive) as zf:
        tmp = Path(tmp)
        try:
            zf.extractall(tmp)
        except RuntimeError:  # encrypted members
            zf.extractall(tmp, pwd=UCR_ZIP_PASSWORD.encode("utf-8"))
        top = list(tmp.iterdir())
        yield top[0] if len(top) == 1 and top[0].is_dir() else tmp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, help="UCRArchive_2018.zip or the unpacked directory")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="subset of datasets (default: all 112)")
    args = parser.parse_args()

    with archive_dir(args.archive) as root:
        convert(root, args.datasets)


def convert(root: Path, datasets):
    missing = [name for name in datasets if not (root / name / f"{name}_TRAIN.tsv").exists()]
    if missing:
        raise SystemExit(f"{root} is not the UCRArchive_2018 directory: no train file for {', '.join(missing[:5])}")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        arrays = fix_labels_and_drop_nans(root / name / f"{name}_TRAIN.tsv", root / name / f"{name}_TEST.tsv")
        labels = np.unique(np.concatenate([arrays["y_train"], arrays["y_test"]]))
        if not np.array_equal(labels, np.arange(len(labels))):
            raise SystemExit(f"{name}: labels after conversion are {labels.tolist()}, expected 0..{len(labels) - 1}")
        for key, value in arrays.items():
            path = SPLITS_DIR / f"{name}_{key}.npy"
            np.save(path, value)
        print(f"{name}: train {arrays['X_train'].shape}, test {arrays['X_test'].shape}, classes {len(labels)}")
    print(f"\nWrote {4 * len(datasets)} files to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
