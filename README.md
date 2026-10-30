# Backpropagation-free Deep Learning for Time-Series Classification

Code and results for the paper "Backpropagation-free Deep Learning for Time-Series Classification"
(M. Kaps, B. Bertalanič, C. Fortuna, Jožef Stefan Institute). The repository trains three
backpropagation-free learning rules, Direct Feedback Alignment (DFA), Direct Random Target
Projection (DRTP) and NoProp, with 1D convolutional architectures on 112 univariate datasets of
the UCR archive, together with backpropagation controls on identical architectures.

## Repository structure

```
bpfree/                 library
  data.py               dataset loading, per-seed resplitting, standardization
  function.py, module.py  DFA/DRTP training hooks (derived from the DRTP code base)
  models.py, models_GAP.py  DFA/DRTP/BP networks with flatten or global-average-pooling readout
  train.py, train_GAP.py    DFA/DRTP/BP training and evaluation
  train_noprop*.py      NoProp model and block-local training (flatten, reference, GAP)
  train_yesprop*.py     NoProp architecture trained end-to-end with backpropagation (flatten, GAP)
experiments/            one runner per configuration, shared CLI in common.py
  summarize.py          converts runner outputs to the format of results/
analysis/               scripts that compute the paper tables from results/
results/                published per-dataset results (6 seeds per dataset)
prepare_data.py         UCR archive -> data/raw_splits/*.npy
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run from the repository root.

## Data

Download the UCR Time Series Classification Archive 2018 (`UCRArchive_2018.zip`) from
https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ and run

```bash
python prepare_data.py /path/to/UCRArchive_2018.zip   # or the unpacked directory
```

This writes the official train/test split of each dataset to `data/raw_splits/`. Dataset
metadata (train/test size, length, number of classes) are read from `data/DataSummary.csv`,
which is included in the repository. The data location can be changed with the `BPFREE_DATA`
environment variable.

Seed 3 trains on the official split. Seeds 4 to 8 merge train and test and draw a stratified split
with the official test-set size (`bpfree/data.py`). Each series is standardized per time step
with training-set statistics.

## Running experiments

Each runner trains one configuration on all datasets and seeds (600 epochs, batch size 64):

```bash
python -m experiments.DFA.DFA
python -m experiments.NoProp.NoProp_GAP --datasets FordA Wafer --seeds 3 4 --epochs 600
```

| Runner | Configuration | Paper |
|---|---|---|
| `{DFA,DRTP,NoProp}_original` | architecture of the original publications | reference-architecture table |
| `{DFA,DRTP,NoProp}` | optimized architecture, native rule | optimized-model table, appendix |
| `{DFA,DRTP,NoProp}_BP` | optimized architecture, backpropagation control | computational-cost table, readout analysis |
| `{DFA,DRTP,NoProp}_GAP` | native rule, global-average-pooling readout | readout analysis |
| `{DFA,DRTP,NoProp}_GAP_BP` | backpropagation control, global-average-pooling readout | readout analysis |

`NoProp_BP` and `NoProp_GAP_BP` train the NoProp blocks end-to-end with backpropagation and
cross-entropy on the final output. Between blocks they apply a learned gate,
`z = tanh(g_t) p + (1 - tanh(g_t)) z`, instead of the fixed noise schedule of NoProp
(`bpfree/train_yesprop.py`).

Runners write raw per-seed results to `outputs/` (`BPFREE_OUTPUT` overrides the location).
Failed runs are reported as `Error on dataset ...` and stored as empty values. To convert the raw
files into the format of `results/` (per-dataset mean and std over seeds, plus `AVERAGE` and `STD`
rows over datasets):

```bash
python -m experiments.summarize          # writes outputs/results/
```

### Computational cost

FLOPs (`torch.utils.flop_counter`), energy and CO2 (`codecarbon`) and peak GPU memory are disabled
by default because the measurements add overhead to the training time. Enable them by setting
`count_flops`, `measure_energy` and `measure_memory` to `True` in the runner's argument namespace,
run the configuration separately, and summarize with `python -m experiments.summarize --compute`.
The FLOP counter reports the FLOPs of the last training epoch, which the runners divide by the
number of training (test) instances. The published experiments ran on a single NVIDIA A100 80GB.

## Results

`results/{DFA,DRTP,NoProp}/` contains one CSV per configuration with per-seed accuracy, macro
precision, macro recall, macro F1 and training time, the per-dataset mean and standard deviation
over the six seeds, an `AVERAGE` row (mean over datasets) and an `STD` row (standard deviation over
datasets of the per-dataset means). GAP files contain F1 only. `NoProp_BP.csv` keeps the column
prefix `NoProp_` written by its runner.

```bash
python -m analysis.summary_tables        # mean / std across datasets per configuration
python -m analysis.readout_dd            # flatten vs GAP readout, difference-in-differences
```