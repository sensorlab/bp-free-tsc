from bpfree import SUMMARY_CSV, OUTPUT_DIR
from experiments.common import parse_args
from bpfree import train
from bpfree import data as setup
import argparse
import pandas as pd

cli = parse_args()
datasets = cli.datasets
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_options = ['DFA']
seeds = cli.seeds

summary_csv = SUMMARY_CSV
df_meta = pd.read_csv(summary_csv)

# One row per dataset
results_dict = {d: {'Dataset': d} for d in datasets}

for seed in seeds:
    for dataset in datasets:
        
        row = df_meta.loc[df_meta['Name'] == dataset]
        num_class = int(row['Class'].values[0])
        num_train = int(row['Train '].values[0])
        num_test = int(row['Test '].values[0])
        
        for model in model_options:
            try:
                args = argparse.Namespace(
                    cpu=False,
                    dataset=dataset,
                    train_mode=model,
                    optimizer='NAG',
                    loss='BCE',
                    freeze_conv_layers=False,
                    fc_zero_init=False,
                    dropout=0,
                    trials=1,
                    epochs=cli.epochs,
                    batch_size=64,
                    test_batch_size=64,
                    lr=1e-4,
                    topology="CONV_32_5_1_2_FC_1000_" + "FC_" + str(num_class),
                    conv_act='tanh',
                    hidden_act='tanh',
                    output_act='sigmoid',
                    seed=seed,
                    count_flops=False,
                    measure_energy=False,
                    measure_memory=False,
                )
                
                device, train_loader, traintest_loader, test_loader = setup.setup(args)
                
                accuracy, precision, recall, f1, train_time, flops, energy, peak_mem = train.train(
                    args, device, train_loader, traintest_loader, test_loader
                )
                
                
                results_dict[dataset][f'{model}_Accuracy_seed{seed}'] = accuracy
                results_dict[dataset][f'{model}_Precision_seed{seed}'] = precision
                results_dict[dataset][f'{model}_Recall_seed{seed}'] = recall
                results_dict[dataset][f'{model}_F1_seed{seed}'] = f1
                results_dict[dataset][f'{model}_TimeSeconds_seed{seed}'] = train_time
                results_dict[dataset][f'{model}_TrainFlops_seed{seed}'] = flops[0] / num_train
                results_dict[dataset][f'{model}_TestFlops_seed{seed}'] = flops[1] / num_test
                results_dict[dataset][f'{model}_gpuEnergy(kwh)_seed{seed}'] = energy[0]
                results_dict[dataset][f'{model}_EnergyConsumed(kwh)_seed{seed}'] = energy[1]
                results_dict[dataset][f'{model}_Emissions(kg)_seed{seed}'] = energy[2]
                results_dict[dataset][f'{model}_PeakMemory_seed{seed}'] = peak_mem
                

            except Exception as e:
                print(f"Error on dataset {dataset}, seed {seed}: {e}")
                results_dict[dataset][f'{model}_Accuracy_seed{seed}'] = None
                results_dict[dataset][f'{model}_Precision_seed{seed}'] = None
                results_dict[dataset][f'{model}_Recall_seed{seed}'] = None
                results_dict[dataset][f'{model}_F1_seed{seed}'] = None
                results_dict[dataset][f'{model}_TimeSeconds_seed{seed}'] = None
                results_dict[dataset][f'{model}_TrainFlops_seed{seed}'] = None
                results_dict[dataset][f'{model}_TestFlops_seed{seed}'] = None
                results_dict[dataset][f'{model}_gpuEnergy(kwh)_seed{seed}'] = None
                results_dict[dataset][f'{model}_EnergyConsumed(kwh)_seed{seed}'] = None
                results_dict[dataset][f'{model}_Emissions(kg)_seed{seed}'] = None
                results_dict[dataset][f'{model}_PeakMemory_seed{seed}'] = None

    pd.DataFrame(results_dict.values()).to_csv(OUTPUT_DIR / 'DFAi_original.csv', index=False)

df_results = pd.DataFrame(results_dict.values()).set_index('Dataset')


metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'TimeSeconds', 'gpuEnergy(kwh)',
           'EnergyConsumed(kwh)', 'Emissions(kg)', 'PeakMemory']

for model in model_options:
    for metric in metrics:
        seed_cols = [f'{model}_{metric}_seed{s}' for s in seeds if f'{model}_{metric}_seed{s}' in df_results.columns]

        # Mean/std across seeds for each dataset
        df_results[f'{model}_{metric}_mean'] = df_results[seed_cols].mean(axis=1, skipna=True)
        df_results[f'{model}_{metric}_std']  = df_results[seed_cols].std(axis=1, skipna=True)


df_results.loc['Average'] = df_results.mean(numeric_only=True)

df_results.to_csv(OUTPUT_DIR / 'DFA_original.csv', index_label='Dataset')
