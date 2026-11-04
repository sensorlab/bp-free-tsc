from bpfree import SUMMARY_CSV, OUTPUT_DIR
from experiments.common import parse_args
from bpfree.train_yesprop_gap import (
    load_dataset, make_model, train, set_determinism
)

import argparse
import pandas as pd


cli = parse_args()
datasets = cli.datasets
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_name = "NoProp"
seeds = cli.seeds
    
summary_csv = SUMMARY_CSV
df_meta = pd.read_csv(summary_csv)

results_dict = {d: {'Dataset': d} for d in datasets}
model = None

for seed in seeds:
    for dataset in datasets:
        print(dataset)
        row = df_meta.loc[df_meta['Name'] == dataset]
        num_class = int(row['Class'].values[0])
        num_train = int(row['Train '].values[0])
        num_test = int(row['Test '].values[0])
        
        try:
            args = argparse.Namespace(
                    dataset=dataset,
                    epochs=cli.epochs,
                    batch_size=64,
                    seed=seed,
                    T = 10,
                    emb_d = 0,
                    eta = 0.7,
                    lr = 1e-3,
                    wd = 1e-3,
                    k_size = 17,
                    dropout = 0.1,
                    n_blocks = 6,
                    base_ch = 16,
                    n_layers = 5,
                    n_merged = 1,
                    base = 256,
                    count_flops=False,
                    measure_energy=False,
                    measure_memory=False,
            )
            
            set_determinism(args)

            trainloader, testloader = load_dataset(args)                    

            model = make_model(args)
            
            accuracy, precision, recall, f1, train_time, flops, energy, peak_mem = train(model, trainloader, testloader, args)
            print(accuracy, precision, recall, f1)
            
            results_dict[dataset][f'{model_name}_Accuracy_seed{seed}'] = accuracy
            results_dict[dataset][f'{model_name}_Precision_seed{seed}'] = precision
            results_dict[dataset][f'{model_name}_Recall_seed{seed}'] = recall
            results_dict[dataset][f'{model_name}_F1_seed{seed}'] = f1
            results_dict[dataset][f'{model_name}_TimeSeconds_seed{seed}'] = train_time

        except Exception as e:
            print(f"Error on dataset {dataset}, seed {seed}: {e}")

            results_dict[dataset][f'{model_name}_Accuracy_seed{seed}'] = None
            results_dict[dataset][f'{model_name}_Precision_seed{seed}'] = None
            results_dict[dataset][f'{model_name}_Recall_seed{seed}'] = None
            results_dict[dataset][f'{model_name}_F1_seed{seed}'] = None
            results_dict[dataset][f'{model_name}_TimeSeconds_seed{seed}'] = None


    pd.DataFrame(results_dict.values()).to_csv(OUTPUT_DIR / 'NoPropi_GAP_BP.csv', index=False)
   
df_results = pd.DataFrame(results_dict.values()).set_index('Dataset')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1','TimeSeconds']

for metric in metrics:
    seed_cols = [
        f'{model_name}_{metric}_seed{s}'
        for s in seeds
        if f'{model_name}_{metric}_seed{s}' in df_results.columns
    ]

    df_results[f'{model_name}_{metric}_mean'] = df_results[seed_cols].mean(axis=1, skipna=True)
    df_results[f'{model_name}_{metric}_std'] = df_results[seed_cols].std(axis=1, skipna=True)


df_results.loc['Average'] = df_results.mean(numeric_only=True)

df_results.to_csv(OUTPUT_DIR / 'NoProp_GAP_BP.csv', index_label='Dataset')
