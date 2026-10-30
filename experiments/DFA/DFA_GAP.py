from bpfree import SUMMARY_CSV, OUTPUT_DIR
from experiments.common import parse_args
import argparse
from bpfree import train_GAP
from bpfree import data as setup
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
                    loss='CE',
                    freeze_conv_layers=True,
                    fc_zero_init=False,
                    dropout=0.4,
                    trials=1,
                    epochs=cli.epochs,
                    batch_size=64,
                    test_batch_size=64,
                    lr=0.01,
                    topology='CONV_128_5_1_2_CONV_256_5_1_2_CONV_512_5_1_2_FC_1024' + "_FC_" + str(num_class),
                    conv_act='tanh',
                    hidden_act='sigmoid',
                    output_act='none',
                    seed=seed,
                    count_flops=False,
                    measure_energy=False,
                    measure_memory=False,
                )
                device, train_loader, traintest_loader, test_loader = setup.setup(args)
                    
                accuracy, precision, recall, f1, train_time, flops, energy, peak_mem = train_GAP.train(
                    args, device, train_loader, traintest_loader, test_loader
                )
                results_dict[dataset][f'{model}_GAP_F1_seed{seed}'] = f1
        
            except Exception as e:
                print(f"Error on dataset {dataset}, seed {seed}: {e}")
                results_dict[dataset][f'{model}_GAP_F1_seed{seed}'] = None

    pd.DataFrame(results_dict.values()).to_csv(OUTPUT_DIR / 'DFAi_GAP.csv', index=False)

df_results = pd.DataFrame(results_dict.values()).set_index('Dataset')
df_results.to_csv(OUTPUT_DIR / 'DFA_GAP.csv', index_label='Dataset')
