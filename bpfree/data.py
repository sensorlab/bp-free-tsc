# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "data.py" - Setup configuration and dataset loading.
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

 Modified by M. Kaps, B. Bertalanic and C. Fortuna, Jozef Stefan Institute, 2026,
 for "Backpropagation-free Deep Learning for Time-Series Classification". Changes:
   - renamed from setup.py
   - UCR time series loading, per-seed stratified resplitting and per-time-step standardization
   - data paths resolved from bpfree.SPLITS_DIR / bpfree.SUMMARY_CSV

------------------------------------------------------------------------------
"""


import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import SPLITS_DIR, SUMMARY_CSV

def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        memory_load = get_gpu_memory_usage()
        cuda_device = np.argmin(memory_load).item()
        torch.cuda.set_device(cuda_device)
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
    
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    
    print(f"=== Loading the {args.dataset} dataset...")
    (train_loader, traintest_loader, test_loader) = load_dataset(args, kwargs)
        
    args.regression = (args.dataset == "regression_synth")
    
    return (device, train_loader, traintest_loader, test_loader)

def get_gpu_memory_usage():
    if sys.platform == "win32":
        curr_dir = os.getcwd()
        nvsmi_dir = r"C:\Program Files\NVIDIA Corporation\NVSMI"
        os.chdir(nvsmi_dir)
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
        os.chdir(curr_dir)
    else:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return gpu_memory


def load_dataset(args, kwargs):
    X_train = np.load(SPLITS_DIR / f"{args.dataset}_X_train.npy")
    y_train = np.load(SPLITS_DIR / f"{args.dataset}_y_train.npy")
    X_test = np.load(SPLITS_DIR / f"{args.dataset}_X_test.npy")
    y_test = np.load(SPLITS_DIR / f"{args.dataset}_y_test.npy")
    
    if args.seed > 3:
        test_length = X_test.shape[0]
        
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
    
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_length, stratify=y, random_state = args.seed
        )

    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) 
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) 
    y_test = torch.tensor(y_test, dtype=torch.long)

    df_meta = pd.read_csv(SUMMARY_CSV)
    row = df_meta.loc[df_meta['Name'] == args.dataset]
    
    if row.empty:
        raise ValueError(f"Dataset {args.dataset} not found in summary.")

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, drop_last = False, shuffle=True)
    traintest_loader =  DataLoader(TensorDataset(X_train, y_train), batch_size=args.test_batch_size, drop_last = False)
    
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.test_batch_size, drop_last = False)

    args.input_size     = int(row['Length'].values[0])
    args.input_channels = 1
    args.label_features = int(row['Class'].values[0])

    return (train_loader, traintest_loader, test_loader)