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

 "train.py" - Initializing the network, optimizer and loss for training and testing.
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback:
             Fixed random learning signals allow for feedforward training of deep neural networks,"
             Frontiers in Neuroscience, vol. 15, no. 629892, 2021. doi: 10.3389/fnins.2021.629892

 Modified by M. Kaps, B. Bertalanic and C. Fortuna, Jozef Stefan Institute, 2026,
 for "Backpropagation-free Deep Learning for Time-Series Classification". Changes:
   - training loop for the global-average-pooling models
   - package-relative imports

------------------------------------------------------------------------------
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from . import models_GAP
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch.utils.flop_counter import FlopCounterMode
import time
from codecarbon import EmissionsTracker
import gc
import os
import random
import numpy as np
import torch

def train(args, device, train_loader, traintest_loader, test_loader):

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    for trial in range(1,args.trials+1):
        # Network topology
        model = models_GAP.NetworkBuilder(args.topology, input_size=args.input_size, input_channels=args.input_channels, label_features=args.label_features, train_batch_size=args.batch_size, train_mode=args.train_mode, dropout=args.dropout, conv_act=args.conv_act, hidden_act=args.hidden_act, output_act=args.output_act, fc_zero_init=args.fc_zero_init, loss=args.loss, device=device)
        
        if args.cuda:
            model.cuda()
        
        # Optimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        elif args.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, eps=1e-8)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")
        
        # Loss function
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")
        
        #print("\n\n=== Starting model training with %d epochs:\n" % (args.epochs,))        

        train_flops = 0
        test_flops = 0 
        
        gpu_energy_kwh = 0
        energy_consumed_kwh = 0
        emissions_kg = 0
        
        peak_memory = 0
        
        if args.measure_energy:
            tracker = EmissionsTracker()
            tracker.start()
            
        if args.measure_memory:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        
        start = time.perf_counter()
        
        # TRAIN
        for _ in range(1, args.epochs + 1):    
            if args.count_flops:
                with FlopCounterMode(display=False) as flop_counter:
                    train_epoch(args, model, device, train_loader, optimizer, loss)
                train_flops = flop_counter.get_total_flops()
            else:
                train_epoch(args, model, device, train_loader, optimizer, loss)    

        train_time = time.perf_counter() - start

        
        if args.measure_memory:
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

        if args.measure_energy:
            torch.cuda.synchronize()
            tracker.stop()
            gpu_energy_kwh = tracker.final_emissions_data.gpu_energy
            energy_consumed_kwh = tracker.final_emissions_data.energy_consumed
            emissions_kg = tracker.final_emissions_data.emissions

        
        # TEST  
        if args.count_flops:
            with FlopCounterMode(display=False) as flop_counter:
                accuracy, precision, recall, f1 = test_epoch(args, model, device, test_loader, loss, 'Test')
            test_flops = flop_counter.get_total_flops()
        else:
            accuracy, precision, recall, f1 = test_epoch(args, model, device, test_loader, loss, 'Test')

        flops = (train_flops, test_flops)
        energy = (gpu_energy_kwh, energy_consumed_kwh, emissions_kg)
        #print("Test accuracy:")
        print(accuracy, precision, recall, f1)  
        
    return accuracy, precision, recall, f1, train_time, flops, energy, peak_memory

def train_epoch(args, model, device, train_loader, optimizer, loss):
    
    model.train()
    if args.freeze_conv_layers:
        for i in range(model.conv_to_fc):
            for param in model.layers[i].conv.parameters():
                param.requires_grad = False
    
    for _, (data, label) in enumerate(tqdm(train_loader, disable = True)):
        data, label = data.to(device), label.to(device)
        
        if args.regression:
            targets = label
        else:
            targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1), 1.0)

        
        optimizer.zero_grad()
        output = model(data, targets)
        loss_val = loss[0](output, loss[1](targets))
        loss_val.backward()
           
        optimizer.step()

        if args.measure_memory:
            del data, label, targets, output, loss_val
            gc.collect()
            torch.cuda.synchronize()

    return


def test_epoch(args, model, device, test_loader, loss, phase):
    model.eval()

    test_loss, correct = 0, 0
    len_dataset = len(test_loader.dataset)

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            if args.regression:
                targets = label
            else:
                targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1), 1.0)
            output = model(data, None)
            test_loss += loss[0](output, loss[1](targets), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            if not args.regression:
                correct += pred.eq(label.view_as(pred)).sum().item()
                all_preds.extend(pred.view(-1).cpu().numpy())
                all_labels.extend(label.cpu().numpy())
    
    loss = test_loss / len_dataset
    #print(f"Test loss: ", loss)
    precision, recall, f1 = 0, 0, 0
    if not args.regression:
        accuracy = correct / len_dataset
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return accuracy, precision, recall, f1
    