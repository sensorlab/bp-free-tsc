import time
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from codecarbon import EmissionsTracker
from torch.amp import GradScaler
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from . import SPLITS_DIR, SUMMARY_CSV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def set_determinism(args):  
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True   
    torch.use_deterministic_algorithms(True, warn_only=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=========================
# NoProp Architecture
#=========================

class NoPropBlock(nn.Module):
    def __init__(self, input_length, emb_dimension, kernel_size, dpout, channels, fc_layers, merged_layers):
        super().__init__()

        ## Build convolutional feature extractor ##
        layers = []
        current_length = input_length
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ))
            
            layers.append(nn.ReLU())
            
            if current_length > 4:
                layers.append(nn.MaxPool1d(kernel_size=2))
                current_length //= 2
                
            layers.append(nn.Dropout(dpout))

        layers.append(nn.Flatten())
        self.conv_layers = nn.Sequential(*layers)

        ## Figure out flattened size ##
        with torch.no_grad():
            dummy = torch.zeros(1, channels[0], input_length)
            flat_size = self.conv_layers(dummy).shape[1]
            
        ## Fully connected layer for convolutional output ##
        self.fc_input = nn.Sequential(
            nn.Linear(flat_size, 256, bias = False),
            nn.BatchNorm1d(256),
            #nn.ReLU()  # try: with
        )
        
        ## First layer for residual connection ##
        self.first_layer = nn.Sequential(
            nn.Linear(fc_layers[0], fc_layers[1], bias = False), 
            nn.BatchNorm1d(fc_layers[1]),
            nn.ReLU()
        )
        
        ## Fully connected Layers ##
        layers = []
        for i in range(1, len(fc_layers) - 1):
            layers.append(nn.Linear(fc_layers[i], fc_layers[i+1], bias = False)) 
            layers.append(nn.BatchNorm1d(fc_layers[i+1]))
            if i < len(fc_layers) - 2:
                layers.append(nn.ReLU())
          
        self.fc_layers = nn.Sequential(*layers)
        
        ## Merged fully connected layers ##
        layers = []
        for i in range(len(merged_layers) - 1):
            layers.append(nn.Linear(merged_layers[i], merged_layers[i+1], bias = False)) 
            if i < (len(merged_layers) - 2):
                layers.append(nn.BatchNorm1d(merged_layers[i+1]))
                layers.append(nn.ReLU())

        self.fc_merged = nn.Sequential(*layers)

    def forward(self, x, z):
        fx = self.conv_layers(x)
        fx = self.fc_input(fx)
        fz1 = self.first_layer(z)
        fz = self.fc_layers(fz1)
        fz = fz + fz1
        fused = torch.cat([fx, fz], dim = -1)
        out = self.fc_merged(fused)
        return out


class NoPropModel(nn.Module):
    def __init__(self, num_blocks, input_length, emb_dimension, num_classes, kernels, dpout, channels, fc_layers, merged_layers):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.input_length = input_length

        ## Learnable embedding ##
        if emb_dimension != 0:
            self.W_embed = torch.nn.Parameter(torch.randn(num_classes, emb_dimension), requires_grad=True)
            self.emb_dimension = emb_dimension
        else:
            self.W_embed = nn.Parameter(torch.eye(num_classes, num_classes), requires_grad=False)
            self.emb_dimension = num_classes

        ## Blocks ##
        self.blocks = nn.ModuleList([
            NoPropBlock(input_length, self.emb_dimension, kernels, dpout, channels, fc_layers, merged_layers)
            for _ in range(num_blocks)
        ])      
    
        ## Final classifier ##
        self.out_head = nn.Linear(self.emb_dimension, num_classes) 

        alphas, alpha_hats = noise_schedule(num_blocks + 1)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)
        at = torch.sqrt(self.alpha_hats[1:]) * (1 - self.alphas[:-1]) / (1 - self.alpha_hats[:-1])
        bt = torch.sqrt(self.alphas[:-1]) * (1 - self.alpha_hats[1:]) / (1 - self.alpha_hats[:-1])
        ct = (1 - self.alpha_hats[1:]) * (1 - self.alphas[:-1]) / (1 - self.alpha_hats[:-1])
        self.register_buffer("at", at)
        self.register_buffer("bt", bt)
        self.register_buffer("ct", ct)
    
    def forward(self, x, z):   
        # Pass through all T blocks
        for i in range(self.num_blocks):
            block = self.blocks[i]
            logits = block(x, z) 
            p = F.softmax(logits, dim=-1) @ self.W_embed
            noise = torch.randn_like(z)
            z = self.at[i + 1] * p + self.bt[i + 1] * z + torch.sqrt(self.ct[i + 1]) * noise
        
        # Final prediction
        logits = self.out_head(z)

        return logits

#=========================
# Functions
#=========================

def noise_schedule(T, s=0.008):
    steps = torch.arange(0, T+1, dtype=torch.float32)
    alpha_hats = torch.cos((((steps / T) + s) / (1 + s)) * torch.pi / 2) ** 2
    alpha_hats = alpha_hats / alpha_hats[0]    
    
    alphas = torch.ones_like(alpha_hats)
    alphas[1:] = alpha_hats[1:] / alpha_hats[:-1]

    return torch.flip(alphas, dims = [0]), torch.flip(alpha_hats, dims = [0])

def snr(t, alpha_hats):
    return alpha_hats[t] / (1 - alpha_hats[t])

def sample(alpha_hats, t, uy):
    a = alpha_hats[t]
    mean = torch.sqrt(a) * uy
    std = torch.sqrt(1 - a)
    noise = torch.randn_like(uy)
    return mean + std * noise

def kl_divergence_term(uy, alpha_0):
    mu_q = torch.sqrt(alpha_0) * uy
    sigma2_q = 1.0 - alpha_0
    mu_norm2 = mu_q.pow(2).sum(dim=1)
    d = uy.shape[1]
    kl = 0.5 * (mu_norm2 + d * (sigma2_q - 1 - torch.log(sigma2_q)))
    return kl.mean()

#=========================
# Data
#=========================

def load_dataset(args):

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
    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, drop_last = False, shuffle = True)
    testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, drop_last = False)

    return trainloader, testloader

#=========================
# Model
#=========================

def make_model(args):
    summary_csv = SUMMARY_CSV
    df_meta = pd.read_csv(summary_csv)
    row = df_meta.loc[df_meta['Name'] == args.dataset]

    channels = [args.base_ch] * args.n_blocks
    fc_layers = [256] * args.n_layers
    fc_merged = [max(args.base // (2 ** i), 32) for i in range(args.n_merged)]
    
    if row.empty:
        raise ValueError(f"Dataset {args.dataset} not found in summary.")

    input_length = int(row['Length'].values[0])
    num_classes = int(row['Class'].values[0])

    if args.emb_d == 0:
        fc_layers = [num_classes] + fc_layers
    else:
        fc_layers = [args.emb_d] + fc_layers
    
    fc_layers = fc_layers + [256]

    fc_merged = [512] + fc_merged + [num_classes]
    
    if channels[0] != 1:
        channels = [1] + channels

    return NoPropModel(args.T, input_length, args.emb_d, num_classes, args.k_size, args.dropout, channels, fc_layers, fc_merged).to(device)


#=========================
# Train & Evaluate
#=========================

def train_epoch(model, trainloader, args, optimizer, scaler, alpha_hats, alpha_0):
    
    for t in range(1, args.T + 1): 
        for _, (data, labels) in enumerate(tqdm(trainloader, disable = True)):                     
            data, labels = data.to(device), labels.to(device) 

            optimizer.zero_grad()
                                
                                    
            # Collect embeddings
            uy = model.W_embed[labels]
                                    
            # Sample z_(t-1), z_T
            z_tm1 = sample(alpha_hats, t-1, uy)
            z_T = sample(alpha_hats, args.T, uy)
                                
            # Local forward pass  
            z_pred = model.blocks[t-1](data, z_tm1) # Indexed from 0, therefore t-1
            z_pred = F.softmax(z_pred, dim=-1)
            u_hat = z_pred @ model.W_embed
                                    
            # Compute local L2 loss
            diff_snr = snr(t, alpha_hats) - snr(t-1, alpha_hats)
            local_loss = diff_snr * F.mse_loss(u_hat, uy, reduction="mean") * (args.T / 2) * args.eta
            
            # Compute global CE loss
            logits = model.out_head(z_T)
            global_loss = F.cross_entropy(logits, labels)
            
            # Compute KL Divergence
            kl_loss = kl_divergence_term(uy, alpha_0)
                
            # Total loss
            loss = local_loss + global_loss + kl_loss

            if not torch.isfinite(loss):
                return
                        
            scaler.scale(loss).backward()
                
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
                
    return


def train(model, trainloader, testloader, args): 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd) 
    scaler = GradScaler(device="cuda")
    
    alpha_hats = model.alpha_hats 
    alpha_0 = alpha_hats[0]
    
    train_flops = 0
    test_flops = 0 
        
    gpu_energy_kwh = 0
    energy_consumed_kwh = 0
    emissions_kg = 0
        
    peak_mem = 0

    if args.measure_energy:
        tracker = EmissionsTracker()
        tracker.start()
            
    if args.measure_memory:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    
    start = time.time() 
    
    for _ in range(args.epochs): 
        model.train()

        if args.count_flops:
            with FlopCounterMode(display=False) as flop_counter:
                train_epoch(model, trainloader, args, optimizer, scaler, alpha_hats, alpha_0)      
            train_flops = flop_counter.get_total_flops()
        else:
            train_epoch(model, trainloader, args, optimizer, scaler, alpha_hats, alpha_0)      
            
    train_time = time.time() - start
    
    if args.measure_memory:
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    if args.measure_energy:
        torch.cuda.synchronize()
        tracker.stop()
        gpu_energy_kwh = tracker.final_emissions_data.gpu_energy
        energy_consumed_kwh = tracker.final_emissions_data.energy_consumed
        emissions_kg = tracker.final_emissions_data.emissions


    if args.count_flops:
        with FlopCounterMode(display=False) as flop_counter:
            y_true, y_pred = predict(model, testloader)
        test_flops = flop_counter.get_total_flops()
    else:
        y_true, y_pred = predict(model, testloader)
    
    metrics = evaluate(y_true, y_pred)

    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    flops = (train_flops, test_flops)
    energy = (gpu_energy_kwh, energy_consumed_kwh, emissions_kg)

    return accuracy, precision, recall, f1, train_time, flops, energy, peak_mem


def predict(model, testloader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            
            # Start z0 ~ N(0, 1)
            z = torch.randn(data.size(0), model.emb_dimension, device=device)

            logits = model(data, z)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return all_labels, all_preds



def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=1)),
        "recall": float(recall_score(y_true, y_pred, average="macro"))
    }