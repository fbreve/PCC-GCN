# -*- coding: utf-8 -*-
"""
Gridsearch LN-PCC Stage 1: p_grd x d_exp
Fixed Thresholds: rem_thresh=0.5, rel_thresh=0.5
Metrics: Accuracy Gain Heatmaps (LNPCC+GCN vs. Baseline GCN).
Parallelized with GPU Semaphore.

@author: Fabricio Breve
"""

import sys
import os
import logging
import re
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import multiprocessing
import faulthandler
import time
import random
from datetime import datetime
from itertools import product
from joblib import Parallel, delayed
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from lnpcc import LabelNoisePCC

faulthandler.enable()

# ============================================================
#  CONFIGURAÇÃO PRINCIPAL
# ============================================================
DATASETS        = ["Cora", "CiteSeer", "PubMed"]
NOISE_RATES     = [0.1, 0.2, 0.3, 0.4, 0.5]
P_GRD_RANGE     = np.round(np.arange(0.0, 1.0, 0.1), 1)
DEXP_RANGE      = np.round(np.arange(0, 11, 1), 0)
FIXED_REM_THRESH = 0.5
FIXED_REL_THRESH = 0.5
N_SEEDS         = 10
N_REPEATS       = 10
DELTA_V         = 0.1
UNIFORM_LABELED = False
N_JOBS          = 16 
GPU_IDS         = [0, 1]

# GCN Hyperparams
GCN_HIDDEN = 16
GCN_LR     = 0.01
GCN_WD     = 5e-4
GCN_EPOCHS = 200

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

# ============================================================
#  GCN Model
# ============================================================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_gcn(x, edge_index, y_train, y_true, train_mask, val_mask, test_mask, num_classes, seed, device):
    torch.manual_seed(seed)
    model = GCN(x.size(1), GCN_HIDDEN, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GCN_LR, weight_decay=GCN_WD)
    
    best_val_acc = 0.0
    best_state = None
    
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    y_train = y_train.to(device)
    y_true = y_true.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    for _ in range(GCN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y_train[train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            val_acc = (pred[val_mask] == y_true[val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        test_acc = (pred[test_mask] == y_true[test_mask]).float().mean().item()
    return test_acc

# ============================================================
#  Utils
# ============================================================
def set_idle_priority():
    try:
        psutil.Process(os.getpid()).nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception: pass

def flip_labels(labels, mask, num_classes, noise_rate, rng):
    noisy = labels.copy()
    idx = np.where(mask)[0]
    n_flip = int(len(idx) * noise_rate)
    flip_idx = rng.choice(idx, size=n_flip, replace=False)
    for i in flip_idx:
        current = noisy[i]
        new_label = (current + rng.randint(1, num_classes)) % num_classes
        noisy[i] = new_label
    return noisy

def compute_uncertainty(owndeg):
    top2 = np.partition(owndeg, -2, axis=1)[:, -2:]
    max1 = top2[:, 1]
    max2 = top2[:, 0]
    uncertainty = np.zeros_like(max1)
    mask = max1 > 0
    uncertainty[mask] = max2[mask] / max1[mask]
    return uncertainty

# ============================================================
#  Worker
# ============================================================
def run_one_grouped_stage1(dataset_name, noise_rate, seed, 
                          p_grd_range, dexp_range,
                          x_np, edge_index_np, y_true_np, train_mask_np, val_mask_np, test_mask_np,
                          neib_padded_np, neib_qt_np, gpu_semaphore):
    
    time.sleep(random.uniform(0, 0.5))
    
    rng = np.random.RandomState(seed)
    num_classes = int(y_true_np.max() + 1)
    
    x_torch = torch.from_numpy(x_np.copy()).float()
    edge_index_torch = torch.from_numpy(edge_index_np.copy()).long()
    y_true_torch = torch.from_numpy(y_true_np.copy()).long()
    train_mask_torch = torch.from_numpy(train_mask_np.copy()).bool()
    val_mask_torch = torch.from_numpy(val_mask_np.copy()).bool()
    test_mask_torch = torch.from_numpy(test_mask_np.copy()).bool()

    # 1. Induce Noise
    noisy_labels_np = flip_labels(y_true_np, train_mask_np, num_classes, noise_rate, rng)
    
    # 2. GCN Baseline
    print(f"[{dataset_name} NR={noise_rate} Seed={seed}] Stage 1: Training Baseline...")
    gpu_semaphore.acquire()
    try:
        device_id = seed % len(GPU_IDS)
        device = torch.device(f"cuda:{GPU_IDS[device_id]}")
        y_noisy_torch = torch.from_numpy(noisy_labels_np).long()
        acc_baseline = train_gcn(x_torch, edge_index_torch, y_noisy_torch, y_true_torch, 
                                 train_mask_torch, val_mask_torch, test_mask_torch, 
                                 num_classes, seed, device)
    finally:
        gpu_semaphore.release()
    print(f"[{dataset_name} NR={noise_rate} Seed={seed}] Baseline Acc: {acc_baseline:.4f}")

    # 3. Grid Search Over p_grd and d_exp
    results = []
    grid_points = list(product(p_grd_range, dexp_range))
    best_gain_in_job = -1.0
    
    lnpcc = LabelNoisePCC(impl="cython")
    lnpcc.set_graph(neib_padded_np, neib_qt_np)
    
    slabel = np.full(len(y_true_np), -1, dtype=np.int64)
    slabel[train_mask_np] = noisy_labels_np[train_mask_np]

    for idx, (p_grd, dexp) in enumerate(grid_points):
        # A. Run LNPCC
        preds = lnpcc.fit_predict(
            slabel, p_grd=p_grd, dexp=dexp, delta_v=DELTA_V,
            n_repeats=N_REPEATS, uniform_labeled=UNIFORM_LABELED
        ).astype(np.int64)
        uncertainty = compute_uncertainty(lnpcc.owndeg)
        
        # B. Clean Labels (Using fixed thresholds)
        cleaned_labels_np = noisy_labels_np.copy()
        cleaned_mask_np = train_mask_np.copy()
        
        current_train_idx = np.where(train_mask_np)[0]
        for i in current_train_idx:
            u = uncertainty[i]
            if preds[i] == noisy_labels_np[i]:
                if u > FIXED_REM_THRESH: cleaned_mask_np[i] = False
            else:
                if u <= FIXED_REL_THRESH: cleaned_labels_np[i] = preds[i]
                else: cleaned_mask_np[i] = False
        
        # C. GCN Cleaned
        gpu_semaphore.acquire()
        try:
            device_id = seed % len(GPU_IDS)
            device = torch.device(f"cuda:{GPU_IDS[device_id]}")
            y_cleaned_torch = torch.from_numpy(cleaned_labels_np).long()
            cleaned_mask_torch = torch.from_numpy(cleaned_mask_np).bool()
            
            if cleaned_mask_torch.sum() == 0:
                acc_cleaned = 0.0
            else:
                acc_cleaned = train_gcn(x_torch, edge_index_torch, y_cleaned_torch, y_true_torch, 
                                        cleaned_mask_torch, val_mask_torch, test_mask_torch, 
                                        num_classes, seed, device)
        finally:
            gpu_semaphore.release()
            
        gain = acc_cleaned - acc_baseline
        if gain > best_gain_in_job: best_gain_in_job = gain
            
        results.append({
            "dataset": dataset_name, "noise_rate": noise_rate, "seed": seed,
            "p_grd": p_grd, "dexp": dexp, "gain": gain
        })
        
        if (idx + 1) % 50 == 0:
            print(f"[{dataset_name} NR={noise_rate} Seed={seed}] Stage 1 Progress: {idx+1}/{len(grid_points)} points...")
            
    print(f"[{dataset_name} NR={noise_rate} Seed={seed}] Stage 1 Done. Best Gain: {best_gain_in_job*100:+.2f}%")
    return results

# ============================================================
#  Main Loop
# ============================================================
def main():
    set_idle_priority()
    t_start = datetime.now()
    
    manager = multiprocessing.Manager()
    gpu_sem = manager.Semaphore(len(GPU_IDS))
    
    for dataset_name in DATASETS:
        print(f"\n# Dataset: {dataset_name} - STAGE 1 #")
        
        ds = Planetoid(root=os.path.join(_THIS_DIR, "data", "Planetoid", dataset_name), name=dataset_name)
        data = ds[0]
        x_np, y_np, tm_np, vm_np, testm_np, ei_np = data.x.numpy(), data.y.numpy().astype(np.int64), data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy(), data.edge_index.numpy()
        num_nodes = x_np.shape[0]
        
        edge_raw = torch.from_numpy(ei_np)
        from torch_geometric.utils import to_undirected
        edge_undir = to_undirected(edge_raw).numpy()
        neib_dict = defaultdict(list)
        for s, t in edge_undir.T: neib_dict[s].append(t)
        neib_lists = [np.array(neib_dict[i], dtype=np.int64) for i in range(num_nodes)]
        max_deg = max((len(n) for n in neib_lists), default=0)
        neib_padded = np.full((num_nodes, max_deg), -1, dtype=np.int64)
        neib_qt = np.zeros(num_nodes, dtype=np.int64)
        for i, ns in enumerate(neib_lists):
            neib_padded[i, :len(ns)] = ns
            neib_qt[i] = len(ns)
            
        noise_seed_pairs = list(product(NOISE_RATES, range(N_SEEDS)))
        print(f"Total Stage 1 jobs for {dataset_name}: {len(noise_seed_pairs)}")
        
        nested_results = Parallel(n_jobs=N_JOBS, verbose=10, backend="loky")(
            delayed(run_one_grouped_stage1)(
                dataset_name, nr, seed,
                P_GRD_RANGE, DEXP_RANGE,
                x_np, ei_np, y_np, tm_np, vm_np, testm_np,
                neib_padded, neib_qt, gpu_sem
            )
            for nr, seed in noise_seed_pairs
        )
        
        all_results = [res for sublist in nested_results for res in sublist]
        df = pd.DataFrame(all_results)
        
        out_dir = os.path.join(_THIS_DIR, "results", "stage1", dataset_name.lower())
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(out_dir, f"lnpcc_stage1_{dataset_name}_{ts}.csv"), index=False)

    print(f"\nStage 1 completed in {datetime.now() - t_start}")

if __name__ == "__main__":
    main()
