# -*- coding: utf-8 -*-
"""
Gridsearch LN-PCC Stage 3: Graph Fusion Strategies vs. k
Strategies:
  - 'u': Unused (Original Graph only)
  - 's': Same (Original + KNN edges with same labels)
  - 'd': Difference (Original + KNN edges NOT having different labels)
  - 'p': Pure (Original + All KNN edges)

@author: Fabricio Breve
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import multiprocessing
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
from lnpcc_graph import build_knn_graph, build_graph_from_edge_index, build_augmented_graph

# ============================================================
#  CONFIGURAÇÃO DO ESTÁGIO 3
# ============================================================
DATASETS        = ["Cora", "CiteSeer", "PubMed"]
NOISE_RATES     = [0.1, 0.2, 0.3, 0.4, 0.5]
STRATEGIES      = ["s", "d", "p", "u"]
K_VALUES        = [2, 5, 10, 15, 20, 30, 50, 75, 100]

# Best parameters found in previous stages
BEST_CONFIGS = {
    "Cora":     {"p_grd": 0.1, "dexp": 3.0, "rem_thresh": 0.1, "rel_thresh": 0.1},
    "CiteSeer": {"p_grd": 0.0, "dexp": 0.0, "rem_thresh": 0.1, "rel_thresh": 0.1},
    "PubMed":   {"p_grd": 0.4, "dexp": 2.0, "rem_thresh": 0.1, "rel_thresh": 0.1},
}

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
                
    if best_state: model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        test_acc = (pred[test_mask] == y_true[test_mask]).float().mean().item()
    return test_acc

# ============================================================
#  Utils
# ============================================================
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
    uncertainty = np.zeros(owndeg.shape[0])
    mask = top2[:, 1] > 0
    uncertainty[mask] = top2[mask, 0] / top2[mask, 1]
    return uncertainty

# ============================================================
#  Worker
# ============================================================
def run_one_grouped_stage3(dataset_name, noise_rate, seed,
                          x_np, ei_np, y_true_np, tm_np, vm_np, testm_np,
                          neib_list_edge, neib_qt_edge, 
                          knn_dict, config, gpu_semaphore):
    
    time.sleep(random.uniform(0, 0.5))
    rng = np.random.RandomState(seed)
    num_classes = int(y_true_np.max() + 1)
    
    x_torch = torch.from_numpy(x_np.copy()).float()
    edge_index_torch = torch.from_numpy(ei_np.copy()).long()
    y_true_torch = torch.from_numpy(y_true_np.copy()).long()
    tm_torch = torch.from_numpy(tm_np.copy()).bool()
    vm_torch = torch.from_numpy(vm_np.copy()).bool()
    testm_torch = torch.from_numpy(testm_np.copy()).bool()

    # 1. Induced Noise
    noisy_labels_np = flip_labels(y_true_np, tm_np, num_classes, noise_rate, rng)
    
    # 2. GCN Baseline
    gpu_semaphore.acquire()
    try:
        device_id = seed % len(GPU_IDS)
        device = torch.device(f"cuda:{GPU_IDS[device_id]}")
        y_noisy_torch = torch.from_numpy(noisy_labels_np).long()
        acc_baseline = train_gcn(x_torch, edge_index_torch, y_noisy_torch, y_true_torch, tm_torch, vm_torch, testm_torch, num_classes, seed, device)
    finally:
        gpu_semaphore.release()
    
    # 3. Ablation Loop
    results = []
    grid_points = []
    for s in STRATEGIES:
        if s == 'u': grid_points.append(('u', 0))
        else:
            for k in K_VALUES: grid_points.append((s, k))

    print(f"[{dataset_name} NR={noise_rate} Seed={seed}] Starting Stage 3 Job ({len(grid_points)} combinations)...")
    
    best_gain_in_job = -99.0
    
    for idx, (s, k) in enumerate(grid_points):
        # A. Build Graph
        # ... (graph building / LNPCC / cleaning logic remains same) ...
        if s == 'u':
            neib_aug, neib_qt_aug = build_augmented_graph(neib_list_edge, neib_qt_edge, None, None, noisy_labels_np, tm_np, strategy='u')
        else:
            neib_knn, neib_qt_knn = knn_dict[k]
            neib_aug, neib_qt_aug = build_augmented_graph(neib_list_edge, neib_qt_edge, neib_knn, neib_qt_knn, noisy_labels_np, tm_np, strategy=s)
        
        lnpcc = LabelNoisePCC(impl="cython")
        lnpcc.set_graph(neib_aug, neib_qt_aug)
        slabel = np.full(len(noisy_labels_np), -1, dtype=np.int64)
        slabel[tm_np] = noisy_labels_np[tm_np]
        preds = lnpcc.fit_predict(slabel, p_grd=config['p_grd'], dexp=config['dexp'], n_repeats=N_REPEATS, uniform_labeled=UNIFORM_LABELED).astype(np.int64)
        uncertainty = compute_uncertainty(lnpcc.owndeg)
        
        cleaned_labels_np = noisy_labels_np.copy()
        cleaned_mask_np = tm_np.copy()
        idx_train = np.where(tm_np)[0]
        for i in idx_train:
            u = uncertainty[i]
            if preds[i] == noisy_labels_np[i]:
                if u > config['rem_thresh']: cleaned_mask_np[i] = False
            else:
                if u <= config['rel_thresh']: cleaned_labels_np[i] = preds[i]
                else: cleaned_mask_np[i] = False
                
        gpu_semaphore.acquire()
        try:
            device_id = seed % len(GPU_IDS)
            device = torch.device(f"cuda:{GPU_IDS[device_id]}")
            y_clean_torch = torch.from_numpy(cleaned_labels_np).long()
            cm_torch = torch.from_numpy(cleaned_mask_np).bool()
            
            if cm_torch.sum() == 0: acc_cleaned = 0.0
            else: acc_cleaned = train_gcn(x_torch, edge_index_torch, y_clean_torch, y_true_torch, cm_torch, vm_torch, testm_torch, num_classes, seed, device)
        finally:
            gpu_semaphore.release()
            
        gain = acc_cleaned - acc_baseline
        if gain > best_gain_in_job:
            best_gain_in_job = gain
            
        results.append({
            "dataset": dataset_name, "noise_rate": noise_rate, "seed": seed,
            "strategy": s, "k": k if s != 'u' else 0, "gain": gain
        })

        if (idx + 1) % 5 == 0:
            print(f"[{dataset_name} NR={noise_rate} Seed={seed}] Progress: {idx+1}/{len(grid_points)} done. Best current gain: {best_gain_in_job*100:+.2f}%")
        
    print(f"[{dataset_name} NR={noise_rate} Seed={seed}] Job Completed. Max Gain: {best_gain_in_job*100:+.2f}%")
    return results

# ============================================================
#  Main Loop
# ============================================================
def main():
    t_start = datetime.now()
    manager = multiprocessing.Manager()
    gpu_sem = manager.Semaphore(len(GPU_IDS))
    
    for dataset_name in DATASETS:
        print(f"\n# Dataset: {dataset_name} - STAGE 3 #")
        ds = Planetoid(root=os.path.join(_THIS_DIR, "data", "Planetoid", dataset_name), name=dataset_name)
        data = ds[0]
        x_np, y_np = data.x.numpy(), data.y.numpy().astype(np.int64)
        tm_np, vm_np, testm_np = data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy()
        ei_np = data.edge_index.numpy()
        
        nl_edge, nq_edge = build_graph_from_edge_index(data.num_nodes, ei_np)
        sca = StandardScaler()
        x_std = sca.fit_transform(x_np)
        
        knn_graphs = {k: build_knn_graph(x_std, k_nn=k) for k in K_VALUES}
            
        jobs = list(product(NOISE_RATES, range(N_SEEDS)))
        print(f"  Starting {len(jobs)} Stage 3 jobs...")
        
        nested = Parallel(n_jobs=N_JOBS, verbose=10, backend="loky")(
            delayed(run_one_grouped_stage3)(
                dataset_name, nr, seed, x_np, ei_np, y_np, tm_np, vm_np, testm_np,
                nl_edge, nq_edge, knn_graphs, BEST_CONFIGS[dataset_name], gpu_sem
            )
            for nr, seed in jobs
        )
        
        all_res = [r for sub in nested for r in sub]
        df = pd.DataFrame(all_res)
        
        out_dir = os.path.join(_THIS_DIR, "results", "stage3", dataset_name.lower())
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(out_dir, f"lnpcc_stage3_{dataset_name}_{ts}.csv"), index=False)

    print(f"\nStage 3 completed in {datetime.now() - t_start}")

if __name__ == "__main__":
    main()
