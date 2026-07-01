# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:38:05 2026

@author: fbrev
"""

import torch
import torch_geometric.datasets as datasets
from torch_geometric.utils import to_undirected
import numpy as np
from lnpcc import LabelNoisePCC

# Cora: 2708 nodes, 7 classes, 1433 feats
dataset = datasets.Planetoid(root='./tmp/Cora', name='Cora')
data = dataset[0]

# Features/labels como numpy
X = data.x.numpy()
y = data.y.numpy()  # 0-6 classes

# 5% labeled (20/140 train mask) + 15% noise
labeled_size = 140  # ~5%
labeled_idx = torch.where(data.train_mask)[0].numpy()[:labeled_size]
slabel = np.full(2708, -1)
slabel[labeled_idx] = y[labeled_idx]

# Ruído symmetric 15%
noise_rate = 0.15
noisy_n = int(labeled_size * noise_rate)
noisy_idx = np.random.choice(labeled_idx, noisy_n, replace=False)
slabel[noisy_idx] = np.random.randint(0, 7, noisy_n)

print(f"Cora: {len(labeled_idx)} labeled, {noisy_n} noisy")

from collections import defaultdict
edge_index = to_undirected(data.edge_index).cpu().numpy()

neib_dict = defaultdict(list)
for src, tgt in edge_index.T:
    neib_dict[src].append(tgt)

# Pad pra array uniforme (max_neib ~20 em Cora)
neib_lists = [np.array(neib_dict[i], dtype=np.int64) for i in range(data.num_nodes)]
max_neib = max(len(n) for n in neib_lists) if neib_lists else 0
neib_list_padded = np.full((data.num_nodes, max_neib), -1, dtype=np.int64)
neib_qt = np.zeros(data.num_nodes, dtype=np.int64)

for i, neibs in enumerate(neib_lists):
    neib_list_padded[i, :len(neibs)] = neibs
    neib_qt[i] = len(neibs)

print(f"Grafo padded: {neib_list_padded.shape}, qt mean: {neib_qt.mean():.1f}")

# LN-PCC!
lnpcc = LabelNoisePCC(impl="auto")
lnpcc.set_graph(neib_list_padded, neib_qt)
preds = lnpcc.fit_predict(slabel, p_grd=0.5, delta_v=0.1, n_repeats=20)

print(f"Backend: {lnpcc.impl}")
print(f"Acc total: {np.mean(preds == y):.3f}, labeled: {np.mean(preds[labeled_idx] == y[labeled_idx]):.3f}")
