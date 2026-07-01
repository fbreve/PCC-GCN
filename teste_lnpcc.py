# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:34:45 2026

@author: fbrev
"""

import numpy as np
from lnpcc import LabelNoisePCC  # Seu arquivo corrigido

# Dados sintéticos: 200 nós, 3 classes, 20% rotulados, 10% ruído
n_nodes, n_features, n_classes = 200, 16, 3
data = np.random.rand(n_nodes, n_features)

# Labels verdadeiros (clusterizados)
true_labels = np.repeat([0,1,2], [70, 65, 65])
np.random.shuffle(true_labels)

# Supervisão: 20% rotulados (-1 = não rotulado)
slabel = np.full(n_nodes, -1, dtype=int)
labeled_idx = np.random.choice(n_nodes, 40, replace=False)
slabel[labeled_idx] = true_labels[labeled_idx]

# Ruído: flip 10% dos rotulados
noisy_idx = np.random.choice(labeled_idx, 4, replace=False)
slabel[noisy_idx] = np.random.randint(0, n_classes, 4)

print("Rótulos supervisão (com ruído):", slabel[:20])

# Teste LN-PCC
lnpcc = LabelNoisePCC(impl="auto")
lnpcc.build_graph(data, slabel, k_nn=10)
preds = lnpcc.fit_predict(slabel, n_repeats=5, uniform_labeled=True, dexp=2.0)

print(f"Backend usado: {lnpcc.impl}")
print("Preds primeiras 20:", preds[:20])
print("Acurácia vs verdadeiros:", np.mean(preds == true_labels))
print("Own-degree sample:\n", lnpcc.owndeg[:5])  # Confiança por classe

# Clusters reais (mimica Cora-like)
from sklearn.datasets import make_blobs
data, true_labels = make_blobs(n_samples=200, centers=3, n_features=16, random_state=42)
data = data / np.linalg.norm(data, axis=1, keepdims=True)  # Normalize

# 20% labeled + 10% noise
slabel = np.full(200, -1)
labeled_idx = np.random.choice(200, 40, replace=False)
slabel[labeled_idx] = true_labels[labeled_idx]
noisy_idx = np.random.choice(labeled_idx, 4)
slabel[noisy_idx] = 3 - slabel[noisy_idx]  # Flip classes

lnpcc = LabelNoisePCC(impl="auto")
lnpcc.build_graph(data, slabel, k_nn=8)  # k=8 pro paper
preds = lnpcc.fit_predict(slabel, p_grd=0.3, delta_v=0.05, n_repeats=20, dexp=1.5)

print(f"Acc blobs: {np.mean(preds == true_labels):.3f}")

