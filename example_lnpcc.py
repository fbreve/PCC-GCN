# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 16:18:26 2026

@author: fbrev
"""

# example_lnpcc.py
# Exemplo de uso do LN-PCC com rótulos ruidosos

import time
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import os
import sys

# adiciona ../pcc ao sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PCC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "pypcc"))
if PCC_DIR not in sys.path:
    sys.path.insert(0, PCC_DIR)

from pcc import ParticleCompetitionAndCooperation
from lnpcc import LabelNoisePCC

K_NN = 10
P_GRD = 0.5
DELTA_V = 0.1
MAX_ITER = 500000
EARLY_STOP = True
ES_CHK = 2000
LABELED_FRACTION = 0.1      # fração de exemplos inicialmente rotulados
NOISE_FRACTION = 0.3        # fração dos rótulos rotulados que será corrompida
N_REPEATS_LNPCC = 10        # número de repetições no LN-PCC
DEXP = 1.0                  # expoente da distância no LN-PCC
UNIFORM_LABELED = True


def make_ssl_labels(y, labeled_fraction, rng):
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_labeled = int(n * labeled_fraction)
    labeled_idx = idx[:n_labeled]
    unlabeled_idx = idx[n_labeled:]
    y_ssl = -1 * np.ones_like(y)
    y_ssl[labeled_idx] = y[labeled_idx]
    return y_ssl, labeled_idx, unlabeled_idx


def corrupt_labels(y, labeled_idx, noise_fraction, n_classes, rng):
    """
    Corrompe uma fração dos rótulos em labeled_idx.
    """
    y_noisy = y.copy()
    n_labeled = len(labeled_idx)
    n_noise = int(n_labeled * noise_fraction)

    if n_noise == 0:
        return y_noisy

    idx_noise = rng.choice(labeled_idx, size=n_noise, replace=False)
    for i in idx_noise:
        true_label = y_noisy[i]
        other_labels = [c for c in range(n_classes) if c != true_label]
        y_noisy[i] = rng.choice(other_labels)
    return y_noisy


def main():
    print("Loading the Wine dataset...")
    data = load_wine()
    X = data.data.astype(np.float64)
    y_true = data.target.astype(np.int64)
    n_classes = len(np.unique(y_true))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rng = np.random.RandomState(42)

    # gera cenário semi-supervisionado
    print("Generating semi-supervised setting...")
    y_ssl, labeled_idx, unlabeled_idx = make_ssl_labels(
        y_true, LABELED_FRACTION, rng
    )

    # cria versão ruidosa dos rótulos (apenas nos rotulados)
    print(f"Corrupting {NOISE_FRACTION*100:.1f}% of labeled data...")
    y_noisy = corrupt_labels(y_ssl, labeled_idx, NOISE_FRACTION, n_classes, rng)

    # === PCC “clássico” com rótulos ruidosos ===
    print("\nRunning PCC with noisy labels (baseline)...")
    pcc = ParticleCompetitionAndCooperation(impl="cython")  # "numba", "cython", or "numpy"
    pcc.build_graph(X, k_nn=K_NN)

    t0 = time.perf_counter()
    y_pred_pcc = pcc.fit_predict(
        y_noisy, p_grd=P_GRD, delta_v=DELTA_V,
        max_iter=MAX_ITER, early_stop=EARLY_STOP, es_chk=ES_CHK
    )
    t1 = time.perf_counter()
    time_pcc = t1 - t0

    acc_pcc_all = accuracy_score(y_true, y_pred_pcc)
    acc_pcc_unl = accuracy_score(y_true[unlabeled_idx], y_pred_pcc[unlabeled_idx])

    print(f"PCC time: {time_pcc:.3f}s")
    print(f"PCC accuracy (all):      {acc_pcc_all:.4f}")
    print(f"PCC accuracy (unlabeled): {acc_pcc_unl:.4f}")

    # === LN-PCC com mesmo cenário ruidoso ===
    print("\nRunning LN-PCC with noisy labels...")
    lnpcc = LabelNoisePCC(impl="cython")  # "numba", "cython", or "numpy"
    # build_graph usa rótulos “observados” (ruidosos) slabel:
    lnpcc.build_graph(X, slabel=y_noisy, k_nn=K_NN)

    t0 = time.perf_counter()
    y_pred_lnpcc = lnpcc.fit_predict(
        labels=y_noisy,
        p_grd=P_GRD,
        delta_v=DELTA_V,
        dexp=DEXP,
        max_iter=MAX_ITER,
        early_stop=EARLY_STOP,
        es_chk=ES_CHK,
        n_repeats=N_REPEATS_LNPCC,
        uniform_labeled=UNIFORM_LABELED,
    )
    t1 = time.perf_counter()
    time_lnpcc = t1 - t0

    acc_lnpcc_all = accuracy_score(y_true, y_pred_lnpcc)
    acc_lnpcc_unl = accuracy_score(y_true[unlabeled_idx], y_pred_lnpcc[unlabeled_idx])

    print(f"LN-PCC time: {time_lnpcc:.3f}s")
    print(f"LN-PCC accuracy (all):      {acc_lnpcc_all:.4f}")
    print(f"LN-PCC accuracy (unlabeled): {acc_lnpcc_unl:.4f}")

    # relatório em cima dos não rotulados (mais próximo do cenário de interesse)
    print("\nClassification report (unlabeled, PCC):")
    print(classification_report(y_true[unlabeled_idx], y_pred_pcc[unlabeled_idx]))

    print("Classification report (unlabeled, LN-PCC):")
    print(classification_report(y_true[unlabeled_idx], y_pred_lnpcc[unlabeled_idx]))


if __name__ == "__main__":
    main()
