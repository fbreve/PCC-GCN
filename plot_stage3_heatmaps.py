# -*- coding: utf-8 -*-
"""
Plot Stage 3 Heatmaps: k-NN Mode vs. k
Refined Version: Individual colorbars, compact layout.

@author: Fabricio Breve
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================
#  CONFIGURAÇÃO
# ============================================================
DATASETS = ["cora", "citeseer", "pubmed"]
BASE_RESULTS_DIR = r"results\stage3"
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

def plot_stage3(dataset_name):
    ds_dir = os.path.join(_THIS_DIR, BASE_RESULTS_DIR, dataset_name)
    if not os.path.exists(ds_dir):
        print(f"Directory not found: {ds_dir}")
        return

    pattern = os.path.join(ds_dir, f"lnpcc_stage3_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No CSV files found in {ds_dir}")
        return

    latest_csv = files[-1]
    print(f"\nProcessing Stage 3 - {dataset_name.upper()} using: {os.path.basename(latest_csv)}")
    
    df = pd.read_csv(latest_csv)
    
    s_map = {
        'u': 'None', 
        's': 'Same-Label', 
        'd': 'Non-Conflicting', 
        'p': 'Full'
    }
    df['Strategy'] = df['strategy'].map(s_map)

    noise_rates = sorted(df['noise_rate'].unique())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    unique_k = sorted([k for k in df['k'].unique() if k > 0])

    # 15x7 is more compact for the wide matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    fig.suptitle(f"{dataset_name.upper()} STAGE 3 - Acc Gain % (k-NN Mode vs. k)", 
                 fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()

    def get_pivot(data_in):
        piv = data_in.groupby(['Strategy', 'k'])['gain_pct'].mean().reset_index()
        u_row = piv[piv['Strategy'] == s_map['u']]
        if not u_row.empty:
            u_val = u_row['gain_pct'].iloc[0]
            piv = piv[piv['Strategy'] != s_map['u']]
            u_baseline = pd.DataFrame({'Strategy': [s_map['u']]*len(unique_k), 'k': unique_k, 'gain_pct': [u_val]*len(unique_k)})
            piv = pd.concat([piv, u_baseline])
        res = piv.pivot(index='Strategy', columns='k', values='gain_pct')
        order = [s_map['u'], s_map['s'], s_map['d'], s_map['p']]
        res = res.reindex(order)
        res.index.name = 'k-NN Mode'
        return res

    for i, nr in enumerate(noise_rates):
        ax = axes_flat[i]
        sub = df[df['noise_rate'] == nr].copy()
        sub['gain_pct'] = sub['gain'] * 100
        matrix = get_pivot(sub)
        
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", 
                    annot_kws={"size": 7}, ax=ax, square=False,
                    cbar_kws={'label': 'Gain (%)', 'shrink': 1.0})
        ax.set_title(f"Noise {int(nr*100)}%", fontsize=10)
        ax.set_ylabel("k-NN Mode", fontsize=8)
        ax.set_xlabel("k", fontsize=8)
        ax.tick_params(labelsize=7)

    # Global Mean
    ax_mean = axes_flat[5]
    mean_df = df.copy()
    mean_df['gain_pct'] = mean_df['gain'] * 100
    matrix_mean = get_pivot(mean_df)
    
    sns.heatmap(matrix_mean, annot=True, fmt=".2f", cmap="viridis", 
                annot_kws={"size": 7}, ax=ax_mean, square=False,
                cbar_kws={'label': 'Mean Gain (%)', 'shrink': 1.0})
    ax_mean.set_title("GLOBAL MEAN", fontsize=10)
    ax_mean.set_ylabel("k-NN Mode", fontsize=8)
    ax_mean.set_xlabel("k", fontsize=8)
    ax_mean.tick_params(labelsize=7)

    for j in range(i + 2, 6): axes_flat[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(ds_dir, f"heatmaps_STAGE3_{dataset_name}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Heatmap saved to: {out_path}")
    
    out_path_pdf = os.path.join(ds_dir, f"heatmaps_STAGE3_{dataset_name}.pdf")
    plt.savefig(out_path_pdf, dpi=300)
    print(f"Heatmap saved to: {out_path_pdf}")
    
    plt.close()

if __name__ == "__main__":
    for ds in DATASETS: plot_stage3(ds)
