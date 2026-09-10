# -*- coding: utf-8 -*-
"""
Plot Stage 1 Heatmaps: p_grd vs d_exp
Filtering: p_grd < 1.0

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
BASE_RESULTS_DIR = r"results\stage1"
FILTER_PGRD_1 = True
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

def plot_stage1(dataset_name):
    ds_dir = os.path.join(_THIS_DIR, BASE_RESULTS_DIR, dataset_name)
    if not os.path.exists(ds_dir):
        print(f"Directory not found: {ds_dir}")
        return

    pattern = os.path.join(ds_dir, f"lnpcc_stage1_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No CSV files found in {ds_dir}")
        return

    latest_csv = files[-1]
    print(f"\nProcessing Stage 1 - {dataset_name.upper()} using: {os.path.basename(latest_csv)}")
    
    df = pd.read_csv(latest_csv)
    if FILTER_PGRD_1:
        df = df[df['p_grd'] < 1.0].copy()

    noise_rates = sorted(df['noise_rate'].unique())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"{dataset_name.upper()} STAGE 1 - Accuracy Gain (pp) ($p_{{grd}}$ vs $d_{{exp}}$)", 
                 fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()

    for i, nr in enumerate(noise_rates):
        ax = axes_flat[i]
        sub = df[df['noise_rate'] == nr].copy()
        sub['gain_pct'] = sub['gain'] * 100
        
        # Average gain for p_grd == 0 to replicate across all d_exp
        if 0.0 in sub['p_grd'].values:
            mean_pgrd_0 = sub[sub['p_grd'] == 0.0]['gain_pct'].mean()
            sub.loc[sub['p_grd'] == 0.0, 'gain_pct'] = mean_pgrd_0

        matrix = sub.groupby(['dexp', 'p_grd'])['gain_pct'].mean().reset_index()
        matrix = matrix.pivot(index='dexp', columns='p_grd', values='gain_pct').sort_index(ascending=False)
        
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", 
                    annot_kws={"size": 8}, ax=ax, square=False,
                    cbar_kws={'label': 'Accuracy Gain (pp)', 'shrink': 1.00})
        ax.set_title(f"Noise {int(nr*100)}%")
        ax.set_xlabel(r"$p_{grd}$")
        ax.set_ylabel(r"$d_{exp}$")

    # Global Mean
    ax_mean = axes_flat[5]
    mean_df = df.copy()
    mean_df['gain_pct'] = mean_df['gain'] * 100
    if 0.0 in mean_df['p_grd'].values:
        mean_pgrd_0_global = mean_df[mean_df['p_grd'] == 0.0]['gain_pct'].mean()
        mean_df.loc[mean_df['p_grd'] == 0.0, 'gain_pct'] = mean_pgrd_0_global

    mean_matrix = mean_df.groupby(['dexp', 'p_grd'])['gain_pct'].mean().reset_index()
    mean_matrix = mean_matrix.pivot(index='dexp', columns='p_grd', values='gain_pct').sort_index(ascending=False)
    
    sns.heatmap(mean_matrix, annot=True, fmt=".2f", cmap="viridis", 
                annot_kws={"size": 8}, ax=ax_mean, square=False,
                cbar_kws={'label': 'Mean Accuracy Gain (pp)', 'shrink': 1.00})
    ax_mean.set_title("GLOBAL MEAN")
    ax_mean.set_xlabel(r"$p_{grd}$")
    ax_mean.set_ylabel(r"$d_{exp}$")

    for j in range(i + 1, 5): axes_flat[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(ds_dir, f"heatmaps_STAGE1_{dataset_name}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Heatmap saved to: {out_path}")
    
    out_path_pdf = os.path.join(ds_dir, f"heatmaps_STAGE1_{dataset_name}.pdf")
    plt.savefig(out_path_pdf, dpi=300)
    print(f"Heatmap saved to: {out_path_pdf}")
    
    plt.close()

if __name__ == "__main__":
    for ds in DATASETS: plot_stage1(ds)
