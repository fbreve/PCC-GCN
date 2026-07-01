# -*- coding: utf-8 -*-
"""
Plot Stage 2 Heatmaps: \tau_{rem} vs \tau_{rel}
Filtering: \tau_{rem} > 0.0

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
BASE_RESULTS_DIR = r"results\stage2"
FILTER_REM_0 = True
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

def plot_stage2(dataset_name):
    ds_dir = os.path.join(_THIS_DIR, BASE_RESULTS_DIR, dataset_name)
    if not os.path.exists(ds_dir):
        print(f"Directory not found: {ds_dir}")
        return

    pattern = os.path.join(ds_dir, f"lnpcc_stage2_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No CSV files found in {ds_dir}")
        return

    latest_csv = files[-1]
    print(f"\nProcessing Stage 2 - {dataset_name.upper()} using: {os.path.basename(latest_csv)}")
    
    df = pd.read_csv(latest_csv)
    if FILTER_REM_0:
        df = df[df['rem_thresh'] > 0.0].copy()

    noise_rates = sorted(df['noise_rate'].unique())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"{dataset_name.upper()} STAGE 2 - Acc Gain % ($\\tau_{{\\mathrm{{rem}}}}$ vs $\\tau_{{\\mathrm{{rel}}}}$)", 
                 fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()

    for i, nr in enumerate(noise_rates):
        ax = axes_flat[i]
        sub = df[df['noise_rate'] == nr].copy()
        sub['gain_pct'] = sub['gain'] * 100
        if FILTER_REM_0:
            sub = sub[sub['rem_thresh'] > 0.0].copy()
            
        matrix = sub.groupby(['rem_thresh', 'rel_thresh'])['gain_pct'].mean().reset_index()
        matrix = matrix.pivot(index='rem_thresh', columns='rel_thresh', values='gain_pct').sort_index(ascending=False)
        
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", 
                    annot_kws={"size": 8}, ax=ax, square=False,
                    cbar_kws={'label': 'Acc Gain (%)', 'shrink': 1.00})
        ax.set_title(f"Noise {int(nr*100)}%")
        ax.set_xlabel(r"$\tau_{\mathrm{rel}}$")
        ax.set_ylabel(r"$\tau_{\mathrm{rem}}$")

    # Global Mean
    ax_mean = axes_flat[5]
    mean_df = df.copy()
    mean_df['gain_pct'] = mean_df['gain'] * 100
    if FILTER_REM_0:
        mean_df = mean_df[mean_df['rem_thresh'] > 0.0].copy()

    mean_matrix = mean_df.groupby(['rem_thresh', 'rel_thresh'])['gain_pct'].mean().reset_index()
    mean_matrix = mean_matrix.pivot(index='rem_thresh', columns='rel_thresh', values='gain_pct').sort_index(ascending=False)
    
    sns.heatmap(mean_matrix, annot=True, fmt=".2f", cmap="viridis", 
                annot_kws={"size": 8}, ax=ax_mean, square=False,
                cbar_kws={'label': 'Mean Acc Gain (%)', 'shrink': 1.00})
    ax_mean.set_title("GLOBAL MEAN")
    ax_mean.set_xlabel(r"$\tau_{\mathrm{rel}}$")
    ax_mean.set_ylabel(r"$\tau_{\mathrm{rem}}$")

    for j in range(i + 1, 5): axes_flat[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(ds_dir, f"heatmaps_STAGE2_{dataset_name}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Heatmap saved to: {out_path}")
    
    out_path_pdf = os.path.join(ds_dir, f"heatmaps_STAGE2_{dataset_name}.pdf")
    plt.savefig(out_path_pdf, dpi=300)
    print(f"Heatmap saved to: {out_path_pdf}")
    
    plt.close()

if __name__ == "__main__":
    for ds in DATASETS: plot_stage2(ds)
