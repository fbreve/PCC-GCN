import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fix 3: noise rates explícitos para garantir um arquivo por noise rate
NOISE_RATES = [10, 20, 30, 40, 50]


def parse_heatmap(filepath):
    """Lê o CSV usando os índices reais (dexp nas linhas, p_grd nas colunas)."""
    df = pd.read_csv(filepath, index_col=0)
    df.index   = df.index.astype(float)
    df.columns = df.columns.astype(float)
    return df.values.astype(float), df.index.to_numpy(), df.columns.to_numpy()


def analyze_folder(folder):
    # Fix 3: pega exatamente um arquivo por noise rate (o mais recente de cada)
    matrices   = []
    dexp_l_ref = None
    pgrd_l_ref = None

    for nr in NOISE_RATES:
        pattern    = os.path.join(folder, f"heatmap_*acc_labeled*noise{nr:02d}*.csv")
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not candidates:
            print(f"  [!] Nenhum CSV para noise={nr}% em {os.path.basename(folder)}")
            continue
        matrix, dexp_l, pgrd_l = parse_heatmap(candidates[0])
        matrices.append(matrix)
        dexp_l_ref = dexp_l
        pgrd_l_ref = pgrd_l

    if not matrices:
        return None, None, None

    avg_matrix      = np.mean(matrices, axis=0)
    i_best, j_best  = np.unravel_index(np.argmax(avg_matrix), avg_matrix.shape)

    print(f"{os.path.basename(folder):12} | {avg_matrix[i_best, j_best]:7.4f} "
          f"| dexp={dexp_l_ref[i_best]:5.1f} | pgrd={pgrd_l_ref[j_best]:6.2f}")

    df_avg = pd.DataFrame(avg_matrix, index=dexp_l_ref, columns=pgrd_l_ref).round(4)
    df_avg.to_csv(os.path.join(folder, "MEDIA_acc_labeled.csv"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_avg, annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'{os.path.basename(folder)} - acc_labeled')
    plt.xlabel('p_grd')
    plt.ylabel('d_exp')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "MEDIA_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(folder, "MEDIA_heatmap.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    return avg_matrix, dexp_l_ref, pgrd_l_ref


# EXECUÇÃO
print("Dataset      | acc_labeled | dexp  | pgrd  ")
print("-" * 40)

datasets = [
    os.path.join('results', 'gridsearch', 'cora'),
    os.path.join('results', 'gridsearch', 'citeseer'),
    os.path.join('results', 'gridsearch', 'pubmed'),
]

all_avgs      = []
dexp_l_global = None
pgrd_l_global = None

for ds in datasets:
    if not os.path.exists(ds):
        print(f"[!] Diretório não encontrado: {ds}")
        continue
    avg, dexp_l, pgrd_l = analyze_folder(ds)
    if avg is not None:
        all_avgs.append(avg)
        dexp_l_global = dexp_l
        pgrd_l_global = pgrd_l

# Fix 4: avisa se nem todos os datasets foram carregados
if len(all_avgs) == 0:
    print("\n[x] Nenhum resultado encontrado. Verifique os diretórios.")
else:
    if len(all_avgs) < len(datasets):
        print(f"\n[!] Média global calculada sobre {len(all_avgs)} de {len(datasets)} datasets.")

    global_avg = np.mean(all_avgs, axis=0)
    i_g, j_g   = np.unravel_index(np.argmax(global_avg), global_avg.shape)

    print("\nGLOBAL       | {:7.4f} | dexp={:5.1f} | pgrd={:6.2f}".format(
        global_avg[i_g, j_g], dexp_l_global[i_g], pgrd_l_global[j_g]))

    global_dir = os.path.join('results', 'gridsearch')
    df_global  = pd.DataFrame(global_avg, index=dexp_l_global, columns=pgrd_l_global).round(4)
    df_global.to_csv(os.path.join(global_dir, "GLOBAL_MEDIA_acc_labeled.csv"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_global, annot=True, fmt='.3f', cmap='viridis')
    plt.title('GLOBAL - All Datasets (acc_labeled)')
    plt.xlabel('p_grd')
    plt.ylabel('d_exp')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # Fix 1: path correto com os.path.join (era "results\\\\gridsearch\\\\...")
    plt.savefig(os.path.join(global_dir, "GLOBAL_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(global_dir, "GLOBAL_heatmap.pdf"), dpi=300, bbox_inches='tight')
    plt.show()

print("\n[OK] Concluído.")
