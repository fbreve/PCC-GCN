"""
select_best_config.py  —  versão corrigida
"""

import glob
import pandas as pd

# ─── CONFIGURAÇÃO ────────────────────────────────────────────────────────────
STATS_CSV    = r"c:\users\fbrev\documents\acadêmico\simulações\python\gcn+lnpcc\results\gridsearch\citeseer\microgrid\gridsearch_Citeseer_stats_*.csv"
DATASET_NAME = "Citeseer"

NOISE_FILTER = [0.10, 0.20, 0.30, 0.40, 0.50]   # None = todos os noise rates

R_MIN = 0.89    # retention_clean mínima  → PCC-remove
P_MIN = 0.65    # precision_noise mínima  → PCC-rerótula

TOP_N = 10
# ─────────────────────────────────────────────────────────────────────────────

METRICS = ["f1_noise", "recall_noise", "precision_noise",
           "retention_clean", "corrected_to_true", "acc_labeled", "acc"]

# ── 1. Carrega CSV ────────────────────────────────────────────────────────────
files = sorted(glob.glob(STATS_CSV))
if not files:
    raise FileNotFoundError(f"Nenhum arquivo encontrado: {STATS_CSV}")
print(f"Usando: {files[-1]}\n")

df = pd.read_csv(files[-1])
df.columns = [c.strip().lower() for c in df.columns]
print("Colunas disponíveis:", df.columns.tolist(), "\n")

# ── 2. Detecta formato das colunas de média ───────────────────────────────────
sample = METRICS[0]   # "f1_noise"

if f"{sample}_mean" in df.columns:
    mean_cols = {m: f"{m}_mean" for m in METRICS}   # sufixo _mean
    print("Formato detectado: sufixo  →  ex.: 'f1_noise_mean'")
elif f"mean_{sample}" in df.columns:
    mean_cols = {m: f"mean_{m}" for m in METRICS}   # prefixo mean_
    print("Formato detectado: prefixo →  ex.: 'mean_f1_noise'")
elif sample in df.columns:
    mean_cols = {m: m for m in METRICS}              # sem sufixo/prefixo
    print("Formato detectado: sem prefixo/sufixo →  ex.: 'f1_noise'")
else:
    raise KeyError(
        f"Coluna de métrica não reconhecida.\n"
        f"Esperava '{sample}', '{sample}_mean' ou 'mean_{sample}'.\n"
        f"Colunas disponíveis: {df.columns.tolist()}"
    )


# ── 3. Identifica coluna noise_rate ──────────────────────────────────────────
nr_col = next((c for c in ["noise_rate", "nr", "noise"] if c in df.columns), None)
if nr_col is None:
    raise KeyError(f"Coluna de noise_rate não encontrada. Colunas: {df.columns.tolist()}")

# ── 4. Filtra noise rates — SÓ para a agregação ──────────────────────────────
if NOISE_FILTER:
    mask = df[nr_col].round(2).isin([round(n, 2) for n in NOISE_FILTER])
    df_agg = df[mask].copy()          # ← df original intacto para o breakdown
    print(f"Noise rates na média global: {NOISE_FILTER}")
else:
    df_agg = df.copy()
    print(f"Noise rates na média global: {sorted(df[nr_col].unique())}")

# ── 5. Agrega por (p_grd, dexp) ───────────────────────────────────────────────
agg_dict = {mean_cols[m]: "mean" for m in METRICS}
global_df = (
    df_agg.groupby(["p_grd", "dexp"])   # ← df_agg, não df
          .agg(agg_dict)
          .reset_index()
          .rename(columns={mean_cols[m]: m for m in METRICS})
)


# ── helpers ───────────────────────────────────────────────────────────────────
SHOW_COLS = ["p_grd", "dexp"] + METRICS

def show_table(title, sub):
    print(f"\n{'='*76}\n  {title}\n{'='*76}")
    print(sub[SHOW_COLS].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

def per_noise_breakdown(df_raw, p, d):
    mask = (df_raw["p_grd"].round(4) == round(p, 4)) & \
           (df_raw["dexp"].round(4)  == round(d, 4))
    sub = df_raw[mask].sort_values(nr_col)[[nr_col] + [mean_cols[m] for m in METRICS]]
    sub = sub.rename(columns={mean_cols[m]: m for m in METRICS})
    return sub

# ── 6. PCC-REMOVE ─────────────────────────────────────────────────────────────
print(f"\n{'═'*76}")
print(f"  PCC-REMOVE  |  max f1_noise  |  retention_clean >= {R_MIN}")
print(f"{'═'*76}")

cand_rm = global_df[global_df["retention_clean"] >= R_MIN].copy()
if cand_rm.empty:
    best_ret = global_df["retention_clean"].max()
    print(f"  ⚠️  Nenhum config com retention_clean >= {R_MIN}. "
          f"Relaxando para >= {best_ret - 0.005:.3f}...")
    cand_rm = global_df[global_df["retention_clean"] >= best_ret - 0.005].copy()

cand_rm = cand_rm.sort_values("f1_noise", ascending=False)
best_rm  = cand_rm.iloc[0]

show_table(f"Top {TOP_N} PCC-remove  (retention ≥ {R_MIN})", cand_rm.head(TOP_N))

print(f"\n  ★  MELHOR PCC-REMOVE : p_grd={best_rm['p_grd']:.2f}, dexp={best_rm['dexp']:.1f}")
print(f"     f1_noise={best_rm['f1_noise']:.4f}  "
      f"retention_clean={best_rm['retention_clean']:.4f}  "
      f"precision_noise={best_rm['precision_noise']:.4f}")
print("\n  Breakdown por noise_rate:")
print(per_noise_breakdown(df, best_rm["p_grd"], best_rm["dexp"])
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── 7. PCC-RERÓTULA ───────────────────────────────────────────────────────────
print(f"\n{'═'*76}")
print(f"  PCC-RERÓTULA  |  max corrected_to_true  |  precision_noise >= {P_MIN}")
print(f"{'═'*76}")

cand_rl = global_df[global_df["precision_noise"] >= P_MIN].copy()
if cand_rl.empty:
    best_prec = global_df["precision_noise"].max()
    print(f"  ⚠️  Nenhum config com precision_noise >= {P_MIN}. "
          f"Relaxando para >= {best_prec - 0.01:.3f}...")
    cand_rl = global_df[global_df["precision_noise"] >= best_prec - 0.01].copy()

cand_rl = cand_rl.sort_values("corrected_to_true", ascending=False)
best_rl  = cand_rl.iloc[0]

show_table(f"Top {TOP_N} PCC-rerótula  (precision ≥ {P_MIN})", cand_rl.head(TOP_N))

print(f"\n  ★  MELHOR PCC-RERÓTULA: p_grd={best_rl['p_grd']:.2f}, dexp={best_rl['dexp']:.1f}")
print(f"     corrected_to_true={best_rl['corrected_to_true']:.4f}  "
      f"precision_noise={best_rl['precision_noise']:.4f}  "
      f"retention_clean={best_rl['retention_clean']:.4f}")
print("\n  Breakdown por noise_rate:")
print(per_noise_breakdown(df, best_rl["p_grd"], best_rl["dexp"])
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── 8. Resumo final + CSV ─────────────────────────────────────────────────────
print(f"\n{'═'*76}\n  RESUMO FINAL — {DATASET_NAME}\n{'═'*76}")
summary = pd.DataFrame([
    {"método": "PCC-remove",   "p_grd": best_rm["p_grd"],  "dexp": best_rm["dexp"],  **{m: best_rm[m]  for m in METRICS}},
    {"método": "PCC-rerótula", "p_grd": best_rl["p_grd"],  "dexp": best_rl["dexp"],  **{m: best_rl[m]  for m in METRICS}},
])
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

out_path = files[-1].replace("stats_", f"best_configs_{DATASET_NAME}_").rstrip(".csv") + \
           f"_Rmin{R_MIN}_Pmin{P_MIN}.csv"
summary.to_csv(out_path, index=False)
print(f"\n  💾 Salvo em: {out_path}")
