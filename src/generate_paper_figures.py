#!/usr/bin/env python3
"""Generate publication-quality figures and tables for the ELEN6772 paper.

Incorporates all zero-shot, fine-tuned, ML, and DL experiment results.
Uses IEEE conference style (compact, readable at column width).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# IEEE column width ~3.5in, double column ~7.16in
COL_W = 3.5
DBL_W = 7.16

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_BASE, 'figures')
TABLE_OUT = os.path.join(_BASE, 'tables')

# ============================================================
# Colour palette
# ============================================================
C_NSL   = '#2196F3'   # blue
C_UNSW  = '#FF9800'   # orange
C_NSL_L = '#90CAF9'   # light blue
C_UNSW_L= '#FFE0B2'   # light orange
C_FT    = '#4CAF50'   # green
C_REF   = '#E53935'   # red for reference lines


# ============================================================
# DATA
# ============================================================

# --- Zero-Shot (Qwen2.5 family, by size) ---
ZS_SIZES      = [1.5, 3, 7, 14, 32]
ZS_SIZE_LABELS= ['1.5B', '3B', '7B', '14B', '32B']
ZS_NSL_F1     = [0.1206, 0.2956, 0.4259, 0.4196, 0.4160]
ZS_UNSW_F1    = [0.0873, 0.0427, 0.0615, 0.0573, 0.0361]

# --- Fine-Tuned (Qwen2.5 family, by size -- no 14B) ---
FT_SIZES      = [1.5, 3, 7, 32]
FT_SIZE_LABELS= ['1.5B', '3B', '7B', '32B']
FT_NSL_F1     = [0.4478, 0.3936, 0.4610, 0.4636]
FT_UNSW_F1    = [0.0937, 0.0898, 0.1212, 0.1742]

# --- Family comparison (7B class) ---
FAMILIES       = ['Qwen2.5-7B', 'Mistral-7B', 'DeepSeek-7B']
FAM_ZS_NSL_F1  = [0.4259, 0.2385, 0.0856]
FAM_FT_NSL_F1  = [0.4610, 0.4641, 0.1206]
FAM_ZS_UNSW_F1 = [0.0615, 0.0081, 0.0723]
FAM_FT_UNSW_F1 = [0.1212, 0.1612, 0.0622]

# --- ML/DL baselines on ~500-sample subsets ---
# From results/nsl_kdd/subset_500_ml_results.json & unsw_nb15 equivalent
ML_MODELS = ['XGBoost', 'RF', 'LR', 'SVM']
ML_NSL_F1 = [0.6408, 0.4949, 0.7039, 0.5067]
ML_UNSW_F1= [0.6109, 0.4790, 0.3164, 0.3343]

# 1D-CNN baseline (user-provided, on comparable test sets)
CNN_NSL_F1 = 0.6383
CNN_UNSW_F1= 0.3788

# Reference lines for XGBoost (user-specified values)
XGBOOST_NSL_REF = 0.5641
XGBOOST_UNSW_REF= 0.5086

# Best FT per family for fair comparison
BEST_FT = {
    'Qwen-7B':    {'nsl': 0.4610, 'unsw': 0.1212},
    'Mistral-7B': {'nsl': 0.4641, 'unsw': 0.1612},
    'DeepSeek-7B':{'nsl': 0.1206, 'unsw': 0.0622},
}


# ============================================================
# Figure 1: Size Scaling (Qwen2.5 family)
# ============================================================
def fig_size_scaling():
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    # ZS lines (dashed)
    ax.plot(ZS_SIZES, ZS_NSL_F1, 's--', color=C_NSL, label='ZS -- NSL-KDD',
            markersize=4.5, linewidth=1.3)
    ax.plot(ZS_SIZES, ZS_UNSW_F1, 's--', color=C_UNSW, label='ZS -- UNSW-NB15',
            markersize=4.5, linewidth=1.3)

    # FT lines (solid)
    ax.plot(FT_SIZES, FT_NSL_F1, 'D-', color=C_NSL, label='FT -- NSL-KDD',
            markersize=4.5, linewidth=1.3, markerfacecolor=C_NSL)
    ax.plot(FT_SIZES, FT_UNSW_F1, 'D-', color=C_UNSW, label='FT -- UNSW-NB15',
            markersize=4.5, linewidth=1.3, markerfacecolor=C_UNSW)

    # XGBoost reference lines
    ax.axhline(y=XGBOOST_NSL_REF, color=C_NSL, linestyle=':', linewidth=0.9, alpha=0.6)
    ax.text(32, XGBOOST_NSL_REF + 0.012, 'XGB', fontsize=6, color=C_NSL, alpha=0.7,
            ha='right')
    ax.axhline(y=XGBOOST_UNSW_REF, color=C_UNSW, linestyle=':', linewidth=0.9, alpha=0.6)
    ax.text(32, XGBOOST_UNSW_REF + 0.012, 'XGB', fontsize=6, color=C_UNSW, alpha=0.7,
            ha='right')

    ax.set_xscale('log')
    ax.set_xticks(ZS_SIZES)
    ax.set_xticklabels(ZS_SIZE_LABELS)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel('Model Size (Qwen2.5)')
    ax.set_ylabel('Macro F1')
    ax.set_ylim(-0.02, 0.62)
    ax.legend(loc='upper left', framealpha=0.9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(OUT, 'size_scaling.pdf'))
    fig.savefig(os.path.join(OUT, 'size_scaling.png'))
    plt.close(fig)
    print('  [OK] size_scaling.pdf')


# ============================================================
# Figure 2: Family Comparison (7B class) -- ZS vs FT
# ============================================================
def fig_family_comparison():
    x = np.arange(len(FAMILIES))
    w = 0.18  # bar width -- 4 bars per group

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    ax.bar(x - 1.5*w, FAM_ZS_NSL_F1, w, label='ZS -- NSL-KDD',
           color=C_NSL_L, edgecolor=C_NSL, linewidth=0.6)
    ax.bar(x - 0.5*w, FAM_FT_NSL_F1, w, label='FT -- NSL-KDD',
           color=C_NSL, edgecolor='white', linewidth=0.5)
    ax.bar(x + 0.5*w, FAM_ZS_UNSW_F1, w, label='ZS -- UNSW-NB15',
           color=C_UNSW_L, edgecolor=C_UNSW, linewidth=0.6)
    ax.bar(x + 1.5*w, FAM_FT_UNSW_F1, w, label='FT -- UNSW-NB15',
           color=C_UNSW, edgecolor='white', linewidth=0.5)

    # XGBoost reference
    ax.axhline(y=XGBOOST_NSL_REF, color=C_REF, linestyle=':', linewidth=0.9, alpha=0.55)
    ax.text(len(FAMILIES)-0.6, XGBOOST_NSL_REF + 0.012, 'XGB (NSL)',
            fontsize=5.5, color=C_REF, alpha=0.7, ha='right')
    ax.axhline(y=XGBOOST_UNSW_REF, color=C_REF, linestyle='--', linewidth=0.9, alpha=0.45)
    ax.text(len(FAMILIES)-0.6, XGBOOST_UNSW_REF + 0.012, 'XGB (UNSW)',
            fontsize=5.5, color=C_REF, alpha=0.7, ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES)
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0, 0.62)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(OUT, 'family_comparison.pdf'))
    fig.savefig(os.path.join(OUT, 'family_comparison.png'))
    plt.close(fig)
    print('  [OK] family_comparison.pdf')


# ============================================================
# Figure 3: Fair Comparison -- ML/DL vs best LLM FT (~500 samples)
# ============================================================
def fig_fair_comparison():
    # Merge ML baselines + CNN + best LLM FT
    models = ['LR', 'SVM', 'RF', 'XGBoost', '1D-CNN',
              'FT-Qwen', 'FT-Mistral', 'FT-DeepSeek']
    nsl_f1 = [0.7039, 0.5067, 0.4949, 0.6408, CNN_NSL_F1,
              BEST_FT['Qwen-7B']['nsl'],
              BEST_FT['Mistral-7B']['nsl'],
              BEST_FT['DeepSeek-7B']['nsl']]
    unsw_f1= [0.3164, 0.3343, 0.4790, 0.6109, CNN_UNSW_F1,
              BEST_FT['Qwen-7B']['unsw'],
              BEST_FT['Mistral-7B']['unsw'],
              BEST_FT['DeepSeek-7B']['unsw']]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(DBL_W * 0.65, 2.4))

    # Colour: solid for ML/DL, lighter for LLM
    c_nsl  = [C_NSL]*5  + [C_NSL_L]*3
    c_unsw = [C_UNSW]*5 + [C_UNSW_L]*3

    bars1 = ax.bar(x - w/2, nsl_f1, w, color=c_nsl, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, unsw_f1, w, color=c_unsw, edgecolor='white', linewidth=0.5)

    # Value labels on top
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom', fontsize=5.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom', fontsize=5.5)

    # Divider
    ax.axvline(x=4.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(2.0, 0.78, 'ML / DL', ha='center', fontsize=7, color='gray', fontstyle='italic')
    ax.text(6.0, 0.78, 'LLM (LoRA FT)', ha='center', fontsize=7, color='gray', fontstyle='italic')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=C_NSL, label='NSL-KDD'),
                       Patch(facecolor=C_UNSW, label='UNSW-NB15')]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0, 0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(OUT, 'fair_comparison.pdf'))
    fig.savefig(os.path.join(OUT, 'fair_comparison.png'))
    plt.close(fig)
    print('  [OK] fair_comparison.pdf')


# ============================================================
# LaTeX table: ft_comparison.tex
# ============================================================
def gen_ft_comparison_table():
    content = r"""\begin{table}[htbp]
\centering
\caption{Multi-model LoRA fine-tuning comparison ($N{\approx}500$). All models use identical LoRA config (rank 8, $\alpha$=16, 5K training samples, 3 epochs).}
\label{tab:ft_comparison}
\small
\begin{tabular}{llcccc}
\toprule
\multirow{2}{*}{Model} & \multirow{2}{*}{Size} & \multicolumn{2}{c}{NSL-KDD} & \multicolumn{2}{c}{UNSW-NB15} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
 & & Acc & F1 & Acc & F1 \\
\midrule
Qwen2.5-1.5B & 1.5B & .719 & .448 & .353 & .094 \\
Qwen2.5-3B & 3B & .679 & .394 & .284 & .090 \\
Qwen2.5-7B & 7B & .749 & .461 & .462 & .121 \\
Qwen2.5-32B & 32B & .737 & .464 & .464 & .174 \\
Mistral-7B & 7B & .743 & .464 & .375 & .161 \\
DeepSeek-7B & 7B & .432 & .121 & .452 & .062 \\
\midrule
\textit{XGBoost (ref.)} & -- & .767 & .564 & .767 & .509 \\
\bottomrule
\end{tabular}
\end{table}
"""
    path = os.path.join(TABLE_OUT, 'ft_comparison.tex')
    with open(path, 'w') as f:
        f.write(content)
    print(f'  [OK] {path}')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TABLE_OUT, exist_ok=True)
    print('Generating paper figures and tables...')
    fig_family_comparison()
    gen_ft_comparison_table()
    print(f'All outputs saved to {OUT}/ and {TABLE_OUT}/')
