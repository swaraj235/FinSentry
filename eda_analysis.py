"""
FinSentry — Comprehensive EDA Script
Performs detailed exploratory data analysis on the fraud detection dataset.
Generates all charts as PNG files in the eda_outputs/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Setup ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Styling
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'figure.dpi': 150,
    'font.family': 'sans-serif',
})

print("=" * 70)
print("  FinSentry — Exploratory Data Analysis")
print("=" * 70)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("\n[1/10] Loading dataset...")
df = pd.read_csv("dataset/dataset.csv")
print(f"  ✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# ── 2. Basic Info ────────────────────────────────────────────────────────────
print("\n[2/10] Basic Dataset Info:")
target_col = df.columns[-1]  # Last column is target
print(f"  Target column: {target_col}")
print(f"  Feature columns: {df.shape[1] - 1}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Data types
dtypes = df.dtypes.value_counts()
print(f"\n  Data Types:")
for dtype, count in dtypes.items():
    print(f"    {dtype}: {count} columns")

# ── 3. Target Variable Analysis ─────────────────────────────────────────────
print("\n[3/10] Target Variable Analysis:")
target_counts = df[target_col].value_counts()
target_pcts = df[target_col].value_counts(normalize=True) * 100
for val in sorted(target_counts.index):
    label = "Legitimate" if val == 0 else "Fraudulent"
    print(f"  Class {val} ({label}): {target_counts[val]:,} ({target_pcts[val]:.2f}%)")

imbalance_ratio = target_counts[0] / target_counts[1] if 1 in target_counts.index else float('inf')
print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")

# Chart 1: Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ['#238636', '#da3633']
bars = axes[0].bar(['Legitimate (0)', 'Fraudulent (1)'],
                    [target_counts.get(0, 0), target_counts.get(1, 0)],
                    color=colors, edgecolor='#30363d', linewidth=1.5)
axes[0].set_title('Transaction Count by Class', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('Number of Transactions', fontsize=11)
for bar, count in zip(bars, [target_counts.get(0, 0), target_counts.get(1, 0)]):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#e6edf3')

# Pie chart
axes[1].pie([target_counts.get(0, 0), target_counts.get(1, 0)],
            labels=['Legitimate', 'Fraudulent'],
            colors=colors, autopct='%1.2f%%',
            startangle=90, textprops={'color': '#e6edf3', 'fontsize': 12},
            wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2},
            explode=(0, 0.1))
axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold', pad=15)

plt.suptitle('Target Variable: Extreme Class Imbalance', fontsize=16, fontweight='bold', y=1.02, color='#f0883e')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_target_distribution.png', bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  📊 Saved: 01_target_distribution.png")

# ── 4. Missing Values Analysis ──────────────────────────────────────────────
print("\n[4/10] Missing Values Analysis:")
missing_per_col = df.isnull().sum()
missing_pct_per_col = (missing_per_col / len(df)) * 100

total_missing = missing_per_col.sum()
total_cells = df.shape[0] * df.shape[1]
print(f"  Total missing values: {total_missing:,} / {total_cells:,} ({total_missing/total_cells*100:.2f}%)")

cols_no_missing = (missing_per_col == 0).sum()
cols_some_missing = ((missing_per_col > 0) & (missing_pct_per_col <= 50)).sum()
cols_half_missing = ((missing_pct_per_col > 50) & (missing_pct_per_col <= 80)).sum()
cols_mostly_missing = (missing_pct_per_col > 80).sum()

print(f"  Columns with 0% missing:    {cols_no_missing}")
print(f"  Columns with 1-50% missing: {cols_some_missing}")
print(f"  Columns with 50-80% missing:{cols_half_missing}")
print(f"  Columns with >80% missing:  {cols_mostly_missing}")

# Chart 2: Missing Values Distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram of missing percentages
missing_nonzero = missing_pct_per_col[missing_pct_per_col > 0]
if len(missing_nonzero) > 0:
    axes[0].hist(missing_nonzero, bins=50, color='#f0883e', edgecolor='#0d1117', alpha=0.85)
    axes[0].set_xlabel('Missing Value Percentage (%)', fontsize=11)
    axes[0].set_ylabel('Number of Columns', fontsize=11)
    axes[0].set_title('Distribution of Missing Values Across Columns', fontsize=13, fontweight='bold', pad=10)
    axes[0].axvline(x=80, color='#da3633', linestyle='--', linewidth=2, label='80% threshold (drop)')
    axes[0].legend(fontsize=10)

# Bar: Buckets
categories = ['0% Missing', '1-50%', '50-80%', '>80%']
counts = [cols_no_missing, cols_some_missing, cols_half_missing, cols_mostly_missing]
bucket_colors = ['#238636', '#f0883e', '#da3633', '#8957e5']
bars = axes[1].bar(categories, counts, color=bucket_colors, edgecolor='#30363d', linewidth=1.5)
axes[1].set_title('Columns Grouped by Missing %', fontsize=13, fontweight='bold', pad=10)
axes[1].set_ylabel('Number of Columns', fontsize=11)
for bar, count in zip(bars, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold', color='#e6edf3')

plt.suptitle('Missing Data Landscape', fontsize=16, fontweight='bold', y=1.02, color='#f0883e')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_missing_values.png', bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  📊 Saved: 02_missing_values.png")

# ── 5. Feature Type Classification ──────────────────────────────────────────
print("\n[5/10] Feature Type Classification:")
feature_cols = [c for c in df.columns if c != target_col]

binary_cols = []
continuous_cols = []
sparse_cols = []
other_cols = []

for col in feature_cols:
    non_null = df[col].dropna()
    if len(non_null) == 0:
        sparse_cols.append(col)
        continue
    unique_vals = non_null.unique()
    null_pct = df[col].isnull().sum() / len(df)
    
    if null_pct > 0.8:
        sparse_cols.append(col)
    elif len(unique_vals) <= 2:
        binary_cols.append(col)
    elif non_null.dtype in ['float64', 'float32', 'int64'] and len(unique_vals) > 10:
        continuous_cols.append(col)
    else:
        other_cols.append(col)

print(f"  Binary features (0/1):     {len(binary_cols)}")
print(f"  Continuous features:       {len(continuous_cols)}")
print(f"  Sparse features (>80% NA): {len(sparse_cols)}")
print(f"  Other features:            {len(other_cols)}")

# Chart 3: Feature Type Breakdown
fig, ax = plt.subplots(figsize=(10, 6))
feat_types = ['Continuous', 'Binary', 'Sparse (>80% NA)', 'Other']
feat_counts = [len(continuous_cols), len(binary_cols), len(sparse_cols), len(other_cols)]
feat_colors = ['#58a6ff', '#238636', '#da3633', '#8b949e']

wedges, texts, autotexts = ax.pie(feat_counts, labels=feat_types, colors=feat_colors,
                                   autopct='%1.1f%%', startangle=140,
                                   textprops={'color': '#e6edf3', 'fontsize': 11},
                                   wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2},
                                   pctdistance=0.8)
ax.set_title('Feature Type Breakdown (3,924 Features)', fontsize=15, fontweight='bold', pad=20, color='#f0883e')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_feature_types.png', bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  📊 Saved: 03_feature_types.png")

# ── 6. Value Range Analysis (Continuous Features) ───────────────────────────
print("\n[6/10] Value Range Analysis (sample of continuous features):")
if continuous_cols:
    sample_cont = continuous_cols[:min(20, len(continuous_cols))]
    stats = df[sample_cont].describe().T
    print(f"  Sample of {len(sample_cont)} continuous features:")
    print(f"  Mean range:  [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
    print(f"  Std range:   [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")
    print(f"  Min range:   [{stats['min'].min():.4f}, {stats['min'].max():.4f}]")
    print(f"  Max range:   [{stats['max'].min():.4f}, {stats['max'].max():.4f}]")
    
    # Check if data is pre-normalized (0-1 range)
    all_cont_stats = df[continuous_cols].describe().T
    pct_in_01 = ((all_cont_stats['min'] >= -0.01) & (all_cont_stats['max'] <= 1.01)).sum() / len(continuous_cols) * 100
    print(f"  Features in [0, 1] range:  {pct_in_01:.1f}% → {'Data appears PRE-NORMALIZED' if pct_in_01 > 70 else 'Mixed range data'}")

    # Chart 4: Distribution of 12 random continuous features
    np.random.seed(42)
    plot_cols = np.random.choice(continuous_cols, size=min(12, len(continuous_cols)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(plot_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=50, color='#58a6ff', edgecolor='#0d1117', alpha=0.8)
        axes[i].set_title(col, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].tick_params(labelsize=8)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribution of 12 Random Continuous Features', fontsize=16, fontweight='bold', y=1.01, color='#f0883e')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_feature_distributions.png', bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("  📊 Saved: 04_feature_distributions.png")

# ── 7. Fraud vs Legitimate Comparison ───────────────────────────────────────
print("\n[7/10] Fraud vs Legitimate Feature Comparison:")
legit = df[df[target_col] == 0]
fraud = df[df[target_col] == 1]

# Find features with biggest difference between fraud and legit
if continuous_cols:
    mean_diffs = []
    for col in continuous_cols:
        legit_mean = legit[col].mean()
        fraud_mean = fraud[col].mean()
        if pd.notna(legit_mean) and pd.notna(fraud_mean):
            diff = abs(fraud_mean - legit_mean)
            legit_std = legit[col].std()
            if legit_std > 0:
                effect_size = diff / legit_std  # Cohen's d-like
            else:
                effect_size = 0
            mean_diffs.append((col, legit_mean, fraud_mean, diff, effect_size))
    
    mean_diffs.sort(key=lambda x: x[4], reverse=True)
    
    print(f"\n  Top 10 Most Discriminative Features (by effect size):")
    print(f"  {'Feature':<12} {'Legit Mean':>12} {'Fraud Mean':>12} {'Effect Size':>12}")
    print(f"  {'-'*48}")
    for col, lm, fm, d, es in mean_diffs[:10]:
        print(f"  {col:<12} {lm:>12.4f} {fm:>12.4f} {es:>12.4f}")
    
    # Chart 5: Top discriminative features
    top_disc = mean_diffs[:8]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (col, lm, fm, d, es) in enumerate(top_disc):
        legit_data = legit[col].dropna()
        fraud_data = fraud[col].dropna()
        axes[i].hist(legit_data, bins=40, alpha=0.6, color='#238636', label='Legit', density=True, edgecolor='#0d1117')
        axes[i].hist(fraud_data, bins=40, alpha=0.7, color='#da3633', label='Fraud', density=True, edgecolor='#0d1117')
        axes[i].set_title(f'{col}\n(effect={es:.2f})', fontsize=10, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].tick_params(labelsize=8)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Top 8 Most Discriminative Features: Fraud vs Legitimate', fontsize=16, fontweight='bold', y=1.01, color='#f0883e')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_fraud_vs_legit.png', bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("  📊 Saved: 05_fraud_vs_legit.png")

# ── 8. Correlation Analysis ─────────────────────────────────────────────────
print("\n[8/10] Correlation Analysis:")
# Use top discriminative features for correlation
if mean_diffs:
    top_features = [x[0] for x in mean_diffs[:20]]
    corr_data = df[top_features + [target_col]].dropna()
    corr_matrix = corr_data.corr()
    
    # Correlation with target
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    print(f"\n  Correlation with Target (top features):")
    print(f"  {'Feature':<12} {'Correlation':>12}")
    print(f"  {'-'*24}")
    for feat, corr_val in target_corr.head(10).items():
        print(f"  {feat:<12} {corr_val:>12.4f}")
    
    # Chart 6: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, linecolor='#21262d',
                cbar_kws={'label': 'Correlation', 'shrink': 0.8}, ax=ax)
    ax.set_title('Correlation Heatmap: Top 20 Discriminative Features + Target', fontsize=14, fontweight='bold', pad=15, color='#f0883e')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_correlation_heatmap.png', bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("  📊 Saved: 06_correlation_heatmap.png")

# ── 9. Variance Analysis ────────────────────────────────────────────────────
print("\n[9/10] Variance Analysis:")
if continuous_cols:
    variances = df[continuous_cols].var().sort_values()
    near_zero_var = (variances < 0.001).sum()
    low_var = ((variances >= 0.001) & (variances < 0.01)).sum()
    normal_var = (variances >= 0.01).sum()
    print(f"  Near-zero variance (<0.001): {near_zero_var} features → candidates for REMOVAL")
    print(f"  Low variance (0.001-0.01):   {low_var} features")
    print(f"  Normal variance (>0.01):     {normal_var} features")
    
    # Chart 7: Variance distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(variances, bins=80, color='#8957e5', edgecolor='#0d1117', alpha=0.85)
    ax.axvline(x=0.001, color='#da3633', linestyle='--', linewidth=2, label='Near-zero threshold (0.001)')
    ax.axvline(x=0.01, color='#f0883e', linestyle='--', linewidth=2, label='Low variance threshold (0.01)')
    ax.set_xlabel('Variance', fontsize=11)
    ax.set_ylabel('Number of Features', fontsize=11)
    ax.set_title('Feature Variance Distribution (Continuous Features)', fontsize=14, fontweight='bold', pad=10, color='#f0883e')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_variance_distribution.png', bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("  📊 Saved: 07_variance_distribution.png")

# ── 10. Outlier Analysis (for fraud detection) ──────────────────────────────
print("\n[10/10] Outlier Analysis on Top Features:")
if mean_diffs:
    top_6 = [x[0] for x in mean_diffs[:6]]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(top_6):
        data_legit = legit[col].dropna()
        data_fraud = fraud[col].dropna()
        
        bp = axes[i].boxplot([data_legit, data_fraud],
                             labels=['Legit', 'Fraud'],
                             patch_artist=True,
                             boxprops=dict(facecolor='#161b22', edgecolor='#58a6ff'),
                             medianprops=dict(color='#f0883e', linewidth=2),
                             whiskerprops=dict(color='#8b949e'),
                             capprops=dict(color='#8b949e'),
                             flierprops=dict(markerfacecolor='#da3633', marker='o', markersize=3, alpha=0.5))
        bp['boxes'][0].set_facecolor('#238636')
        bp['boxes'][0].set_alpha(0.3)
        bp['boxes'][1].set_facecolor('#da3633')
        bp['boxes'][1].set_alpha(0.3)
        
        axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.2)
    
    plt.suptitle('Box Plots: Fraud vs Legitimate (Top 6 Features)', fontsize=16, fontweight='bold', y=1.01, color='#f0883e')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_outlier_boxplots.png', bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print("  📊 Saved: 08_outlier_boxplots.png")

# ── Summary Statistics Export ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  EDA SUMMARY")
print("=" * 70)
print(f"""
  Dataset Size:           {df.shape[0]:,} rows × {df.shape[1]:,} columns
  Target Column:          {target_col}
  Legitimate Txns:        {target_counts.get(0, 0):,} ({target_pcts.get(0, 0):.2f}%)
  Fraudulent Txns:        {target_counts.get(1, 0):,} ({target_pcts.get(1, 0):.2f}%)
  Imbalance Ratio:        {imbalance_ratio:.0f}:1
  Binary Features:        {len(binary_cols)}
  Continuous Features:    {len(continuous_cols)}
  Sparse Features:        {len(sparse_cols)}
  Memory Usage:           {df.memory_usage(deep=True).sum() / 1e9:.2f} GB
  Total Missing Cells:    {total_missing:,} ({total_missing/total_cells*100:.2f}%)
  
  Charts saved to: {OUTPUT_DIR.resolve()}
""")
print("✅ EDA Complete!")
