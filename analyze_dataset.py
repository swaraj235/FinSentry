import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Key features from the problem statement + target
key_features = ['F115','F321','F527','F531','F670','F1692','F2082','F2122',
                'F2582','F2678','F2737','F2956','F3043','F3836','F3887',
                'F3889','F3891','F3894']
target = 'F3924'
cols_to_load = key_features + [target]

print('Loading only key columns...')
df = pd.read_csv('dataset/dataset.csv/dataset.csv', usecols=cols_to_load)
print(f'Shape: {df.shape}')
print(f'\nTarget (F3924) distribution:')
print(df[target].value_counts())
print(f'Target NA: {df[target].isna().sum()}')

print('\n' + '='*60)
print('KEY FEATURES DEEP ANALYSIS')
print('='*60)

for f in key_features:
    col = df[f]
    na_pct = col.isna().mean() * 100
    nunique = col.nunique()
    
    print(f'\n--- {f} ---')
    print(f'  dtype={col.dtype}, NA={na_pct:.1f}%, unique={nunique}')
    
    if col.dtype in ['float64','int64','float32','int32']:
        print(f'  min={col.min()}, max={col.max()}, median={col.median()}, mean={col.mean():.4f}, std={col.std():.4f}')
        
        # Distribution breakdown
        fraud = df[df[target]==1][f]
        legit = df[df[target]==0][f]
        
        print(f'  LEGIT: mean={legit.mean():.4f}, median={legit.median()}, std={legit.std():.4f}')
        print(f'  FRAUD: mean={fraud.mean():.4f}, median={fraud.median()}, std={fraud.std():.4f}')
        
        # Effect size
        if fraud.std() > 0 and legit.std() > 0:
            pooled = np.sqrt((fraud.std()**2 + legit.std()**2)/2)
            d = (fraud.mean() - legit.mean()) / pooled
            print(f'  Cohen d effect size: {d:.4f} ({"LARGE" if abs(d) > 0.8 else "MEDIUM" if abs(d) > 0.5 else "SMALL" if abs(d) > 0.2 else "NEGLIGIBLE"})')
        
        # Percentile comparison
        for p in [25, 50, 75, 90, 95, 99]:
            lp = np.nanpercentile(legit, p)
            fp = np.nanpercentile(fraud, p)
            print(f'  P{p}: Legit={lp:.4f}, Fraud={fp:.4f}')
        
        # Value distribution type inference
        unique_vals = col.dropna().unique()
        if len(unique_vals) <= 10:
            print(f'  CATEGORICAL/ORDINAL: {sorted(unique_vals)}')
            print(f'  Value counts:')
            for v in sorted(unique_vals):
                total = (col == v).sum()
                fraud_count = (df[df[target]==1][f] == v).sum()
                fraud_rate = fraud_count / total * 100 if total > 0 else 0
                print(f'    {v}: n={total}, fraud={fraud_count}, fraud_rate={fraud_rate:.2f}%')
        elif col.min() >= 0 and col.max() <= 1:
            print(f'  Likely NORMALIZED/PROBABILITY feature')
        else:
            print(f'  Likely RAW NUMERIC feature')
    else:
        print(f'  String/Object type')
        print(f'  Top 10 values: {dict(list(col.value_counts().head(10).items()))}')

    # Missingness difference
    fraud_na = df[df[target]==1][f].isna().mean()
    legit_na = df[df[target]==0][f].isna().mean()
    print(f'  Missingness: Legit={legit_na:.3f}, Fraud={fraud_na:.3f}')

# Correlations
print('\n' + '='*60)
print('CORRELATIONS WITH TARGET')
print('='*60)
numeric_key = [f for f in key_features if df[f].dtype in ['float64','int64','float32','int32']]
corrs = []
for f in numeric_key:
    c = df[f].corr(df[target])
    corrs.append((f, c))
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for feat, c in corrs:
    strength = 'STRONG' if abs(c) > 0.3 else 'MODERATE' if abs(c) > 0.1 else 'WEAK'
    print(f'  {feat}: r={c:.6f} ({strength})')

# Cross-correlations
print('\n' + '='*60)
print('CROSS-CORRELATIONS AMONG KEY FEATURES (|r| > 0.5)')
print('='*60)
key_corr = df[numeric_key].corr()
pairs = []
for i in range(len(numeric_key)):
    for j in range(i+1, len(numeric_key)):
        c = key_corr.iloc[i,j]
        if abs(c) > 0.5:
            pairs.append((numeric_key[i], numeric_key[j], c))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for f1, f2, c in pairs:
    print(f'  {f1} <-> {f2}: r={c:.4f}')

# Statistical tests
from scipy import stats
print('\n' + '='*60)
print('MANN-WHITNEY U TESTS (Fraud vs Legit)')
print('='*60)
for f in numeric_key:
    fraud_vals = df[df[target]==1][f].dropna()
    legit_vals = df[df[target]==0][f].dropna()
    if len(fraud_vals) > 5 and len(legit_vals) > 5:
        stat, pval = stats.mannwhitneyu(fraud_vals, legit_vals, alternative='two-sided')
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        print(f'  {f}: p={pval:.8f} {sig}')
