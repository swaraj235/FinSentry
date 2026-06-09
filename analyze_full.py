import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

target = 'F3924'

# Part 1: Analyze the two categorical features (F3889 and F3891) deeply
key_cats = ['F3889', 'F3891', target]
df_cat = pd.read_csv('dataset/dataset.csv/dataset.csv', usecols=key_cats)

print('='*60)
print('CATEGORICAL FEATURE: F3889 (Account Age/Activity Category)')
print('='*60)
ct = pd.crosstab(df_cat['F3889'], df_cat[target])
ct.columns = ['Legit', 'Fraud']
ct['Total'] = ct.sum(axis=1)
ct['Fraud_Rate%'] = (ct['Fraud'] / ct['Total'] * 100).round(2)
ct = ct.sort_values('Fraud_Rate%', ascending=False)
print(ct.to_string())

print('\n' + '='*60)
print('CATEGORICAL FEATURE: F3891 (Occupation/Account Type)')
print('='*60)
ct2 = pd.crosstab(df_cat['F3891'], df_cat[target])
ct2.columns = ['Legit', 'Fraud']
ct2['Total'] = ct2.sum(axis=1)
ct2['Fraud_Rate%'] = (ct2['Fraud'] / ct2['Total'] * 100).round(2)
ct2 = ct2.sort_values('Fraud_Rate%', ascending=False)
print(ct2.to_string())

# Part 2: Load ALL columns, find top 50 most correlated with target
print('\n\nLoading full dataset for top-feature discovery...')
# Read first row to get all column names
all_cols = pd.read_csv('dataset/dataset.csv/dataset.csv', nrows=0).columns.tolist()
print(f'Total columns: {len(all_cols)}')

# Process in chunks to find correlations
chunk_size = 200  # columns at a time
all_corrs = {}

for start in range(0, len(all_cols), chunk_size):
    end = min(start + chunk_size, len(all_cols))
    cols_chunk = all_cols[start:end]
    if target not in cols_chunk:
        cols_chunk.append(target)
    
    chunk_df = pd.read_csv('dataset/dataset.csv/dataset.csv', usecols=cols_chunk)
    numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        for col in numeric_cols:
            if col != target:
                c = chunk_df[col].corr(chunk_df[target])
                if not np.isnan(c):
                    all_corrs[col] = c
    
    if start % 1000 == 0:
        print(f'  Processed columns {start}-{end}...')

# Sort and display top features
sorted_corrs = sorted(all_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

print('\n' + '='*60)
print('TOP 60 FEATURES MOST CORRELATED WITH TARGET (F3924)')
print('='*60)
for feat, c in sorted_corrs[:60]:
    direction = '+' if c > 0 else '-'
    print(f'  {feat}: r={c:.6f} ({direction})')

# Check which key features appear in top 100
key_features = ['F115','F321','F527','F531','F670','F1692','F2082','F2122',
                'F2582','F2678','F2737','F2956','F3043','F3836','F3887',
                'F3889','F3891','F3894']
top100 = set(f for f, _ in sorted_corrs[:100])
key_in_top = [f for f in key_features if f in top100]
print(f'\nKey features in top 100 correlated: {key_in_top}')

# Features with strongest fraud signal (top 20 positive, top 20 negative)
pos_corrs = [(f, c) for f, c in sorted_corrs if c > 0][:20]
neg_corrs = [(f, c) for f, c in sorted_corrs if c < 0][:20]

print('\n\nTop 20 POSITIVE correlators (fraud-indicating):')
for f, c in pos_corrs:
    print(f'  {f}: {c:.6f}')

print('\nTop 20 NEGATIVE correlators (legitimacy-indicating):')
for f, c in neg_corrs:
    print(f'  {f}: {c:.6f}')

# Part 3: Data type summary
print('\n' + '='*60)
print('FEATURE TYPE SUMMARY')
print('='*60)
full_row = pd.read_csv('dataset/dataset.csv/dataset.csv', nrows=100)
dtypes = full_row.dtypes
print(f'Object/String columns: {(dtypes == "object").sum()}')
print(f'Float columns: {(dtypes == "float64").sum()}')
print(f'Int columns: {(dtypes == "int64").sum()}')

# Find all string/object columns
obj_cols = dtypes[dtypes == 'object'].index.tolist()
print(f'\nString columns: {obj_cols}')

# Load and analyze string columns
if obj_cols:
    str_df = pd.read_csv('dataset/dataset.csv/dataset.csv', usecols=obj_cols + [target])
    for col in obj_cols:
        print(f'\n{col}: {str_df[col].nunique()} unique values')
        ct = pd.crosstab(str_df[col], str_df[target])
        ct.columns = ['Legit', 'Fraud']
        ct['Total'] = ct.sum(axis=1)
        ct['Fraud%'] = (ct['Fraud']/ct['Total']*100).round(2)
        ct = ct.sort_values('Fraud%', ascending=False)
        print(ct.head(10).to_string())
