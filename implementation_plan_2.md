# FinSentry — Revised Analysis & New Ideation

## 🔍 Critical New Discoveries from Dataset Re-Analysis

### Discovery 1: 8 String/Categorical Columns with REAL Domain Semantics

Your dataset is **NOT** fully anonymized! We found **8 interpretable categorical columns** that provide rich banking context:

| Feature | Description | Unique Values | Key Insight |
|---------|-------------|---------------|-------------|
| **F2230** | Transaction/Event Month | 4 (`Sep25`, `Oct25`, `Nov25`, `Dec25`) | ⚠️ **PERFECT separator** — ALL fraud is Sep/Nov/Dec, ALL legit is Oct |
| **F3886** | Account Type | 17 (`Savings`, `Current`, `MSME Medium`, `Agri Adv`, etc.) | Savings = 1.28% fraud, Current = 0.20% |
| **F3888** | Account Opening Date | 4,292 unique dates | Date feature — can derive account age |
| **F3889** | Account Age Band | 7 (`G365D`, `L365D`, `L180D`, `L90D`, `L31D`, `L14D`, `L7D`) | Newer accounts ≠ higher fraud (interesting) |
| **F3890** | Location/Area Type | 4 (`R`=Rural, `SU`=Semi-Urban, `U`=Urban, `M`=Metro) | Rural = 1.44% fraud (highest!), Metro = 0.62% |
| **F3891** | Occupation | 7 (`selfemployed`, `salaried`, `student`, etc.) | **Students = 1.94% fraud** (3x average!) |
| **F3892** | Gender | 3 (`M`, `F`, `O`) | Male = 1.26%, Female = 0.92% |
| **F3893** | Customer Segment | 2 (`RETAIL`, `CORPORATE`) | Retail = 1.18%, Corporate = 0.19% (6x difference!) |

> [!CAUTION]
> **F2230 is a MASSIVE DATA LEAK!** It perfectly separates fraud from legit. This means the training/test split is likely **temporal** — Oct 2025 = training (all legit), Sep/Nov/Dec = test (contains fraud). **Your model MUST NOT use F2230 as a feature.** But this tells you the evaluation is about detecting fraud in future/unseen time periods.

### Discovery 2: F3912 Has r = 0.97 with Target

> [!WARNING]
> **F3912** correlates at **0.969** with F3924 (target). This is almost certainly a **derived label or leak feature** (e.g., a "fraud flag" computed from the target). **DO NOT use F3912 as a model feature.** Using it would give you 97% accuracy trivially but would be disqualified. Similarly watch for F3908 (r=0.097) which may be related.

### Discovery 3: The 18 Bank-Suggested Features Are Individually Weak But Collectively Meaningful

**None of the 18 suggested features appear in the top 100 correlated features!**

However, from the Mann-Whitney tests, 6 of them are **statistically significant**:

| Feature | p-value | Significance | Cohen's d | Behavioral Interpretation |
|---------|---------|--------------|-----------|--------------------------|
| **F115** | 0.0000074 | ★★★ | 0.57 (Medium) | Fraud accounts have HIGHER values (0.72 vs 0.59) — likely a risk/probability score |
| **F670** | 0.0000074 | ★★★ | 0.40 (Small) | Binary flag; fraud_rate=2.29% when=1 vs 0.75% when=0 — likely a flag for some activity |
| **F2082** | 0.0002 | ★★★ | — | ALL fraud accounts have 0.0; legit accounts have non-zero — likely a trust/history metric |
| **F2122** | 0.0002 | ★★★ | -0.43 (Small) | Fraud accounts near zero (0.005 vs 0.046) — possibly a stability/age ratio |
| **F2956** | 0.0014 | ★★ | -0.29 (Small) | Fraud median=41 vs legit median=64 — likely a count metric (fewer transactions?) |
| **F1692** | 0.015 | ★ | -0.23 (Small) | Fraud accounts have LOWER values — possibly a counter |

**Key insight:** These features represent **behavioral patterns that banks actually use** (transaction counts, velocity metrics, risk scores, stability indicators). Individually weak but the PS tells you they matter — use them in combination.

### Discovery 4: Top Predictive Features Across the Full Dataset

The truly powerful features (excluding leaks) cluster into groups:

| Group | Features | Correlation Range | Likely Meaning |
|-------|----------|-------------------|----------------|
| **F2506-F2507** | 2 features | r ≈ 0.185 | Appear identical — possibly a binary event flag |
| **F2408-F2409** | 2 features | r ≈ 0.157 | Another paired event flag |
| **F515, F518** | 2 features | r ≈ 0.127-0.137 | Transaction pattern features |
| **F81-F84** | 4 features | r ≈ 0.117 | Highly correlated group — likely same source |
| **F253-F287** | Block | r ≈ 0.106-0.113 | Transaction aggregates |
| **F2502-F2503** | 2 features | r ≈ -0.098 | Legitimacy indicators (negative correlation) |

---

## 🧠 New Ideation Based on Full Problem Statement

### Idea 1: **Temporal Anomaly Detection** (HIGH IMPACT)

Since F2230 reveals the data has a temporal dimension (Sep–Dec 2025), and F3888 gives account opening dates:

- **Account age at transaction time**: `transaction_month - account_opening_date` → New accounts involved in fraud?
- **Temporal velocity**: Combine F2956 (count-like) with F3888 to compute "transactions per day since account opening"
- **Dormancy detection**: Old accounts (F3888 years ago) with sudden activity spikes → classic mule behavior
- **Seasonal pattern**: Banks see fraud spikes during festivals (Diwali ≈ Oct/Nov) — matches your data!

```python
# Feature engineering example
df['account_age_days'] = (pd.to_datetime('2025-10-01') - pd.to_datetime(df['F3888'])).dt.days
df['is_new_account'] = (df['account_age_days'] < 90).astype(int)
df['txn_velocity'] = df['F2956'] / (df['account_age_days'] + 1)  # transactions per day
df['is_dormant_reactivated'] = ((df['account_age_days'] > 365) & (df['F3889'].isin(['L7D','L14D','L31D']))).astype(int)
```

### Idea 2: **Occupation-Risk Profiling** (MEDIUM IMPACT)

Students have **3x the fraud rate** of self-employed. This aligns with real-world mule recruitment:

- Students are disproportionately targeted for mule account recruitment (money for "easy jobs")
- Create **occupation risk weights**: `student=3x`, `agriculture=2x`, `retired=1.5x`, etc.
- **Interaction features**: `student × high_F115 × rural_location` = very high risk combination

```python
occupation_risk = {'student': 1.94, 'agriculture': 1.26, 'retired': 1.04, 
                   'salaried': 0.73, 'selfemployed': 0.66, 'housewife': 0.45, 'others': 0.0}
df['occupation_risk'] = df['F3891'].map(occupation_risk)
df['student_rural'] = ((df['F3891']=='student') & (df['F3890']=='R')).astype(int)
df['student_high_risk'] = ((df['F3891']=='student') & (df['F115'] > 0.7)).astype(int)
```

### Idea 3: **Behavioral Deviation Scoring** (HIGH IMPACT — Differentiator)

The PS specifically mentions "behavioral and transactional patterns." Use the 18 suggested features to build a **deviation score**:

- For each of the 18 features, compute how far a sample is from the **population mean** in units of standard deviation
- Weight by the Mann-Whitney significance
- Sum into a composite "behavioral anomaly score"

```python
# Z-score deviation across key features, weighted by significance
key_numeric = ['F115','F670','F2082','F2122','F2956','F1692','F321','F527','F531']
weights = {'F115': 5, 'F670': 5, 'F2082': 4, 'F2122': 4, 'F2956': 3, 'F1692': 2,
           'F321': 1, 'F527': 1, 'F531': 1}

for f in key_numeric:
    df[f'{f}_zscore'] = (df[f] - df[f].mean()) / df[f].std()

df['behavioral_deviation'] = sum(weights[f] * df[f'{f}_zscore'].abs() for f in key_numeric)
```

### Idea 4: **Segment-Aware Risk Model** (MEDIUM IMPACT)

Retail accounts are **6x more likely** to be fraud than Corporate. Build segment-specific sub-models:

- **Retail model**: Focus on individual behavioral features (occupation, gender, location)
- **Corporate model**: Focus on transactional volume features
- **Combined**: Segment probability × model output

### Idea 5: **Missingness-as-Signal Features** (HIGH IMPACT — Often Overlooked)

Key observation: **Fraud accounts have LESS missingness** in most features:

| Feature | Legit NA% | Fraud NA% | Interpretation |
|---------|-----------|-----------|----------------|
| F115 | 4.0% | 0.0% | Fraud accounts always have this filled |
| F527 | 8.7% | 1.2% | ↓ |
| F531 | 7.1% | 0.0% | ↓ |
| F2582 | 37.7% | 12.3% | Major difference! |
| F2956 | 11.4% | 2.5% | ↓ |
| F2678 | 28.2% | **40.7%** | **Opposite pattern!** |
| F3043 | 64.0% | **82.7%** | **Fraud has MORE missing** |

→ Create binary `_has_value` features for high-differential columns
→ Create `total_missing_count` across all features as a meta-feature

```python
high_diff_cols = ['F115','F527','F531','F2582','F2956','F2678','F3043']
for col in high_diff_cols:
    df[f'{col}_has_value'] = df[col].notna().astype(int)
df['total_key_missing'] = df[high_diff_cols].isna().sum(axis=1)
```

### Idea 6: **Feature Interaction Mining** (HIGH IMPACT)

Create multiplicative interactions between the strongest features:

```python
# Top interactions to try
df['F2506_x_F515'] = df['F2506'] * df['F515']
df['F115_x_F670'] = df['F115'] * df['F670']  # risk score × flag
df['F2082_zero_x_student'] = ((df['F2082']==0) & (df['F3891']=='student')).astype(int)
df['retail_savings_student'] = ((df['F3893']=='RETAIL') & (df['F3886']=='Savings') & (df['F3891']=='student')).astype(int)
```

### Idea 7: **PCA on Feature Groups** (MEDIUM IMPACT)

The correlated feature pairs (F2506/F2507, F2408/F2409, F81-F84, etc.) suggest latent dimensions:

- Run PCA on the top 100 correlated features → extract 10-20 components
- Run separate PCA on the 18 bank-suggested features → extract 3-5 components
- Use these as additional features in the ensemble

---

## 📐 Revised Model Architecture

### Tier 1: Quick Wins (Build First)

**XGBoost with smart feature engineering**

```
Input Features:
├── Top 50 correlated numeric features (excluding F3912, F2230)
├── 18 bank-suggested features
├── 8 categorical features (one-hot encoded)
├── Engineered features:
│   ├── account_age_days (from F3888)
│   ├── txn_velocity (F2956 / account_age)
│   ├── occupation_risk_score
│   ├── behavioral_deviation_score
│   ├── missingness meta-features
│   └── top feature interactions
└── Target: F3924
```

### Tier 2: Anomaly Layer (Build Second)

**Isolation Forest + Autoencoder** trained only on legit Oct 2025 data:
- Any sample that looks different from Oct 2025 "normal" patterns → high anomaly score
- This directly addresses "the system should leverage ML techniques for anomaly detection"

### Tier 3: Intelligent Alert System (Build Third)

Map model outputs to a **3-tier alert system** (directly from PS: "intelligent alert generation"):

| Risk Score | Alert Level | Action |
|------------|-------------|--------|
| > 0.9 | 🔴 **CRITICAL** | Auto-block, immediate investigation |
| 0.7 – 0.9 | 🟡 **HIGH** | Flag for manual review within 1 hour |
| 0.5 – 0.7 | 🟠 **MEDIUM** | Enhanced monitoring, daily review |
| < 0.5 | 🟢 **LOW** | Normal processing |

### Tier 4: Explainability (Hackathon Wow Factor)

**SHAP + Natural Language Explanations:**
```
⚠️ Alert: Account flagged as HIGH RISK (Score: 0.87)
Top reasons:
  1. Student account with high risk indicator (F115=0.91) — contributes +0.23
  2. Rural location with zero trust score (F2082=0.0) — contributes +0.18
  3. Savings account in RETAIL segment — contributes +0.12
  4. Account opened 45 days ago with 3 transactions — unusual velocity
```

---

## ⚠️ Features to EXCLUDE (Data Leaks)

| Feature | Why Exclude |
|---------|-------------|
| **F3912** | r=0.97 with target — almost certainly derived from the label |
| **F2230** | Perfect temporal separation — fraud only in non-Oct months |
| **Unnamed: 0** | Index column with r=0.16 — row ordering artifact |

---

## 🎯 Updated Execution Priority

| # | Task | Time | Impact |
|---|------|------|--------|
| 1 | Feature engineering (account age, occupation risk, missingness, interactions) | 2h | ★★★★★ |
| 2 | XGBoost with engineered features + SMOTE + 5-fold stratified CV | 2h | ★★★★★ |
| 3 | SHAP explainability + alert tier mapping | 1h | ★★★★ |
| 4 | Isolation Forest anomaly detection layer | 1h | ★★★★ |
| 5 | Autoencoder anomaly detector (train on legit only) | 1.5h | ★★★ |
| 6 | Dashboard with risk scoring + SHAP explanations | 2h | ★★★★ |
| 7 | Ensemble stacking (XGBoost + IF + Autoencoder) | 1h | ★★★ |

---

## Open Questions

> [!IMPORTANT]
> 1. **What is the evaluation metric?** Does the hackathon specify AUPRC, F1, AUC-ROC, or something else?
> 2. **Is there a separate test set** or do you submit predictions on a held-out portion of this CSV?
> 3. **How much time do you have left** in the hackathon? This determines scope.
> 4. **Do you need a live demo / dashboard**, or is a Jupyter notebook with results sufficient?
> 5. **Should we start coding now?** I can begin with the feature engineering pipeline immediately.
