"""
Phase 3 (rebuild): True clickstream-derived behavioral features from the real
studentVle.csv (10,655,280 rows). Replaces the assessment-timing proxies used
in v1 with genuine interaction-based idle-time, session, and burstiness features.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import numpy as np

DATA = 'data/raw/'
UP = 'data/raw/'

vle = pd.read_csv(UP + 'studentVle.csv')
si = pd.read_csv(DATA + 'studentInfo.csv')
sa = pd.read_csv(DATA + 'studentAssessment.csv')
sr = pd.read_csv(DATA + 'studentRegistration.csv')
asmt = pd.read_csv(DATA + 'assessments.csv')

key = ['id_student', 'code_module', 'code_presentation']

# ---------- 1. Per-student-module daily activity ----------
daily = vle.groupby(key + ['date'])['sum_click'].sum().reset_index()
daily = daily.sort_values(key + ['date'])

# ---------- 2. Idle-time / gap features (the whole point of this rebuild) ----------
def gap_stats(g):
    days = g['date'].values
    if len(days) < 2:
        return pd.Series({'idle_gap_max': 0, 'idle_gap_mean': 0, 'idle_gap_std': 0, 'active_days': len(days)})
    gaps = np.diff(np.sort(days))
    return pd.Series({
        'idle_gap_max': gaps.max(),
        'idle_gap_mean': gaps.mean(),
        'idle_gap_std': gaps.std() if len(gaps) > 1 else 0,
        'active_days': len(days),
    })

gap_feats = daily.groupby(key).apply(gap_stats, include_groups=False).reset_index()

# ---------- 3. Click intensity / volume features ----------
click_feats = vle.groupby(key).agg(
    total_clicks=('sum_click', 'sum'),
    avg_click_per_event=('sum_click', 'mean'),
    n_distinct_sites=('id_site', 'nunique'),
    n_events=('sum_click', 'count'),
).reset_index()
click_feats['avg_daily_clicks'] = click_feats['total_clicks'] / gap_feats.set_index(key).reindex(
    click_feats.set_index(key).index)['active_days'].values

# ---------- 4. Pre-course vs in-course activity split ----------
vle['is_precourse'] = (vle['date'] < 0).astype(int)
precourse = vle[vle.is_precourse == 1].groupby(key)['sum_click'].sum().reset_index().rename(columns={'sum_click': 'precourse_clicks'})

# ---------- 5. Burstiness: fraction of total clicks concentrated in top-10% busiest days ----------
def burstiness(g):
    daily_sums = g.groupby('date')['sum_click'].sum().sort_values(ascending=False)
    if len(daily_sums) == 0 or daily_sums.sum() == 0:
        return 0.0
    top_n = max(1, int(np.ceil(0.1 * len(daily_sums))))
    return daily_sums.iloc[:top_n].sum() / daily_sums.sum()

burst = vle.groupby(key).apply(burstiness, include_groups=False).reset_index(name='burstiness')

# ---------- 6. Pre-assessment cramming signal: clicks in 3 days before each assessment deadline ----------
# Vectorized merge instead of a per-row Python loop (that would be far too slow over 10.6M rows).
sa2 = sa.merge(asmt[['id_assessment','code_module','code_presentation','date']], on='id_assessment', how='left')
sa2 = sa2.dropna(subset=['date'])[key + ['id_assessment', 'date']].rename(columns={'date': 'deadline'})

merged = sa2.merge(daily, on=key, how='inner', suffixes=('', '_activity'))
mask = (merged['date'] >= merged['deadline'] - 3) & (merged['date'] < merged['deadline'])
cram_df = merged[mask].groupby(key + ['id_assessment'])['sum_click'].sum().reset_index(name='preassessment_clicks')
cram_agg = cram_df.groupby(key)['preassessment_clicks'].mean().reset_index().rename(columns={'preassessment_clicks': 'avg_preassessment_clicks'})

# ---------- Merge everything ----------
df = gap_feats.merge(click_feats, on=key, how='outer')
df = df.merge(precourse, on=key, how='left')
df = df.merge(burst, on=key, how='left')
df = df.merge(cram_agg, on=key, how='left')
df['precourse_clicks'] = df['precourse_clicks'].fillna(0)
df['avg_preassessment_clicks'] = df['avg_preassessment_clicks'].fillna(0)

print("Clickstream feature table:", df.shape)
df.to_csv('data/oulad_clickstream_features.csv', index=False)
print(df.head().to_string())
print(df.describe())
