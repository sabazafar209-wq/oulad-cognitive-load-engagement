"""
Fixes issue #1: the v2 LSTM evaluation was not independent of the labeling
process -- both used clickstream-derived signal. This script builds TWO
strictly disjoint setups:

  Setup 1 (assess->click): cluster labels generated ONLY from assessment/
    registration features (no clickstream at all). LSTM is fed ONLY
    clickstream weekly sequences (no assessment info at all).
  Setup 2 (click->assess): cluster labels generated ONLY from clickstream
    features (no assessment info at all). LSTM is fed ONLY per-assessment
    event sequences (score/error/lateness -- no clickstream info at all).

If the LSTM still predicts meaningfully above chance in both directions,
that is real evidence of an independent behavioral signal rather than the
model recovering its own label-generating features.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

prior = pd.read_csv('data/oulad_behavioral_features.csv')
click = pd.read_csv('data/oulad_clickstream_features.csv')
key = ['id_student', 'code_module', 'code_presentation']
df = prior.merge(click, on=key, how='inner')

for c in ['idle_gap_max','idle_gap_mean','idle_gap_std','active_days','total_clicks',
          'avg_click_per_event','n_distinct_sites','n_events','avg_daily_clicks',
          'precourse_clicks','burstiness','avg_preassessment_clicks']:
    df[c] = df[c].fillna(0)

# ---------- STRICTLY assessment-only features (zero clickstream) ----------
assess_only_feats = ['avg_error_rate', 'pct_late', 'avg_days_late', 'max_days_late',
                      'num_of_prev_attempts', 'std_score']
X_a = StandardScaler().fit_transform(df[assess_only_feats])
km_a = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_a = km_a.fit_predict(X_a)
sil_a = silhouette_score(X_a, raw_a)
comp_a = X_a.mean(axis=1)
order_a = pd.Series(comp_a).groupby(raw_a).mean().sort_values().index.tolist()
remap_a = {old: new for new, old in enumerate(order_a)}
df['label_assess_only'] = pd.Series(raw_a).map(remap_a).map({0: 'low', 1: 'medium', 2: 'high'})
print(f"[assess-only labels] silhouette={sil_a:.3f}")
print(df['label_assess_only'].value_counts())

# ---------- STRICTLY clickstream-only features (zero assessment info) ----------
click_only_feats = ['idle_gap_max', 'idle_gap_mean', 'idle_gap_std', 'burstiness',
                     'avg_preassessment_clicks', 'total_clicks', 'active_days']
X_c = StandardScaler().fit_transform(df[click_only_feats])
km_c = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_c = km_c.fit_predict(X_c)
sil_c = silhouette_score(X_c, raw_c)
comp_c = X_c.mean(axis=1)
order_c = pd.Series(comp_c).groupby(raw_c).mean().sort_values().index.tolist()
remap_c = {old: new for new, old in enumerate(order_c)}
df['label_click_only'] = pd.Series(raw_c).map(remap_c).map({0: 'low', 1: 'medium', 2: 'high'})
print(f"\n[click-only labels] silhouette={sil_c:.3f}")
print(df['label_click_only'].value_counts())

# external validation against final_result for both
print("\n--- assess-only labels vs final_result ---")
print(pd.crosstab(df['label_assess_only'], df['final_result'], normalize='index').round(2))
print("\n--- click-only labels vs final_result ---")
print(pd.crosstab(df['label_click_only'], df['final_result'], normalize='index').round(2))

# agreement between the two independent label sets (sanity: do they agree more than chance?)
agree = (df['label_assess_only'] == df['label_click_only']).mean()
print(f"\nRaw agreement between assess-only and click-only labels: {agree:.3f} (chance ~= 0.33-0.43 depending on class balance)")

out_cols = key + ['label_assess_only', 'label_click_only', 'final_result'] + assess_only_feats + click_only_feats
df[out_cols].to_csv('data/oulad_disjoint_labels.csv', index=False)
print("\nSaved oulad_disjoint_labels.csv:", df[out_cols].shape)
