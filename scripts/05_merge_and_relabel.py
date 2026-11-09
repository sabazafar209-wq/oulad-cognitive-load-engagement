"""
Phase 3b: Merge real clickstream features with the prior assessment-based
features, re-derive cognitive_load / engagement labels using the richer
feature set, and externally validate against final_result again.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

click = pd.read_csv('data/oulad_clickstream_features.csv')
prior = pd.read_csv('data/oulad_behavioral_features.csv')

key = ['id_student', 'code_module', 'code_presentation']
df = prior.merge(click, on=key, how='inner')
print("Merged shape:", df.shape)

# fill any remaining gaps
for c in ['idle_gap_max','idle_gap_mean','idle_gap_std','active_days','total_clicks',
          'avg_click_per_event','n_distinct_sites','n_events','avg_daily_clicks',
          'precourse_clicks','burstiness','avg_preassessment_clicks']:
    df[c] = df[c].fillna(0)

# ---- Re-derive Cognitive Load using REAL idle-time + cramming signals now ----
# High load = high idle gaps (disengagement/struggle pauses), high burstiness/cramming,
# high error rate, high prior attempts, high score volatility.
cl_features = ['avg_error_rate', 'pct_late', 'num_of_prev_attempts', 'std_score',
               'idle_gap_max', 'idle_gap_mean', 'burstiness', 'avg_preassessment_clicks']
X_cl = StandardScaler().fit_transform(df[cl_features])
km = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_cl = km.fit_predict(X_cl)
sil_cl = silhouette_score(X_cl, raw_cl)
print("Cognitive load silhouette (with clickstream):", round(sil_cl, 3))

composite = X_cl.mean(axis=1)
order = pd.Series(composite).groupby(raw_cl).mean().sort_values().index.tolist()
remap = {old: new for new, old in enumerate(order)}
df['cognitive_load'] = pd.Series(raw_cl).map(remap)
df['cognitive_load_label'] = df['cognitive_load'].map({0: 'low', 1: 'medium', 2: 'high'})

# ---- Re-derive Engagement using REAL clicks/active days now ----
eng_features = ['n_assessments', 'total_banked', 'active_days', 'total_clicks', 'n_distinct_sites']
X_eng = StandardScaler().fit_transform(df[eng_features])
X_eng_persist = np.column_stack([X_eng, 1 - df['withdrew'].values.reshape(-1,1)])
km2 = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_eng = km2.fit_predict(X_eng_persist)
sil_eng = silhouette_score(X_eng_persist, raw_eng)
print("Engagement silhouette (with clickstream):", round(sil_eng, 3))

composite2 = X_eng_persist.mean(axis=1)
order2 = pd.Series(composite2).groupby(raw_eng).mean().sort_values().index.tolist()
remap2 = {old: new for new, old in enumerate(order2)}
df['engagement'] = pd.Series(raw_eng).map(remap2)
df['engagement_label'] = df['engagement'].map({0: 'low', 1: 'medium', 2: 'high'})

print("\nCognitive load distribution:\n", df['cognitive_load_label'].value_counts())
print("\nEngagement distribution:\n", df['engagement_label'].value_counts())

print("\n--- External validation vs final_result ---")
print(pd.crosstab(df['cognitive_load_label'], df['final_result'], normalize='index').round(2))
print(pd.crosstab(df['engagement_label'], df['final_result'], normalize='index').round(2))

print("\n--- Cluster feature profiles (cognitive load) ---")
print(df.groupby('cognitive_load_label')[cl_features].mean().round(2))

df.to_csv('data/oulad_labeled_v2_clickstream.csv', index=False)
print("\nSaved v2 dataset:", df.shape)
