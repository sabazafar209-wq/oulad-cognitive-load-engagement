"""
Phase 8a: Build a strictly clickstream-only cognitive-load label set.
Unlike v2's label set (which mixed assessment struggle features + idle-time
features), this uses ONLY features derivable from studentVle.csv, with zero
assessment-table involvement -- not even avg_preassessment_clicks, since
computing that requires assessment deadline dates as a reference point.
This is the label source for the "reverse" disjoint-feature test:
LSTM fed per-assessment sequences -> predicting a purely clickstream-defined label.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

click = pd.read_csv('data/oulad_clickstream_features.csv')
prior = pd.read_csv('data/oulad_behavioral_features.csv')  # only for final_result join
key = ['id_student', 'code_module', 'code_presentation']

df = click.merge(prior[key + ['final_result']], on=key, how='inner')

# Pure clickstream features only -- NO avg_preassessment_clicks (needs assessment dates)
CLICK_ONLY_FEATURES = ['idle_gap_max', 'idle_gap_mean', 'idle_gap_std', 'active_days',
                        'total_clicks', 'n_distinct_sites', 'avg_daily_clicks',
                        'precourse_clicks', 'burstiness']
for c in CLICK_ONLY_FEATURES:
    df[c] = df[c].fillna(0)

X = StandardScaler().fit_transform(df[CLICK_ONLY_FEATURES])
km = KMeans(n_clusters=3, random_state=42, n_init=10)
raw = km.fit_predict(X)
sil = silhouette_score(X, raw)
print("Clickstream-only cognitive load silhouette:", round(sil, 3))

# Order clusters by a DIRECTIONALLY-CORRECT composite, not a naive feature average:
# idle-gap and burstiness features indicate MORE load when high; activity/breadth
# features indicate LESS load when high. A plain average of standardized features
# would incorrectly treat "highly active" as "high load" since it's numerically large.
X_df = pd.DataFrame(X, columns=CLICK_ONLY_FEATURES)
bad_direction = ['idle_gap_max', 'idle_gap_mean', 'idle_gap_std', 'burstiness']
good_direction = ['active_days', 'total_clicks', 'n_distinct_sites', 'avg_daily_clicks', 'precourse_clicks']
composite = X_df[bad_direction].mean(axis=1) - X_df[good_direction].mean(axis=1)

order = composite.groupby(raw).mean().sort_values().index.tolist()
remap = {old: new for new, old in enumerate(order)}
df['cognitive_load_clickonly'] = pd.Series(raw).map(remap)
df['cognitive_load_clickonly_label'] = df['cognitive_load_clickonly'].map({0: 'low', 1: 'medium', 2: 'high'})

print(df['cognitive_load_clickonly_label'].value_counts())
print("\nExternal validation vs final_result:")
print(pd.crosstab(df['cognitive_load_clickonly_label'], df['final_result'], normalize='index').round(2))

df[key + ['cognitive_load_clickonly_label']].to_csv(
    'data/oulad_labels_clickonly.csv', index=False)
print("\nSaved. Shape:", df.shape)
