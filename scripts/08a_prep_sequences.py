"""
Phase 8a: Build the weekly clickstream sequences ONCE, decoupled from any
label file. This lets us pair the same sequences with either:
  - v1 labels (assessment-only clustering)  -> the strict-disjoint test
  - v2 labels (assessment+clickstream clustering) -> the original entangled test
without rebuilding the sequence tensors each time.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd, numpy as np, torch, pickle, time
from sklearn.preprocessing import StandardScaler

t0 = time.time()
weekly = pd.read_csv('data/oulad_weekly_clicks.csv')
key = ['id_student', 'code_module', 'code_presentation']
FEATS = ['week_clicks', 'week_sites', 'week_events']

scaler = StandardScaler()
weekly[FEATS] = scaler.fit_transform(weekly[FEATS])

sequences, keys = [], []
for k, g in weekly.groupby(key):
    g = g.sort_values('week')
    sequences.append(torch.tensor(g[FEATS].values, dtype=torch.float32))
    keys.append(k)

print(f"Built {len(sequences)} clickstream-only sequences in {time.time()-t0:.1f}s")
with open('data/lstm_sequences_only.pkl', 'wb') as f:
    pickle.dump({'sequences': sequences, 'keys': keys}, f)
print("Saved.")
