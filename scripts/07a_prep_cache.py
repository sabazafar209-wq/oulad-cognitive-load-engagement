import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd, numpy as np, torch, pickle, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

t0 = time.time()
labels = pd.read_csv('data/oulad_labeled_v2_clickstream.csv')[
    ['id_student','code_module','code_presentation','cognitive_load_label']]
weekly = pd.read_csv('data/oulad_weekly_clicks.csv')
key = ['id_student', 'code_module', 'code_presentation']
weekly = weekly.merge(labels, on=key, how='inner')

FEATS = ['week_clicks', 'week_sites', 'week_events']
scaler = StandardScaler()
weekly[FEATS] = scaler.fit_transform(weekly[FEATS])

sequences, seq_labels = [], []
for _, g in weekly.groupby(key):
    g = g.sort_values('week')
    sequences.append(torch.tensor(g[FEATS].values, dtype=torch.float32))
    seq_labels.append(g['cognitive_load_label'].iloc[0])

le = LabelEncoder()
y = le.fit_transform(seq_labels)
classes = le.classes_
print("Classes:", classes, f"({time.time()-t0:.1f}s)")

idx_train, idx_test = train_test_split(np.arange(len(sequences)), test_size=0.2, random_state=42, stratify=y)
train_seqs_base = [sequences[i] for i in idx_train]
train_labs_base = [int(y[i]) for i in idx_train]
test_seqs = [sequences[i] for i in idx_test]
test_labs = [int(y[i]) for i in idx_test]
print("Train class counts:", np.bincount(train_labs_base))

def oversample_sequences(seqs, labs, target_class_idx, target_count, jitter_std=0.15, seed=42):
    rng = np.random.RandomState(seed)
    seqs = list(seqs); labs = list(labs)
    minority_idxs = [i for i, l in enumerate(labs) if l == target_class_idx]
    n_needed = max(0, target_count - len(minority_idxs))
    for _ in range(n_needed):
        src = seqs[rng.choice(minority_idxs)].clone()
        synth = src + torch.randn_like(src) * jitter_std
        seqs.append(synth); labs.append(target_class_idx)
    return seqs, labs

high_idx = list(classes).index('high')
medium_count = np.bincount(train_labs_base)[list(classes).index('medium')]
train_seqs_over, train_labs_over = oversample_sequences(train_seqs_base, train_labs_base, high_idx, medium_count)
print("Oversampled train class counts:", np.bincount(train_labs_over))

with open('data/lstm_imbalance_cache.pkl', 'wb') as f:
    pickle.dump({
        'classes': classes,
        'train_seqs_base': train_seqs_base, 'train_labs_base': train_labs_base,
        'train_seqs_over': train_seqs_over, 'train_labs_over': train_labs_over,
        'test_seqs': test_seqs, 'test_labs': test_labs,
    }, f)
print(f"Cached in {time.time()-t0:.1f}s total")
