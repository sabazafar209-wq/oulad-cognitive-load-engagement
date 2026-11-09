"""
Phase 7: Address the precision/recall trade-off on the high-cognitive-load
minority class head-on. Compares four training regimes on the SAME
train/test split (weekly clickstream sequences, v2 labels):

  A. weighted    -- class-weighted cross-entropy (current v2 baseline)
  B. focal       -- focal loss (Lin et al., 2017), forces focus on hard/
                    minority examples via a (1-pt)^gamma modulating term
  C. oversample  -- SMOTE-style sequence oversampling: minority-class
                    sequences are duplicated with small Gaussian jitter
                    (since standard SMOTE interpolates fixed-length feature
                    vectors and cannot operate directly on variable-length
                    time series), then trained with plain CE
  D. focal+over  -- combination of B and C

All four are evaluated on an identical, untouched test set for a fair
comparison, and only the training set is touched by oversampling.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, time, sys, copy

torch.manual_seed(42)
np.random.seed(42)

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
print("Classes:", classes, f"({time.time()-t0:.1f}s)", flush=True)

idx_train, idx_test = train_test_split(np.arange(len(sequences)), test_size=0.2, random_state=42, stratify=y)
train_seqs_base = [sequences[i] for i in idx_train]
train_labs_base = [y[i] for i in idx_train]
test_seqs = [sequences[i] for i in idx_test]
test_labs = [y[i] for i in idx_test]

print("Train class counts:", np.bincount(train_labs_base), flush=True)

# ---------- SMOTE-style sequence oversampling (jitter-based) ----------
def oversample_sequences(seqs, labs, target_class_idx, target_count, jitter_std=0.15, seed=42):
    rng = np.random.RandomState(seed)
    seqs = list(seqs); labs = list(labs)
    minority_idxs = [i for i, l in enumerate(labs) if l == target_class_idx]
    current = len(minority_idxs)
    n_needed = max(0, target_count - current)
    for _ in range(n_needed):
        # pick two random minority sequences and blend one with jitter (SMOTE-like interpolation
        # adapted for variable length: jitter a randomly chosen real minority sequence rather than
        # interpolate mismatched lengths directly)
        src_idx = rng.choice(minority_idxs)
        src = seqs[src_idx].clone()
        noise = torch.randn_like(src) * jitter_std
        synth = src + noise
        seqs.append(synth)
        labs.append(target_class_idx)
    return seqs, labs

high_idx = list(classes).index('high')
medium_count = np.bincount(train_labs_base)[list(classes).index('medium')]
train_seqs_over, train_labs_over = oversample_sequences(
    train_seqs_base, train_labs_base, high_idx, target_count=medium_count)
print("Oversampled train class counts:", np.bincount(train_labs_over), flush=True)

# ---------- Focal loss ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

# ---------- Model / data plumbing (shared across all 4 regimes) ----------
class SeqDataset(Dataset):
    def __init__(self, seqs, labs):
        self.seqs = seqs; self.labs = labs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labs[i]

def collate(batch):
    seqs, labs = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True)
    labs = torch.tensor(labs, dtype=torch.long)
    return padded, lens, labs

class LoadLSTM(nn.Module):
    def __init__(self, input_size=3, hidden=48, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, n_classes)
    def forward(self, x, lens):
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(self.dropout(h_n[-1]))

device = 'cpu'
test_ds = SeqDataset(test_seqs, test_labs)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

def class_weights_for(labs):
    counts = np.bincount(labs)
    w = 1.0 / counts
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)

def run_regime(name, train_seqs, train_labs, use_focal, epochs=15):
    print(f"\n--- Regime: {name} ---", flush=True)
    torch.manual_seed(42)
    train_ds = SeqDataset(train_seqs, train_labs)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    model = LoadLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    w = class_weights_for(train_labs).to(device)
    criterion = FocalLoss(alpha=w, gamma=2.0) if use_focal else nn.CrossEntropyLoss(weight=w)

    for epoch in range(epochs):
        et0 = time.time()
        model.train()
        total_loss = 0
        for x, lens, labs in train_loader:
            x, labs = x.to(device), labs.to(device)
            opt.zero_grad()
            out = model(x, lens)
            loss = criterion(out, labs)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(labs)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs} loss={total_loss/len(train_ds):.4f} ({time.time()-et0:.1f}s)", flush=True)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, lens, labs in test_loader:
            x = x.to(device)
            out = model(x, lens)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            trues.extend(labs.numpy())

    report = classification_report(trues, preds, target_names=classes, output_dict=True)
    macro_f1 = f1_score(trues, preds, average='macro')
    prec, rec, f1c, supp = precision_recall_fscore_support(trues, preds, labels=[list(classes).index('high')])
    print(f"  macro F1={macro_f1:.3f}  high: P={prec[0]:.2f} R={rec[0]:.2f} F1={f1c[0]:.2f}", flush=True)
    return {'name': name, 'macro_f1': macro_f1, 'report': report,
            'high_precision': float(prec[0]), 'high_recall': float(rec[0]), 'high_f1': float(f1c[0]),
            'preds': preds, 'trues': trues}

results = {}
regime = sys.argv[1] if len(sys.argv) > 1 else 'all'

if regime in ('A', 'all'):
    results['A_weighted'] = run_regime('A_weighted (baseline)', train_seqs_base, train_labs_base, use_focal=False)
if regime in ('B', 'all'):
    results['B_focal'] = run_regime('B_focal', train_seqs_base, train_labs_base, use_focal=True)
if regime in ('C', 'all'):
    results['C_oversample'] = run_regime('C_oversample', train_seqs_over, train_labs_over, use_focal=False)
if regime in ('D', 'all'):
    results['D_focal_over'] = run_regime('D_focal_oversample', train_seqs_over, train_labs_over, use_focal=True)

print("\n\n=== SUMMARY (test set, high-cognitive-load class) ===", flush=True)
import os
merged_path = 'models/imbalance_comparison_raw.json'
existing = {}
if os.path.exists(merged_path):
    with open(merged_path) as f:
        existing = json.load(f)
for k, r in results.items():
    existing[k] = {'macro_f1': r['macro_f1'], 'high_precision': r['high_precision'],
                   'high_recall': r['high_recall'], 'high_f1': r['high_f1'],
                   'preds': r['preds'], 'trues': r['trues']}
with open(merged_path, 'w') as f:
    json.dump(existing, f)

for k, r in existing.items():
    print(f"{k:20s} macro_f1={r['macro_f1']:.3f}  high_P={r['high_precision']:.2f}  high_R={r['high_recall']:.2f}  high_F1={r['high_f1']:.2f}")

print(f"\nTotal time this run: {time.time()-t0:.1f}s", flush=True)
print(f"Regimes completed so far: {list(existing.keys())}")
