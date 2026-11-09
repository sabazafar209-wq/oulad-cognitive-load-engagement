"""
Phase 8b: Unified multi-seed training driver.

Usage:
  python3 08b_run_seeds.py <label_set> <regime> <seeds_csv> <epochs>

  label_set: 'v1_strict'  -> v1 labels (pure assessment-clustering), paired
                              with clickstream-ONLY sequences. Strictly
                              disjoint feature families -- the harder test.
             'v2_entangled' -> v2 labels (assessment+clickstream clustering),
                              same clickstream-only sequences. The original
                              (softer) evaluation, now repeated across seeds
                              for statistical robustness.
  regime: A_weighted | B_focal | C_oversample | D_focal_over
  seeds_csv: comma-separated ints, e.g. "1,2,3,4,5"
  epochs: int

Each (label_set, regime, seed) result is appended to
models/multiseed_results.json so repeated calls accumulate.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd, numpy as np, torch, pickle, json, os, time, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support

label_set = sys.argv[1]
regime = sys.argv[2]
seeds = [int(s) for s in sys.argv[3].split(',')]
epochs = int(sys.argv[4])

with open('data/lstm_sequences_only.pkl', 'rb') as f:
    seq_cache = pickle.load(f)
all_sequences, all_keys = seq_cache['sequences'], seq_cache['keys']
key_to_idx = {k: i for i, k in enumerate(all_keys)}

if label_set == 'v1_strict':
    labels_df = pd.read_csv('data/oulad_labeled.csv')
elif label_set == 'v2_entangled':
    labels_df = pd.read_csv('data/oulad_labeled_v2_clickstream.csv')
else:
    raise ValueError(label_set)

labels_df = labels_df[['id_student', 'code_module', 'code_presentation', 'cognitive_load_label']]
labels_df['key'] = list(zip(labels_df.id_student, labels_df.code_module, labels_df.code_presentation))
labels_df = labels_df[labels_df['key'].isin(key_to_idx)]

matched_idx = [key_to_idx[k] for k in labels_df['key']]
sequences = [all_sequences[i] for i in matched_idx]
raw_labels = labels_df['cognitive_load_label'].tolist()

le = LabelEncoder()
y_all = le.fit_transform(raw_labels)
classes = le.classes_
print(f"[{label_set}] N={len(sequences)} matched sequences. Classes: {classes}", flush=True)
print(f"Class counts: {np.bincount(y_all)}", flush=True)

use_focal = regime in ('B_focal', 'D_focal_over')
use_over = regime in ('C_oversample', 'D_focal_over')

class SeqDataset(Dataset):
    def __init__(self, seqs, labs): self.seqs, self.labs = seqs, labs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labs[i]

def collate(batch):
    seqs, labs = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True), lens, torch.tensor(labs, dtype=torch.long)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

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

def class_weights_for(labs):
    counts = np.bincount(labs)
    w = 1.0 / counts
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)

def oversample_sequences(seqs, labs, target_class_idx, target_count, jitter_std=0.15, seed=42):
    rng = np.random.RandomState(seed)
    seqs = list(seqs); labs = list(labs)
    minority_idxs = [i for i, l in enumerate(labs) if l == target_class_idx]
    n_needed = max(0, target_count - len(minority_idxs))
    for _ in range(n_needed):
        src = seqs[rng.choice(minority_idxs)].clone()
        seqs.append(src + torch.randn_like(src) * jitter_std)
        labs.append(target_class_idx)
    return seqs, labs

merged_path = 'models/multiseed_results.json'

for seed in seeds:
    t0 = time.time()
    idx_train, idx_test = train_test_split(np.arange(len(sequences)), test_size=0.2,
                                            random_state=seed, stratify=y_all)
    train_seqs = [sequences[i] for i in idx_train]
    train_labs = [int(y_all[i]) for i in idx_train]
    test_seqs = [sequences[i] for i in idx_test]
    test_labs = [int(y_all[i]) for i in idx_test]

    if use_over:
        high_idx = list(classes).index('high')
        medium_count = np.bincount(train_labs)[list(classes).index('medium')]
        train_seqs, train_labs = oversample_sequences(train_seqs, train_labs, high_idx, medium_count, seed=seed)

    torch.manual_seed(seed)
    train_ds = SeqDataset(train_seqs, train_labs)
    test_ds = SeqDataset(test_seqs, test_labs)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

    model = LoadLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    w = class_weights_for(train_labs)
    criterion = FocalLoss(alpha=w, gamma=2.0) if use_focal else nn.CrossEntropyLoss(weight=w)

    for epoch in range(epochs):
        model.train()
        for x, lens, labs in train_loader:
            opt.zero_grad()
            loss = criterion(model(x, lens), labs)
            loss.backward()
            opt.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, lens, labs in test_loader:
            out = model(x, lens)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            trues.extend(labs.numpy().tolist())

    macro_f1 = f1_score(trues, preds, average='macro')
    prec, rec, f1c, _ = precision_recall_fscore_support(trues, preds, labels=[list(classes).index('high')], zero_division=0)

    print(f"[{label_set}/{regime}] seed={seed} macro_f1={macro_f1:.3f} high_P={prec[0]:.2f} high_R={rec[0]:.2f} high_F1={f1c[0]:.2f} ({time.time()-t0:.1f}s)", flush=True)

    existing = {}
    if os.path.exists(merged_path):
        with open(merged_path) as f:
            existing = json.load(f)
    exp_key = f"{label_set}__{regime}"
    existing.setdefault(exp_key, {})
    existing[exp_key][str(seed)] = {
        'macro_f1': macro_f1, 'high_precision': float(prec[0]),
        'high_recall': float(rec[0]), 'high_f1': float(f1c[0]),
    }
    with open(merged_path, 'w') as f:
        json.dump(existing, f, indent=2)

print("Done.", flush=True)
