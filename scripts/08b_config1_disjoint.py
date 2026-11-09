"""
Phase 8b: The strict-disjointness fix requested.

Prior "independent" LSTM evaluations (v1 script 02, v2 script 06) were not
actually independent of the labeling process: the label-generating features
and the LSTM's input features came from overlapping families (both touched
assessment timing/scores, or both touched clickstream/idle-time). A model
recovering cluster structure from correlated raw signals is not evidence it
detects cognitive load independently of how the label was built.

This script fixes that: labels come from ONE feature family, the LSTM sees
ONLY the OTHER, disjoint family, with zero overlap in underlying data source.

CONFIG 1 (this script): labels from assessment-only clustering (v1 labels:
avg_error_rate, pct_late, avg_days_late, max_days_late, num_of_prev_attempts,
std_score -- zero clickstream). LSTM input: weekly VLE clickstream sequences
(week_clicks, week_sites, week_events -- zero assessment data). If the model
still predicts meaningfully above chance, that is real evidence of a
behavioral signal linking interaction patterns to assessment-defined struggle,
not an artifact of shared feature construction.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.dummy import DummyClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time, pickle

torch.manual_seed(42); np.random.seed(42)
t0 = time.time()

# labels: PURE assessment-only clustering (v1) -- zero clickstream involvement
labels = pd.read_csv('data/oulad_labeled.csv')[
    ['id_student','code_module','code_presentation','cognitive_load_label']]

# input: PURE clickstream weekly aggregation -- zero assessment involvement
weekly = pd.read_csv('data/oulad_weekly_clicks.csv')
key = ['id_student', 'code_module', 'code_presentation']
weekly = weekly.merge(labels, on=key, how='inner')
print(f"Merged: {weekly[key].drop_duplicates().shape[0]} student-modules, loaded in {time.time()-t0:.1f}s", flush=True)

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
print("Classes:", classes, "N:", len(sequences), f"({time.time()-t0:.1f}s)", flush=True)
print("Class distribution:", np.bincount(y), flush=True)

idx_train, idx_test = train_test_split(np.arange(len(sequences)), test_size=0.2, random_state=42, stratify=y)

# cache for the multi-seed sweep later
with open('data/config1_cache.pkl', 'wb') as f:
    pickle.dump({'sequences': sequences, 'y': y, 'classes': classes}, f)

class SeqDataset(Dataset):
    def __init__(self, seqs, labs): self.seqs, self.labs = seqs, labs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labs[i]

def collate(batch):
    seqs, labs = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True), lens, torch.tensor(labs, dtype=torch.long)

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

train_ds = SeqDataset([sequences[i] for i in idx_train], [y[i] for i in idx_train])
test_ds = SeqDataset([sequences[i] for i in idx_test], [y[i] for i in idx_test])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

model = LoadLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
counts = np.bincount(y[idx_train]); w = 1.0/counts; w = w/w.sum()*len(counts)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))

EPOCHS = 15
for epoch in range(EPOCHS):
    et0 = time.time()
    model.train(); total_loss = 0
    for x, lens, labs in train_loader:
        opt.zero_grad()
        out = model(x, lens)
        loss = criterion(out, labs)
        loss.backward(); opt.step()
        total_loss += loss.item()*len(labs)
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_ds):.4f} ({time.time()-et0:.1f}s)", flush=True)

model.eval()
preds, trues = [], []
with torch.no_grad():
    for x, lens, labs in test_loader:
        out = model(x, lens)
        preds.extend(out.argmax(dim=1).numpy())
        trues.extend(labs.numpy())

print("\n=== CONFIG 1: clickstream input -> assessment-only labels (strictly disjoint) ===")
print(classification_report(trues, preds, target_names=classes))
f1_macro = f1_score(trues, preds, average='macro')
print(f"Macro F1: {f1_macro:.3f}")

# random-chance baseline for honest comparison (stratified dummy classifier)
dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(np.zeros((len(idx_train),1)), y[idx_train])
dummy_preds = dummy.predict(np.zeros((len(idx_test),1)))
dummy_f1 = f1_score(y[idx_test], dummy_preds, average='macro')
print(f"Stratified-random baseline macro F1: {dummy_f1:.3f}")

cm = confusion_matrix(trues, preds)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_title('Config 1: Clickstream Input -> Assessment-Only Labels\n(strictly disjoint feature families)')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('figures/confusion_config1_disjoint.png', dpi=150)
plt.close()

import json
with open('models/config1_disjoint_results.json', 'w') as f:
    json.dump({'macro_f1': f1_macro, 'dummy_baseline_f1': dummy_f1,
               'n_sequences': len(sequences), 'classes': classes.tolist()}, f, indent=2)

print(f"\nDone in {time.time()-t0:.1f}s")
