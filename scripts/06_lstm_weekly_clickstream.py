"""
Phase 5b: LSTM v2 -- trained on true weekly VLE click sequences (not sparse
assessment events). Each student-module becomes a sequence of ~(course_length/7)
weekly snapshots: [total_clicks, n_distinct_sites, n_active_days] per week.
Predicts cognitive_load_label, independent of the pre-aggregated cluster features.
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
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

import time
t_start = time.time()
labels = pd.read_csv('data/oulad_labeled_v2_clickstream.csv')[
    ['id_student','code_module','code_presentation','cognitive_load_label']]

key = ['id_student', 'code_module', 'code_presentation']

# use the pre-cached weekly aggregation (built by 06a_cache_weekly.py) instead of
# re-reading the 10.6M-row raw file every run
weekly = pd.read_csv('data/oulad_weekly_clicks.csv')
print(f"Loaded cached weekly data: {weekly.shape} in {time.time()-t_start:.1f}s", flush=True)

weekly = weekly.merge(labels, on=key, how='inner')
weekly = weekly.sort_values(key + ['week'])

FEATS = ['week_clicks', 'week_sites', 'week_events']
scaler = StandardScaler()
weekly[FEATS] = scaler.fit_transform(weekly[FEATS])

t2 = time.time()
groups = weekly.groupby(key)
sequences, seq_labels = [], []
for _, g in groups:
    g = g.sort_values('week')
    sequences.append(torch.tensor(g[FEATS].values, dtype=torch.float32))
    seq_labels.append(g['cognitive_load_label'].iloc[0])
print(f"Built {len(sequences)} sequences in {time.time()-t2:.1f}s", flush=True)
lens_all = [len(s) for s in sequences]
print(f"Sequence length -- min {min(lens_all)}, median {int(np.median(lens_all))}, max {max(lens_all)}")

le = LabelEncoder()
y = le.fit_transform(seq_labels)
print("Classes:", le.classes_)

idx_train, idx_test = train_test_split(np.arange(len(sequences)), test_size=0.2, random_state=42, stratify=y)

class SeqDataset(Dataset):
    def __init__(self, seqs, labs, idxs):
        self.seqs = [seqs[i] for i in idxs]
        self.labs = [labs[i] for i in idxs]
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labs[i]

def collate(batch):
    seqs, labs = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True)
    labs = torch.tensor(labs, dtype=torch.long)
    return padded, lens, labs

train_ds = SeqDataset(sequences, y, idx_train)
test_ds = SeqDataset(sequences, y, idx_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LoadLSTM().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

class_counts = np.bincount(y[idx_train])
weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
weights = weights / weights.sum() * len(class_counts)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

EPOCHS = 15
train_losses = []
for epoch in range(EPOCHS):
    epoch_t0 = time.time()
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
    avg_loss = total_loss / len(train_ds)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}  ({time.time()-epoch_t0:.1f}s)", flush=True)

model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for x, lens, labs in test_loader:
        x = x.to(device)
        out = model(x, lens)
        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_true.extend(labs.numpy())

print("\n=== LSTM v2 (weekly clickstream) — Cognitive Load ===")
print(classification_report(all_true, all_preds, target_names=le.classes_))
f1_macro = f1_score(all_true, all_preds, average='macro')
print(f"Macro F1: {f1_macro:.3f}")

cm = confusion_matrix(all_true, all_preds)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Cognitive Load — LSTM v2 (weekly clickstream)')
plt.tight_layout()
plt.savefig('figures/confusion_lstm_v2_weekly.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(train_losses, color='#55A868')
ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss')
ax.set_title('LSTM v2 Training Loss (weekly clickstream)')
plt.tight_layout()
plt.savefig('figures/lstm_v2_training_loss.png', dpi=150)
plt.close()

torch.save({'model_state': model.state_dict(), 'label_encoder_classes': le.classes_},
           'models/lstm_v2_weekly.pt')

import json
with open('models/lstm_v2_results.json', 'w') as f:
    json.dump({'macro_f1': f1_macro, 'n_sequences': len(sequences),
               'classes': le.classes_.tolist(),
               'seq_len_min': int(min(lens_all)), 'seq_len_median': int(np.median(lens_all)),
               'seq_len_max': int(max(lens_all))}, f, indent=2)

print("LSTM v2 saved.")
