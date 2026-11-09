"""
CONFIG 2 (reverse direction): labels from clickstream-only clustering
(idle_gap_max/mean/std, active_days, total_clicks, n_distinct_sites,
avg_daily_clicks, precourse_clicks, burstiness -- zero assessment data).
LSTM input: per-assessment sequences (score, error_rate, days_late --
zero clickstream data). Completes the disjoint-feature test in both
directions, per the reviewer's "or vice versa" suggestion.
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
import time, pickle, json

torch.manual_seed(42); np.random.seed(42)
t0 = time.time()

DATA = 'data/raw/'
sa = pd.read_csv(DATA + 'studentAssessment.csv')
asmt = pd.read_csv(DATA + 'assessments.csv')
sa = sa.merge(asmt[['id_assessment','code_module','code_presentation','date']], on='id_assessment', how='left')
sa['days_late'] = sa['date_submitted'] - sa['date']
sa['error_rate'] = 100 - sa['score'].fillna(sa['score'].median())
sa['score'] = sa['score'].fillna(sa['score'].median())
sa['days_late'] = sa['days_late'].fillna(0)

labels = pd.read_csv('data/oulad_labels_clickonly.csv')
key = ['id_student','code_module','code_presentation']
sa = sa.merge(labels, on=key, how='inner')
print(f"Merged: {sa[key].drop_duplicates().shape[0]} student-modules ({time.time()-t0:.1f}s)", flush=True)

FEATS = ['score', 'error_rate', 'days_late']
scaler = StandardScaler()
sa[FEATS] = scaler.fit_transform(sa[FEATS])

sequences, seq_labels = [], []
for _, g in sa.groupby(key):
    g = g.sort_values('date_submitted')
    sequences.append(torch.tensor(g[FEATS].values, dtype=torch.float32))
    seq_labels.append(g['cognitive_load_clickonly_label'].iloc[0])

le = LabelEncoder()
y = le.fit_transform(seq_labels)
classes = le.classes_
print("Classes:", classes, "N:", len(sequences), flush=True)
print("Class distribution:", np.bincount(y), flush=True)

with open('data/config2_cache.pkl', 'wb') as f:
    pickle.dump({'sequences': sequences, 'y': y, 'classes': classes}, f)

idx_train, idx_test = train_test_split(np.arange(len(sequences)), test_size=0.2, random_state=42, stratify=y)

class SeqDataset(Dataset):
    def __init__(self, seqs, labs): self.seqs, self.labs = seqs, labs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labs[i]

def collate(batch):
    seqs, labs = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True), lens, torch.tensor(labs, dtype=torch.long)

class LoadLSTM(nn.Module):
    def __init__(self, input_size=3, hidden=32, n_classes=3):
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

print("\n=== CONFIG 2: assessment input -> clickstream-only labels (disjoint) ===")
print(classification_report(trues, preds, target_names=classes))
f1_macro = f1_score(trues, preds, average='macro')
print(f"Macro F1: {f1_macro:.3f}")

dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(np.zeros((len(idx_train),1)), y[idx_train])
dummy_preds = dummy.predict(np.zeros((len(idx_test),1)))
dummy_f1 = f1_score(y[idx_test], dummy_preds, average='macro')
print(f"Stratified-random baseline macro F1: {dummy_f1:.3f}")

cm = confusion_matrix(trues, preds)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_title('Config 2: Assessment Input -> Clickstream-Only Labels\n(strictly disjoint feature families)')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('figures/confusion_config2_disjoint.png', dpi=150)
plt.close()

with open('models/config2_disjoint_results.json', 'w') as f:
    json.dump({'macro_f1': f1_macro, 'dummy_baseline_f1': dummy_f1,
               'n_sequences': len(sequences), 'classes': classes.tolist()}, f, indent=2)
print(f"\nTotal: {time.time()-t0:.1f}s")
