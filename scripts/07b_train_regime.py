import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pickle, json, os, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

regime = sys.argv[1]          # 'A_weighted' | 'B_focal' | 'C_oversample' | 'D_focal_over'
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15

with open('data/lstm_imbalance_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
classes = cache['classes']

use_focal = regime in ('B_focal', 'D_focal_over')
use_over = regime in ('C_oversample', 'D_focal_over')
if use_over:
    train_seqs, train_labs = cache['train_seqs_over'], cache['train_labs_over']
else:
    train_seqs, train_labs = cache['train_seqs_base'], cache['train_labs_base']
test_seqs, test_labs = cache['test_seqs'], cache['test_labs']

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

torch.manual_seed(42)
train_ds = SeqDataset(train_seqs, train_labs)
test_ds = SeqDataset(test_seqs, test_labs)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

model = LoadLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
w = class_weights_for(train_labs)
criterion = FocalLoss(alpha=w, gamma=2.0) if use_focal else nn.CrossEntropyLoss(weight=w)

print(f"Regime {regime}: focal={use_focal} oversample={use_over} epochs={epochs} n_train={len(train_seqs)}", flush=True)
t0 = time.time()
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, lens, labs in train_loader:
        opt.zero_grad()
        out = model(x, lens)
        loss = criterion(out, labs)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(labs)
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"  epoch {epoch+1}/{epochs} loss={total_loss/len(train_ds):.4f} ({time.time()-t0:.1f}s elapsed)", flush=True)

model.eval()
preds, trues = [], []
with torch.no_grad():
    for x, lens, labs in test_loader:
        out = model(x, lens)
        preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
        trues.extend(labs.numpy().tolist())

macro_f1 = f1_score(trues, preds, average='macro')
prec, rec, f1c, _ = precision_recall_fscore_support(trues, preds, labels=[list(classes).index('high')])
print(f"RESULT {regime}: macro_f1={macro_f1:.3f} high_P={prec[0]:.2f} high_R={rec[0]:.2f} high_F1={f1c[0]:.2f}", flush=True)

merged_path = 'models/imbalance_comparison_raw.json'
existing = {}
if os.path.exists(merged_path):
    with open(merged_path) as f:
        existing = json.load(f)
existing[regime] = {'macro_f1': macro_f1, 'high_precision': float(prec[0]),
                     'high_recall': float(rec[0]), 'high_f1': float(f1c[0]),
                     'preds': preds, 'trues': trues}
with open(merged_path, 'w') as f:
    json.dump(existing, f)
print(f"Saved. Total time {time.time()-t0:.1f}s. Regimes done: {list(existing.keys())}")
