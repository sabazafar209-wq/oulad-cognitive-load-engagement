import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pickle, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.dummy import DummyClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

t0 = time.time()
with open('data/config1_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
sequences, y, classes = cache['sequences'], cache['y'], cache['classes']
print(f"Loaded cache: {len(sequences)} sequences ({time.time()-t0:.1f}s)", flush=True)

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
    def __init__(self, input_size=3, hidden=48, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, n_classes)
    def forward(self, x, lens):
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(self.dropout(h_n[-1]))

torch.manual_seed(42)
train_ds = SeqDataset([sequences[i] for i in idx_train], [y[i] for i in idx_train])
test_ds = SeqDataset([sequences[i] for i in idx_test], [y[i] for i in idx_test])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

model = LoadLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
counts = np.bincount(y[idx_train]); w = 1.0/counts; w = w/w.sum()*len(counts)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))

EPOCHS = 12
for epoch in range(EPOCHS):
    et0 = time.time()
    model.train(); total_loss = 0
    for x, lens, labs in train_loader:
        opt.zero_grad()
        out = model(x, lens)
        loss = criterion(out, labs)
        loss.backward(); opt.step()
        total_loss += loss.item()*len(labs)
    print(f"epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_ds):.4f} ({time.time()-et0:.1f}s)", flush=True)

model.eval()
preds, trues = [], []
with torch.no_grad():
    for x, lens, labs in test_loader:
        out = model(x, lens)
        preds.extend(out.argmax(dim=1).numpy())
        trues.extend(labs.numpy())

print("\n=== CONFIG 1: clickstream input -> assessment-only labels (disjoint) ===")
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
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_title('Config 1: Clickstream Input -> Assessment-Only Labels\n(strictly disjoint feature families)')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('figures/confusion_config1_disjoint.png', dpi=150)
plt.close()

with open('models/config1_disjoint_results.json', 'w') as f:
    json.dump({'macro_f1': f1_macro, 'dummy_baseline_f1': dummy_f1,
               'n_sequences': len(sequences), 'classes': classes.tolist()}, f, indent=2)
print(f"\nTotal: {time.time()-t0:.1f}s")
