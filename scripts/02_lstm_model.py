"""
Phase 5: LSTM sequence model.
Input: per-student sequence of assessment-level events (score, error_rate, days_late,
       normalized position in course) IN ORDER.
Output: cognitive_load class (low/medium/high), predicted from the raw trace rather
        than the pre-aggregated features -> a genuinely harder, non-circular test.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv('data/oulad_sequences_raw.csv')
FEATS = ['score', 'error_rate', 'days_late']

# fill missing scores with median (banked/missing submissions)
df['score'] = df['score'].fillna(df['score'].median())
df['error_rate'] = df['error_rate'].fillna(df['error_rate'].median())
df['days_late'] = df['days_late'].fillna(0)

scaler = StandardScaler()
df[FEATS] = scaler.fit_transform(df[FEATS])

groups = df.groupby(['id_student', 'code_module', 'code_presentation'])
sequences, labels = [], []
for _, g in groups:
    g = g.sort_values('date_submitted')
    sequences.append(torch.tensor(g[FEATS].values, dtype=torch.float32))
    labels.append(g['cognitive_load_label'].iloc[0])

le = LabelEncoder()
y = le.fit_transform(labels)
print('Classes:', le.classes_, 'N sequences:', len(sequences))

idx_train, idx_test = train_test_split(np.arange(len(sequences)), test_size=0.2,
                                        random_state=42, stratify=y)

class SeqDataset(Dataset):
    def __init__(self, seqs, labels, idxs):
        self.seqs = [seqs[i] for i in idxs]
        self.labels = [labels[i] for i in idxs]
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labels[i]

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
    def __init__(self, input_size=3, hidden=32, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, n_classes)
    def forward(self, x, lens):
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        out = self.dropout(h_n[-1])
        return self.fc(out)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LoadLSTM().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# class weights for imbalance
class_counts = np.bincount(y[idx_train])
weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
weights = weights / weights.sum() * len(class_counts)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

EPOCHS = 25
train_losses = []
for epoch in range(EPOCHS):
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
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}")

# eval
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for x, lens, labs in test_loader:
        x = x.to(device)
        out = model(x, lens)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labs.numpy())

print("\n=== LSTM — Cognitive Load (predicted from raw sequence) ===")
print(classification_report(all_true, all_preds, target_names=le.classes_))
f1_macro = f1_score(all_true, all_preds, average='macro')
print(f"Macro F1: {f1_macro:.3f}")

cm = confusion_matrix(all_true, all_preds)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Cognitive Load — LSTM (sequence model)')
plt.tight_layout()
plt.savefig('figures/confusion_lstm_cognitive_load.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(train_losses, color='#8172B2')
ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss')
ax.set_title('LSTM Training Loss')
plt.tight_layout()
plt.savefig('figures/lstm_training_loss.png', dpi=150)
plt.close()

torch.save({'model_state': model.state_dict(), 'label_encoder_classes': le.classes_,
            'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_},
           'models/lstm_cognitive_load.pt')

import json
with open('models/lstm_results.json', 'w') as f:
    json.dump({'macro_f1': f1_macro, 'n_sequences': len(sequences),
               'classes': le.classes_.tolist()}, f, indent=2)

print("LSTM model + figures saved.")
