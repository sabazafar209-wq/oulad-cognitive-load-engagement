"""
Phase 4: Baseline models (Random Forest) for
  (a) Cognitive Load classification (low/medium/high)
  (b) Engagement classification (low/medium/high)
Establishes benchmark performance before the LSTM sequence model.
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

sns.set_style('whitegrid')
df = pd.read_csv('data/oulad_labeled.csv')

FEATURES = ['n_assessments', 'avg_score', 'avg_error_rate', 'std_score',
            'pct_late', 'avg_days_late', 'max_days_late', 'total_banked',
            'date_registration', 'withdrew', 'num_of_prev_attempts', 'studied_credits']

X = df[FEATURES].fillna(0)

results = {}
for target_col, target_name in [('cognitive_load_label', 'cognitive_load'),
                                  ('engagement_label', 'engagement')]:
    y = df[target_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)  # alphabetical: high=0, low=1, medium=2 -- keep track

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    clf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"\n=== {target_name.upper()} — Random Forest ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"Macro F1: {f1_macro:.3f}")

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{target_name.replace("_"," ").title()} — Confusion Matrix (RF baseline)')
    plt.tight_layout()
    plt.savefig(f'figures/confusion_{target_name}.png', dpi=150)
    plt.close()

    # feature importance plot
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6,5))
    imp.plot(kind='barh', ax=ax, color='#4C72B0')
    ax.set_title(f'Feature Importance — {target_name.replace("_"," ").title()} (RF)')
    plt.tight_layout()
    plt.savefig(f'figures/importance_{target_name}.png', dpi=150)
    plt.close()

    joblib.dump({'model': clf, 'label_encoder': le, 'features': FEATURES},
                f'models/rf_{target_name}.joblib')
    results[target_name] = {'macro_f1': f1_macro, 'report': report}

import json
with open('models/baseline_results.json', 'w') as f:
    json.dump({k: {'macro_f1': v['macro_f1']} for k,v in results.items()}, f, indent=2)

print("\nBaseline models + figures + reports saved.")
