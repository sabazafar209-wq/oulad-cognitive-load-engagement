"""
Phase 6: Explainability layer (SHAP).
Explains the Random Forest cognitive-load classifier so that predictions are
interpretable for educators -- not just accurate. This is the project's
main differentiator vs. most reviewed literature (few behavioral-only CL
papers include feature-level interpretability).
"""
import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('data/oulad_labeled.csv')
bundle = joblib.load('models/rf_cognitive_load.joblib')
clf, le, FEATURES = bundle['model'], bundle['label_encoder'], bundle['features']

X = df[FEATURES].fillna(0)
X_sample = X.sample(n=1500, random_state=42)  # subsample for speed

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sample)

# shap_values shape: (n_samples, n_features, n_classes) in newer shap versions
print("SHAP values shape:", np.array(shap_values).shape)

class_names = le.classes_
# summary plot per class
if isinstance(shap_values, list):
    sv_high = shap_values[list(class_names).index('high')]
else:
    sv_high = shap_values[:, :, list(class_names).index('high')]

fig = plt.figure(figsize=(8,6))
shap.summary_plot(sv_high, X_sample, feature_names=FEATURES, show=False)
plt.title('SHAP Summary — Predicting HIGH Cognitive Load')
plt.tight_layout()
plt.savefig('figures/shap_summary_high_load.png', dpi=150, bbox_inches='tight')
plt.close()

# bar plot of mean |SHAP| across all classes (global importance)
if isinstance(shap_values, list):
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    mean_abs = np.abs(shap_values).mean(axis=(0,2))

imp_df = pd.Series(mean_abs, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7,5))
imp_df.plot(kind='barh', ax=ax, color='#C44E52')
ax.set_title('Global SHAP Feature Importance — Cognitive Load Model')
ax.set_xlabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig('figures/shap_global_importance.png', dpi=150)
plt.close()

print("Top features driving cognitive load predictions:")
print(imp_df.sort_values(ascending=False).head(6))
print("\nSHAP explainability figures saved.")
