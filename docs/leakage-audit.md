# How I Audited My Own Result for Feature Leakage

This note explains, step by step, how I found that part of my first result came from feature leakage, and how I tested it. Full details are in the [preprint](https://doi.org/10.35542/osf.io/bgnkw_v1), Sections 4.5 and 4.6.

## 1. The result that looked good

My first LSTM predicted cognitive-load tiers with 0.79 accuracy and 0.650 macro F1 (one train/test split). After I added the full clickstream (v2), recall on the high-load class rose from 0.21 to 0.82.

## 2. Why I became suspicious

OULAD has no direct measure of cognitive load, so I created labels by clustering struggle-related features. I then noticed that the model inputs and those labels came from related data:

- **v1:** the LSTM saw per-assessment score, error rate and days late. The labels were per-student averages of those same quantities. The model only had to learn an average.
- **v2:** the labels mixed assessment features with clickstream features (idle gaps, burstiness, pre-deadline clicks). The LSTM's weekly click sequences came from the same clickstream table.

A model can partly reconstruct labels like these without learning anything about cognitive load. So these numbers are upper bounds, not proof of independent predictive power.

## 3. The strict test

I rebuilt the evaluation so that no feature family appears on both sides:

| Side | Built from | Excluded |
|---|---|---|
| Labels | assessment data only (error rate, lateness, prior attempts, score variance) | all clickstream data |
| Model input | weekly clickstream sequences only (clicks, distinct sites, events) | all assessment data |

I ran both the entangled and the strict setup with 5 random seeds each, varying the train/test split and model initialization together (`scripts/08*.py`).

## 4. What happened

| Evaluation (5 seeds) | Macro F1 | High-load F1 |
|---|---|---|
| Entangled (v2 labels) | 0.536 ± 0.022 | 0.390 ± 0.023 |
| Strict feature-disjoint | 0.483 ± 0.004 | 0.244 ± 0.010 |

Every entangled seed scored higher than every strict seed (Mann-Whitney U = 25, one-tailed p = 0.004, the smallest p-value possible with 5 runs per group). Part of the original result was leakage. A real but modest signal remains above a single-class baseline.

## 5. A second check: a claimed improvement that did not replicate

On one split, oversampling the high-load class seemed to improve macro F1 (0.545 to 0.559). Across 5 seeds it did not (0.536 vs. 0.539, Wilcoxon p = 0.625), so I retracted that claim. Focal loss underperformed the baseline in all 5 seeds (0.429 vs. 0.536, p = 0.0625, the smallest possible at n = 5), which is a strong lead but not conventionally significant.

## 6. What I learned

- Leakage can inflate results even when labels and inputs are not literally the same columns.
- A single train/test split can make a small difference look like a win.
- I now treat **0.483 macro F1** as the honest headline, and I report the first numbers alongside it so readers can see the correction.

Results files: `models/statistical_summary.json`, `models/multiseed_results.json` and `models/effect_sizes.json`. Figure: `figures/multiseed_boxplots.png`.
