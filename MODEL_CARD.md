# Model Card: 90-day churn predictor

## What it does
Predicts the chance that a telecom customer cancels in the next 90 days, so a retention team can decide who gets an offer.

## Model
- Logistic regression (scikit-learn), standardized numeric inputs and one-hot plan.
- Chosen because it beat gradient boosting and random forest on the test period, and it is easy to explain.

## Data
- Synthetic data: 250 customers × 24 months = 6,000 rows, generated with seed 42.
- Inputs: months as a customer, plan, monthly charge, support tickets (30 days), usage minutes, late payment. Customer ID is never used.
- Split by date: train to Jun 2025, validate Jul–Sep 2025, test Oct–Dec 2025 (750 rows, 48 churners, 5.7% churn rate overall).

## Test results (Oct–Dec 2025)
| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Logistic regression (baseline, chosen) | 0.735 | 0.278 | 0.055 |
| Hist. gradient boosting | 0.686 | 0.222 | 0.057 |
| Random forest | 0.696 | 0.240 | 0.057 |

## Most important inputs (permutation importance, drop in test ROC-AUC)
Usage minutes (0.093), support tickets (0.050), late payment (0.011), plan (0.008). Monthly charge and months as a customer add almost nothing.

## Checks
7 automated tests pass: no date overlap, no lost rows, no ID or target leakage, preprocessing fit on train only, repeatable runs, best model ≥ baseline, no suspiciously strong feature.

## Limitations
- The data is synthetic, so the predictions say nothing about real customers.
- The test set is small (48 churners), so differences between models may be noise.
- The model slightly under-predicts churn: in most calibration groups the real churn rate is a little higher than predicted.
- ROI numbers in the app depend on the assumptions typed in; they are estimates, not results.

## Intended use
Classroom demo of the spec, plan, verify method. Not for real customer decisions.
