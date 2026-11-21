# AGENTS.md — project rules for AI coding agents

1. Never use random train_test_split on time-indexed data. Use explicit cutoff dates.
2. All imputers/scalers/encoders must live inside a sklearn Pipeline and be fit on Train only.
3. Always include a simple baseline (regularized LogisticRegression) and report it.
4. Report ROC-AUC, PR-AUC, Brier score and top-decile lift, not accuracy alone.
5. Set seed=42 everywhere; two runs must give identical scores.
6. Show a Plan before writing code. Run `pytest` and fix failures before declaring done.
