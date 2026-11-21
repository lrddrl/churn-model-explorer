# Leaderboard

Split: train <= 2025-06 (4500 rows), val 2025-07..09 (750), test >= 2025-10 (750). Seed=42.

| Model                        | Split   |   ROC-AUC |   PR-AUC |   Brier |   Top-decile lift |
|:-----------------------------|:--------|----------:|---------:|--------:|------------------:|
| Baseline: LogisticRegression | Val     |    0.7474 |   0.1876 |  0.043  |            4.1667 |
| Baseline: LogisticRegression | Test    |    0.7351 |   0.2775 |  0.0551 |            3.3333 |
| HistGradientBoosting         | Val     |    0.6293 |   0.0716 |  0.0483 |            1.1111 |
| HistGradientBoosting         | Test    |    0.686  |   0.222  |  0.057  |            3.3333 |
| RandomForest                 | Val     |    0.7141 |   0.1813 |  0.0432 |            4.1667 |
| RandomForest                 | Test    |    0.6956 |   0.2403 |  0.0566 |            3.3333 |
