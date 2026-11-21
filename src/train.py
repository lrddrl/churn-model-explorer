"""Spec-first churn pipeline.

DATA CONTRACT : data/telecom_churn.csv, unit = customer_month, time = snapshot_date,
                target = churn_next_90d (binary)
SPLIT POLICY  : strict temporal. Train <= 2025-06, Val = 2025-07..09, Test >= 2025-10
LEAKAGE POLICY: ALL imputers / scalers / encoders live inside a sklearn Pipeline
                and are fit on Train only. customer_id is never a feature.
BENCHMARK     : Baseline LogisticRegression vs HistGradientBoosting, RandomForest
METRICS       : ROC-AUC, PR-AUC, Brier, top-decile lift
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42
TARGET = "churn_next_90d"
DATE = "snapshot_date"
NUM = ["tenure_months", "monthly_charge", "support_tickets_30d", "usage_minutes", "late_payment"]
CAT = ["plan"]
TRAIN_END = pd.Timestamp("2025-06-30")
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2025-10-01")


def temporal_split(df: pd.DataFrame):
    d = pd.to_datetime(df[DATE])
    return (df[d <= TRAIN_END].copy(),
            df[(d >= VAL_START) & (d <= VAL_END)].copy(),
            df[d >= TEST_START].copy())


def preprocessor() -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


def models():
    return {
        "Baseline: LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                               random_state=SEED, n_jobs=1),
    }


def top_decile_lift(y, p) -> float:
    n = max(1, len(y) // 10)
    idx = np.argsort(-p)[:n]
    return float(np.mean(np.asarray(y)[idx]) / np.mean(y))


def evaluate(y, p) -> dict:
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "Top-decile lift": top_decile_lift(y, p)}


def run(df: pd.DataFrame) -> pd.DataFrame:
    train, val, test = temporal_split(df)
    feats = NUM + CAT
    rows = []
    for name, est in models().items():
        pipe = Pipeline([("prep", preprocessor()), ("model", est)])
        pipe.fit(train[feats], train[TARGET])
        for split_name, part in (("Val", val), ("Test", test)):
            p = pipe.predict_proba(part[feats])[:, 1]
            rows.append({"Model": name, "Split": split_name, **evaluate(part[TARGET], p)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("data/telecom_churn.csv")
    tr, va, te = temporal_split(df)
    print(f"rows train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    res = run(df)
    res.to_csv("results/leaderboard.csv", index=False)
    with open("results/leaderboard.md", "w") as f:
        f.write("# Leaderboard\n\n")
        f.write(f"Split: train <= 2025-06 ({len(tr)} rows), val 2025-07..09 ({len(va)}), "
                f"test >= 2025-10 ({len(te)}). Seed={SEED}.\n\n")
        f.write(res.round(4).to_markdown(index=False))
        f.write("\n")
    print(res.round(4).to_string(index=False))
