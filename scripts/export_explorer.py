"""Export everything the Model Explorer web app needs into web/explorer_data.js."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import train as T

df = pd.read_csv(ROOT / "data" / "telecom_churn.csv")
tr, va, te = T.temporal_split(df)
feats = T.NUM + T.CAT
pipe = Pipeline([("prep", T.preprocessor()), ("model", T.models()["Baseline: LogisticRegression"])])
pipe.fit(tr[feats], tr[T.TARGET])

sc = pipe.named_steps["prep"].named_transformers_["num"].named_steps["sc"]
oh = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
m = pipe.named_steps["model"]

p = pipe.predict_proba(te[feats])[:, 1]
y = te[T.TARGET].values

# Permutation importance on the test period: how much ROC-AUC drops when a feature is shuffled.
pi = permutation_importance(pipe, te[feats], y, scoring="roc_auc", n_repeats=20, random_state=T.SEED)
labels = {"tenure_months": "Months as a customer", "monthly_charge": "Monthly charge",
          "support_tickets_30d": "Support tickets (30 days)", "usage_minutes": "Usage minutes",
          "late_payment": "Late payment", "plan": "Plan"}
importance = sorted(({"feature": labels[f], "drop": round(float(d), 4)}
                     for f, d in zip(feats, pi.importances_mean)), key=lambda r: -r["drop"])

# Calibration: 5 bins by predicted probability (quantiles, so each bin has enough customers).
edges = np.quantile(p, np.linspace(0, 1, 6))
calib = []
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (p >= lo) & (p <= hi)
    calib.append({"predicted": round(float(p[mask].mean()), 4), "observed": round(float(y[mask].mean()), 4),
                  "n": int(mask.sum())})

lb = pd.read_csv(ROOT / "results" / "leaderboard.csv")
out = {
    "model": {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "plans": oh.categories_[0].tolist(),
              "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])},
    "test": {"p": [round(float(v), 5) for v in p], "y": y.astype(int).tolist()},
    "importance": importance,
    "calibration": calib,
    "leaderboard": lb.round(4).to_dict(orient="records"),
}
# Inline the data into web/explorer.html so the page is one self-contained file.
import re
html_path = ROOT / "web" / "explorer.html"
html = html_path.read_text(encoding="utf-8")
block = "<!-- DATA:START --><script>window.EXPLORER = " + json.dumps(out) + ";</script><!-- DATA:END -->"
html = re.sub(r"<!-- DATA:START -->.*?<!-- DATA:END -->", lambda _: block, html, flags=re.S)
html_path.write_text(html, encoding="utf-8")
print("test rows", len(y), "churners", int(y.sum()))
print("importance", importance)
print("calibration", calib)
