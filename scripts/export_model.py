"""Export the trained baseline model's numbers as JSON (used by web/index.html)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from sklearn.pipeline import Pipeline
import train as T

df = pd.read_csv("data/telecom_churn.csv")
tr, va, te = T.temporal_split(df)
pipe = Pipeline([("prep", T.preprocessor()), ("model", T.models()["Baseline: LogisticRegression"])])
pipe.fit(tr[T.NUM + T.CAT], tr[T.TARGET])
sc = pipe.named_steps["prep"].named_transformers_["num"].named_steps["sc"]
oh = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
m = pipe.named_steps["model"]
p = pipe.predict_proba(te[T.NUM + T.CAT])[:, 1]
y = te[T.TARGET].values
rows = []
for t in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
    f = p >= t
    rows.append({"t": t, "flagged": int(f.sum()), "caught": int((f & (y == 1)).sum()),
                 "missed": int((~f & (y == 1)).sum()), "falseAlarms": int((f & (y == 0)).sum())})
out = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "plans": oh.categories_[0].tolist(),
       "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
       "thresholds": rows, "testRows": len(y), "testChurners": int(y.sum())}
Path("results/model.json").write_text(json.dumps(out, indent=1))
print("wrote results/model.json")
