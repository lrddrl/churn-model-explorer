"""Generate a synthetic telecom churn dataset (customer-month rows, 2024-01 .. 2025-12)."""
import numpy as np
import pandas as pd

SEED = 42


def make_data(n_customers: int = 250, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.period_range("2024-01", "2025-12", freq="M")
    rows = []
    for cid in range(n_customers):
        base_tenure = rng.integers(1, 60)
        plan = rng.choice(["basic", "plus", "premium"], p=[0.5, 0.3, 0.2])
        monthly = {"basic": 35, "plus": 60, "premium": 95}[plan] + rng.normal(0, 5)
        risk = rng.normal(0, 1)  # hidden customer propensity
        for i, m in enumerate(months):
            tickets = rng.poisson(max(0.2, 0.8 + 0.6 * risk))
            usage = max(0, rng.normal(300 - 40 * risk, 60))
            late = rng.binomial(1, min(0.9, max(0.02, 0.1 + 0.08 * risk)))
            logit = (-3.0 + 0.5 * risk + 0.25 * tickets + 0.8 * late
                     - 0.004 * (usage - 300) - 0.01 * (base_tenure + i))
            p = 1 / (1 + np.exp(-logit))
            rows.append({
                "customer_id": cid,
                "snapshot_date": m.to_timestamp(),
                "tenure_months": base_tenure + i,
                "plan": plan,
                "monthly_charge": round(monthly, 2),
                "support_tickets_30d": tickets,
                "usage_minutes": round(usage, 1),
                "late_payment": late,
                "churn_next_90d": rng.binomial(1, p),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_data()
    df.to_csv("data/telecom_churn.csv", index=False)
    print(df.shape, f"positive rate={df.churn_next_90d.mean():.3f}")
