"""Deterministic verification gates (the 'Verify' half of vibe-then-verify)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train as T  # noqa: E402
from make_data import make_data  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return make_data()


def test_no_date_overlap(df):
    tr, va, te = T.temporal_split(df)
    assert tr[T.DATE].max() < va[T.DATE].min()
    assert va[T.DATE].max() < te[T.DATE].min()


def test_splits_cover_all_rows_without_overlap(df):
    tr, va, te = T.temporal_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert not (set(tr.index) & set(va.index)) and not (set(va.index) & set(te.index))


def test_customer_id_not_a_feature():
    assert "customer_id" not in T.NUM + T.CAT
    assert T.TARGET not in T.NUM + T.CAT


def test_preprocessing_fit_on_train_only(df):
    tr, _, _ = T.temporal_split(df)
    prep = T.preprocessor().fit(tr[T.NUM + T.CAT])
    scaler = prep.named_transformers_["num"].named_steps["sc"]
    assert np.allclose(scaler.mean_, tr[T.NUM].mean().values, atol=1e-6)


def test_determinism(df):
    a = T.run(df)
    b = T.run(df)
    pd.testing.assert_frame_equal(a, b)


def test_champion_beats_or_matches_baseline_on_val(df):
    res = T.run(df)
    val = res[res.Split == "Val"].set_index("Model")["ROC-AUC"]
    assert val.max() >= val["Baseline: LogisticRegression"]


def test_no_single_feature_looks_leaky(df):
    from sklearn.metrics import roc_auc_score
    for c in T.NUM:
        auc = roc_auc_score(df[T.TARGET], df[c])
        assert max(auc, 1 - auc) < 0.95, f"{c} univariate AUC suspiciously high"
