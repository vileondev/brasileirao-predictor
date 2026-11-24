"""Leakage tests for the Dixon-Coles offset stack."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features, ingestion, stacking


@pytest.fixture(scope="module")
def small() -> pd.DataFrame:
    m = ingestion.build_interim()
    return m[m["season"] <= 2015].reset_index(drop=True)


def test_dc_oos_lambdas_ignore_future(small: pd.DataFrame) -> None:
    base = stacking.dc_oos_lambdas(small, start_season=2014, block=20)
    cut = int(small.loc[small["season"] == 2015, "match_id"].iloc[99])  # 5 blocks into 2015
    future = small.copy()
    late = future["match_id"] > cut
    future.loc[late, "home_goals"] += 5
    future.loc[late, "away_goals"] = 0
    alt = stacking.dc_oos_lambdas(future, start_season=2014, block=20)
    a = base[base["match_id"] <= cut].set_index("match_id")
    b = alt[alt["match_id"] <= cut].set_index("match_id").loc[a.index]
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-9)
    assert len(base) == len(small[small["season"] >= 2014])


def test_offset_model_end_to_end_valid_probabilities() -> None:
    m = ingestion.build_interim()
    m = m[m["season"] <= 2016].reset_index(drop=True)  # needs >= 2 training seasons (2014, 2015)
    feat = stacking.add_dc(features.build_features(m), stacking.dc_oos_lambdas(m))
    preds = stacking.run_offset_backtest(feat, list(stacking.SMALL_FEATURES),
                                         name="t", seasons=[2016])
    p = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    assert len(preds) == 379 and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (preds[["lam_home", "lam_away"]] > 0).all().all()
