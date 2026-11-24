"""Tests for the team-perspective stacked model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import baselines, features, ingestion, stacked, stacking


@pytest.fixture(scope="module")
def small() -> pd.DataFrame:
    m = ingestion.build_interim()
    return m[m["season"] <= 2016].reset_index(drop=True)


@pytest.fixture(scope="module")
def feat(small: pd.DataFrame) -> pd.DataFrame:
    return stacking.add_dc(features.build_features(small), baselines.dc_oos_lambdas(small))


def test_to_long_is_symmetric(feat: pd.DataFrame) -> None:
    long = stacked.to_long(feat)
    n = len(feat)
    assert len(long) == 2 * n and (long["side"].iloc[:n] == "home").all()
    h, a = long.iloc[:n].reset_index(drop=True), long.iloc[n:].reset_index(drop=True)
    assert np.allclose(h["own_gf_r5"], feat["h_gf_r5"], equal_nan=True)
    assert np.allclose(a["opp_gf_r5"], feat["h_gf_r5"], equal_nan=True)
    assert np.allclose(a["elo_diff"], -h["elo_diff"], equal_nan=True)
    assert np.allclose(h["goals"], feat["home_goals"]) and np.allclose(a["goals"], feat["away_goals"])
    assert (h["is_home"] == 1).all() and (a["is_home"] == 0).all()


def test_no_target_or_offset_leaks_into_columns(feat: pd.DataFrame) -> None:
    long = stacked.to_long(feat)
    for with_dc in (False, True):
        cols = stacked.long_columns(long, with_dc)
        assert not {"goals", "dc_lam", "match_id", "season", "side"} & set(cols)
        assert ("dc_log_ratio" in cols) == with_dc


@pytest.mark.parametrize("offset", [False, True])
def test_predictions_ignore_future_results(small: pd.DataFrame, offset: bool) -> None:
    base = stacked.run_stacked(
        stacking.add_dc(features.build_features(small), baselines.dc_oos_lambdas(small)),
        offset, "t", seasons=[2016])
    cut = int(small.loc[small["season"] == 2016, "match_id"].iloc[99])
    future = small.copy()
    late = (future["match_id"] > cut) & (future["season"] == 2016)
    future.loc[late, "home_goals"] += 4
    future.loc[late, "away_goals"] = 0
    alt = stacked.run_stacked(
        stacking.add_dc(features.build_features(future), baselines.dc_oos_lambdas(future)),
        offset, "t", seasons=[2016])
    cols = ["p_home", "p_draw", "p_away", "lam_home", "lam_away"]
    a = base[base["match_id"] <= cut].set_index("match_id")[cols]
    b = alt[alt["match_id"] <= cut].set_index("match_id").loc[a.index, cols]
    assert len(a) == 100 and np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-9)
    assert np.allclose(base[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-6)
