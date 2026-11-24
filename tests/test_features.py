"""Feature leakage tests (truncation invariance + own-outcome invariance)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features, ingestion


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return ingestion.build_interim()


@pytest.fixture(scope="module")
def feats(matches: pd.DataFrame) -> pd.DataFrame:
    return features.build_features(matches)


def _row(df: pd.DataFrame, match_id: int, cols: list[str]) -> np.ndarray:
    return df.loc[df["match_id"] == match_id, cols].to_numpy(dtype=float)[0]


def test_one_row_per_match_and_no_target_in_features(matches: pd.DataFrame, feats: pd.DataFrame) -> None:
    assert len(feats) == len(matches)
    cols = features.feature_columns(feats)
    assert not {"home_goals", "away_goals", "result"} & set(cols)
    assert not set(features.ODDS_FEATURES) & set(cols)


@pytest.mark.parametrize("t", [30, 600, 2000, 4300, 5600])
def test_truncation_invariance(matches: pd.DataFrame, feats: pd.DataFrame, t: int) -> None:
    cols = features.feature_columns(feats)
    trunc = features.build_features(matches.iloc[: t + 1])
    mid = int(matches["match_id"].iloc[t])
    assert np.allclose(_row(trunc, mid, cols), _row(feats, mid, cols), equal_nan=True)


@pytest.mark.parametrize("t", [800, 3000, 5000])
def test_own_outcome_does_not_change_own_features(matches: pd.DataFrame, feats: pd.DataFrame, t: int) -> None:
    cols = features.feature_columns(feats)
    mutated = matches.copy()
    mutated.loc[t, "home_goals"] += 6
    mutated.loc[t, "away_goals"] = 0
    alt = features.build_features(mutated)
    mid = int(matches["match_id"].iloc[t])
    assert np.allclose(_row(alt, mid, cols), _row(feats, mid, cols), equal_nan=True)
    # ...and nothing before it changes either
    before = feats["match_id"] < mid
    assert np.allclose(alt.loc[before, cols].to_numpy(dtype=float),
                       feats.loc[before, cols].to_numpy(dtype=float), equal_nan=True)


def test_early_rows_are_nan_not_zero(feats: pd.DataFrame) -> None:
    first = feats.iloc[0]
    assert np.isnan(first["h_gf_r5"]) and first["h_games_played"] == 0
