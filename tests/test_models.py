"""Leakage, probability-matrix and metric tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest, baselines, evaluate, ingestion, probability


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return ingestion.build_interim()


# ---- probability layer ---------------------------------------------------- #

@pytest.mark.parametrize("rho", [0.0, -0.1, -0.03])
def test_matrix_sums_to_one(rho: float) -> None:
    rng = np.random.default_rng(0)
    lh, la = rng.uniform(0.2, 3.5, 200), rng.uniform(0.2, 3.5, 200)
    mat = probability.scoreline_matrix(lh, la, rho)
    assert np.allclose(mat.sum(axis=(1, 2)), 1.0, atol=1e-6)
    assert np.allclose(probability.outcome_probs(mat).sum(axis=1), 1.0, atol=1e-6)
    assert (mat >= 0).all()


def test_dixon_coles_lifts_low_draws() -> None:
    ind = probability.scoreline_matrix(1.3, 1.0, 0.0)[0]
    dc = probability.scoreline_matrix(1.3, 1.0, -0.1)[0]
    assert dc[0, 0] > ind[0, 0] and dc[1, 1] > ind[1, 1]
    assert dc[1, 0] < ind[1, 0]


def test_markets_in_range() -> None:
    mat = probability.scoreline_matrix([1.5], [1.1], -0.05)
    mk = probability.market_probs(mat)
    assert 0 < mk["over_2_5"][0] < 1 and 0 < mk["btts"][0] < 1
    top = probability.top_scorelines(mat[0])
    assert len(top) == 3 and top[0][2] >= top[1][2] >= top[2][2]


# ---- metrics --------------------------------------------------------------- #

def test_rps_known_values() -> None:
    perfect = evaluate.rps_per_match(np.array([[1.0, 0, 0]]), np.array([0]))
    assert perfect[0] == 0.0
    # predicting home with certainty when away wins -> maximal RPS = 1
    worst = evaluate.rps_per_match(np.array([[1.0, 0, 0]]), np.array([2]))
    assert worst[0] == pytest.approx(1.0)
    uniform = evaluate.rps_per_match(np.full((1, 3), 1 / 3), np.array([0]))
    assert uniform[0] == pytest.approx(0.5 * ((2 / 3) ** 2 + (1 / 3) ** 2))


def test_odds_probs_normalized(matches: pd.DataFrame) -> None:
    p = baselines.odds_probs(matches, "AvgC")
    assert np.allclose(p.sum(axis=1), 1.0)


# ---- leakage ---------------------------------------------------------------- #

def test_elo_truncation_invariance(matches: pd.DataFrame) -> None:
    full = baselines.compute_elo(matches)
    for t in (50, 400, 2500, 5000):
        trunc = baselines.compute_elo(matches.iloc[: t + 1])
        assert np.allclose(trunc.iloc[t].to_numpy(), full.iloc[t].to_numpy())


def test_elo_unaffected_by_own_outcome(matches: pd.DataFrame) -> None:
    base = baselines.compute_elo(matches)
    t = 1234
    mutated = matches.copy()
    mutated.loc[t, "home_goals"] += 5
    after = baselines.compute_elo(mutated)
    assert np.allclose(after.iloc[: t + 1].to_numpy(), base.iloc[: t + 1].to_numpy())


def test_backtest_predictions_ignore_future(matches: pd.DataFrame) -> None:
    small = matches[matches["season"] <= 2017].reset_index(drop=True)
    s17 = small[small["season"] == 2017]
    cutoff = s17["match_id"].iloc[99]  # first 100 matches of 2017 = 5 blocks of 20
    base = backtest.run_backtest(small, first_val_season=2017, block=20)

    future = small.copy()
    mask = future["match_id"] > cutoff
    future.loc[mask & (future["season"] == 2017), "home_goals"] += 4
    future.loc[mask & (future["season"] == 2017), "away_goals"] = 0
    alt = backtest.run_backtest(future, first_val_season=2017, block=20)

    keep = base["match_id"] <= cutoff
    for model in ("elo_poisson", "dixon_coles"):
        a = base[keep & (base["model"] == model)].set_index("match_id")
        b = alt[(alt["match_id"] <= cutoff) & (alt["model"] == model)].set_index("match_id")
        assert len(a) == 100 or model == "dixon_coles" and len(a) >= 100
        cols = ["p_home", "p_draw", "p_away"]
        assert np.allclose(a[cols].to_numpy(), b.loc[a.index, cols].to_numpy(), atol=1e-9), model


# ---- negative binomial ------------------------------------------------------ #

def test_negbin_matrix_valid_and_reduces_to_poisson() -> None:
    lh, la = np.array([1.2, 2.5]), np.array([0.8, 1.1])
    nb = probability.scoreline_matrix(lh, la, -0.05, alpha=0.3)
    assert np.allclose(nb.sum(axis=(1, 2)), 1.0, atol=1e-6) and (nb >= 0).all()
    assert np.allclose(probability.scoreline_matrix(lh, la, -0.05, alpha=1e-12),
                       probability.scoreline_matrix(lh, la, -0.05), atol=1e-9)
    # overdispersion fattens both tails relative to Poisson
    pois = probability.scoreline_matrix(lh, la, 0.0)
    wide = probability.scoreline_matrix(lh, la, 0.0, alpha=0.5)
    assert wide[0, 0, 0] > pois[0, 0, 0]


def test_fit_alpha_recovers_dispersion_and_poisson() -> None:
    rng = np.random.default_rng(1)
    lam = rng.uniform(0.8, 2.2, 20000)
    true_alpha = 0.25
    r = 1 / true_alpha
    y = rng.negative_binomial(r, r / (r + lam))
    est = probability.fit_alpha(lam[:10000], lam[10000:], y[:10000], y[10000:])
    assert est == pytest.approx(true_alpha, abs=0.04)
    yp = rng.poisson(lam)
    assert probability.fit_alpha(lam[:10000], lam[10000:], yp[:10000], yp[10000:]) < 0.03
