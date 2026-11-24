"""Team-perspective (stacked) goal model: one LightGBM Poisson model, two rows per match.

Each match becomes two rows ``(team, opponent, is_home) -> goals scored by team``.
Parameters are shared between attack of the home and of the away side, which doubles
the training rows. Two variants are compared, both with fixed (untuned) hyper-parameters
so there is no selection bias:

* ``stacked_plain``  - all own/opponent features, no Dixon-Coles input;
* ``stacked_dc``     - same plus the out-of-sample Dixon-Coles lambda as Poisson offset.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import backtest, baselines, config, evaluate, features, ingestion, probability
from src.train import DEFAULT_PARAMS, MAX_TREES, TRAIN_FLOOR_SEASON, _diffs, _make, _summary

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

PARAMS: dict = {**DEFAULT_PARAMS, "min_child_samples": 80}
SIGNED = ("elo_diff", "rest_diff", "h2h_gd", "form_diff_r5", "dc_log_ratio")


def to_long(feat: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match from the team's perspective (home row first, then away row).

    ``feat`` must contain the Dixon-Coles columns ``dc_lh, dc_la, dc_log_ratio``
    (see :func:`src.stacking.add_dc`). Signed features flip sign for the away row.
    """
    parts = []
    for home in (True, False):
        own, opp, sign = ("h_", "a_", 1.0) if home else ("a_", "h_", -1.0)
        d = pd.DataFrame({
            "match_id": feat["match_id"], "season": feat["season"], "side": "home" if home else "away",
            "is_home": 1.0 if home else 0.0,
            "goals": feat["home_goals" if home else "away_goals"],
            "dc_lam": feat["dc_lh" if home else "dc_la"],
            "elo_own": feat["elo_home" if home else "elo_away"],
            "elo_opp": feat["elo_away" if home else "elo_home"],
            "att_edge": feat["home_att_edge" if home else "away_att_edge"],
            "att_edge_ewm": feat["home_att_edge_ewm" if home else "away_att_edge_ewm"],
            "matchday": feat["matchday"],
        })
        for c in features.TEAM_FEATURES:
            d[f"own_{c}"] = feat[own + c]
            d[f"opp_{c}"] = feat[opp + c]
        for c in SIGNED:
            d[c] = sign * feat[c]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def long_columns(long: pd.DataFrame, with_dc: bool) -> list[str]:
    """Model input columns (the Dixon-Coles offset column is never a feature)."""
    skip = {"match_id", "season", "side", "goals", "dc_lam"}
    cols = [c for c in long.columns if c not in skip]
    if not with_dc:
        cols = [c for c in cols if c != "dc_log_ratio"]
    return cols


def _fit(train: pd.DataFrame, cols: Sequence[str], n: int, offset: bool,
         val: pd.DataFrame | None = None) -> lgb.LGBMRegressor:
    kw: dict = {}
    if offset:
        kw["init_score"] = np.log(train["dc_lam"].to_numpy())
    if val is not None:
        kw.update(eval_set=[(val[list(cols)], val["goals"])], eval_metric="poisson",
                  callbacks=[lgb.early_stopping(60, verbose=False)])
        if offset:
            kw["eval_init_score"] = [np.log(val["dc_lam"].to_numpy())]
    return _make(PARAMS, n).fit(train[list(cols)], train["goals"], **kw)


def _predict(m: lgb.LGBMRegressor, df: pd.DataFrame, cols: Sequence[str], offset: bool) -> np.ndarray:
    raw = m.predict(df[list(cols)], raw_score=True)
    return np.exp(raw + (np.log(df["dc_lam"].to_numpy()) if offset else 0.0))


def _split_sides(df: pd.DataFrame, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    home = (df["side"] == "home").to_numpy()
    assert (df.loc[home, "match_id"].to_numpy() == df.loc[~home, "match_id"].to_numpy()).all()
    return lam[home], lam[~home]


def run_stacked(
    feat: pd.DataFrame, offset: bool, name: str, seasons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Walk-forward predictions of the stacked model (same schema as ``backtest``).

    Per season: early stopping on the previous season fixes the tree count and provides
    out-of-fold lambdas for ``rho``; the final model is refit on all earlier seasons.
    """
    long = to_long(feat)
    cols = long_columns(long, with_dc=offset)
    seasons = list(seasons or [s for s in sorted(feat["season"].unique())
                               if s >= config.FIRST_VALIDATION_SEASON])
    out = []
    for s in seasons:
        prior = long[(long["season"] < s) & (long["season"] >= TRAIN_FLOOR_SEASON)]
        test = long[long["season"] == s]
        last = prior["season"].max()
        itr, iva = prior[prior["season"] < last], prior[prior["season"] == last]
        m = _fit(itr, cols, MAX_TREES, offset, iva)
        n = max(int((m.best_iteration_ or 1) * 1.15), 1)
        oh, oa = _split_sides(iva, _predict(m, iva, cols, offset))
        gh, ga = _split_sides(iva, iva["goals"].to_numpy())
        rho = probability.fit_rho(oh, oa, gh, ga)
        lh, la = _split_sides(test, _predict(_fit(prior, cols, n, offset), test, cols, offset))
        out.append(backtest._from_lambdas(name, feat[feat["season"] == s], lh, la, rho))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    from src import stacking, train

    pd.options.display.float_format = "{:.4f}".format
    matches = ingestion.build_interim()
    feat = stacking.add_dc(features.build_features(matches), baselines.dc_oos_lambdas(matches))
    cols = features.feature_columns(feat)
    cols = [c for c in cols if c not in ("dc_lh", "dc_la", "dc_log_ratio")]

    runs = [
        backtest.run_backtest(matches),
        train.run_gbm_backtest(feat, cols, name="gbm_two_models"),
        run_stacked(feat, offset=False, name="stacked_plain"),
        run_stacked(feat, offset=True, name="stacked_dc"),
    ]
    df = backtest.score_table(pd.concat(runs, ignore_index=True), matches)
    full = df[df["season"] <= config.TEST_SEASON]
    _summary(full, f"2016-{config.TEST_SEASON} walk-forward (hiperparametros fixos)")
    _diffs(full, "dixon_coles", ["stacked_plain", "stacked_dc", "gbm_two_models"])
    _diffs(full, "gbm_two_models", ["stacked_plain"])
    _diffs(full, "odds_avg", ["stacked_dc", "dixon_coles"])


if __name__ == "__main__":
    main()
