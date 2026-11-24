"""GBM residual correction on top of Dixon-Coles (DC lambdas as a Poisson offset).

``log(lam) = log(lam_dc) + f(x)``: with zero trees the model *is* Dixon-Coles, so
the GBM only has to learn corrections from the extra features. Out-of-sample DC
lambdas are produced for every row with strictly earlier data (same refit-block
scheme as the backtest), so train-time and test-time offsets are equally causal.
"""
from __future__ import annotations

import argparse
import json
import warnings
from typing import Sequence

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src import backtest, baselines, config, evaluate, features, ingestion, probability
from src.train import MAX_TREES, TRAIN_FLOOR_SEASON, _diffs, _make, _summary

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

SMALL_FEATURES: tuple[str, ...] = (
    "dc_log_ratio", "elo_diff", "rest_diff", "h_rest_days", "a_rest_days", "matchday",
    "h2h_gd", "form_diff_r5", "h_pts_r5", "a_pts_r5",
    "home_att_edge_ewm", "away_att_edge_ewm", "h_promoted", "a_promoted",
)

OFFSET_PARAMS: dict = {
    "learning_rate": 0.02, "num_leaves": 4, "max_depth": 2, "min_child_samples": 60,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 20.0,
}


dc_oos_lambdas = baselines.dc_oos_lambdas  # moved to baselines so predict.py need not import lightgbm


def add_dc(feat: pd.DataFrame, dc: pd.DataFrame) -> pd.DataFrame:
    """Merge DC lambdas into the feature table (rows before the start stay NaN)."""
    out = feat.merge(dc, on="match_id", how="left")
    out["dc_log_ratio"] = np.log(out["dc_lh"] / out["dc_la"])
    return out


def _fit(train: pd.DataFrame, cols: Sequence[str], target: str, off: str, params: dict,
         n: int, val: pd.DataFrame | None = None) -> lgb.LGBMRegressor:
    kw: dict = {"init_score": np.log(train[off].to_numpy())}
    if val is not None:
        kw.update(eval_set=[(val[list(cols)], val[target])],
                  eval_init_score=[np.log(val[off].to_numpy())], eval_metric="poisson",
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    return _make(params, n).fit(train[list(cols)], train[target], **kw)


def _predict(m: lgb.LGBMRegressor, df: pd.DataFrame, cols: Sequence[str], off: str) -> np.ndarray:
    return np.exp(m.predict(df[list(cols)], raw_score=True) + np.log(df[off].to_numpy()))


def run_offset_backtest(
    feat: pd.DataFrame, cols: Sequence[str], params: dict | None = None,
    name: str = "gbm_dc_offset", seasons: Sequence[int] | None = None,
    n_trees: int | None = None,
) -> pd.DataFrame:
    """Walk-forward predictions of the DC-offset GBM (same schema as ``backtest``)."""
    params = params or OFFSET_PARAMS
    seasons = list(seasons or [s for s in sorted(feat["season"].unique())
                               if s >= config.FIRST_VALIDATION_SEASON])
    out = []
    for s in seasons:
        prior = feat[(feat["season"] < s) & (feat["season"] >= TRAIN_FLOOR_SEASON)]
        test = feat[feat["season"] == s]
        last = prior["season"].max()
        itr, iva = prior[prior["season"] < last], prior[prior["season"] == last]
        oof, trees = {}, {}
        for tgt, off in (("home_goals", "dc_lh"), ("away_goals", "dc_la")):
            m = _fit(itr, cols, tgt, off, params, MAX_TREES, iva)
            trees[tgt] = max(int(m.best_iteration_ or 1), 1)
            oof[tgt] = _predict(m, iva, cols, off)
        rho = probability.fit_rho(oof["home_goals"], oof["away_goals"],
                                  iva["home_goals"].to_numpy(), iva["away_goals"].to_numpy())
        lam = {}
        for tgt, off in (("home_goals", "dc_lh"), ("away_goals", "dc_la")):
            n = n_trees if n_trees is not None else max(int(trees[tgt] * 1.15), 1)
            lam[tgt] = _predict(_fit(prior, cols, tgt, off, params, n), test, cols, off)
        out.append(backtest._from_lambdas(name, test, lam["home_goals"], lam["away_goals"], rho))
    return pd.concat(out, ignore_index=True)


def tune(feat: pd.DataFrame, cols: Sequence[str],
         fold_seasons: Sequence[int] = (2018, 2019, 2020, 2021, 2022), n_trials: int = 30) -> dict:
    """Optuna on walk-forward folds (validation Poisson deviance); later seasons untouched."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        p = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 10),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
        }
        n = trial.suggest_int("n_estimators", 5, 250)
        devs = []
        for s in fold_seasons:
            tr = feat[(feat["season"] < s) & (feat["season"] >= TRAIN_FLOOR_SEASON)]
            va = feat[feat["season"] == s]
            d = 0.0
            for tgt, off in (("home_goals", "dc_lh"), ("away_goals", "dc_la")):
                d += evaluate.poisson_deviance(va[tgt], _predict(_fit(tr, cols, tgt, off, p, n), va, cols, off))
            devs.append(d)
        return float(np.mean(devs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    n = best.pop("n_estimators")
    best["subsample_freq"] = 1
    return {"params": best, "n_estimators": n, "value": study.best_value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    pd.options.display.float_format = "{:.4f}".format

    matches = ingestion.build_interim()
    feat = add_dc(features.build_features(matches), dc_oos_lambdas(matches))
    all_cols = features.feature_columns(feat)
    all_cols = [c for c in all_cols if c not in ("dc_lh", "dc_la")]
    small = list(SMALL_FEATURES)

    # reference deviance of the raw DC lambdas on the same rows
    base_preds = backtest.run_backtest(matches)
    runs = [
        base_preds,
        run_offset_backtest(feat, small, name="gbm_dc_small"),
        run_offset_backtest(feat, all_cols, name="gbm_dc_all"),
    ]
    print("Otimizando (folds 2018-2022; 2023-2025 intocados)...")
    best = tune(feat, small, n_trials=args.trials)
    print("melhores:", json.dumps(best, indent=1))
    holdout = [2023, 2024, 2025]
    runs.append(run_offset_backtest(feat, small, best["params"], name="gbm_dc_tuned",
                                    seasons=holdout, n_trees=best["n_estimators"]))
    df = backtest.score_table(pd.concat(runs, ignore_index=True), matches)

    full = df[(df["season"] <= config.TEST_SEASON) & (df["model"] != "gbm_dc_tuned")]
    _summary(full, f"2016-{config.TEST_SEASON} walk-forward")
    print(df[df["model"].isin(["odds_avg", "dixon_coles", "gbm_dc_small", "gbm_dc_all"])]
          .pivot_table(index="season", columns="model", values="rps").to_string())
    _diffs(full, "dixon_coles", ["gbm_dc_small", "gbm_dc_all"])
    _diffs(full, "odds_avg", ["gbm_dc_small", "dixon_coles"])
    hold = df[df["season"].isin(holdout)]
    _summary(hold, "Hold-out 2023-2025 (tuning so em 2018-2022)")
    _diffs(hold, "dixon_coles", ["gbm_dc_small", "gbm_dc_all", "gbm_dc_tuned"])
    _diffs(hold, "odds_avg", ["gbm_dc_tuned", "dixon_coles"])


if __name__ == "__main__":
    main()
