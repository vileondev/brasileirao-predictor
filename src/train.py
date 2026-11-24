"""LightGBM Poisson goal models: walk-forward training, Optuna tuning, comparison.

Per predicted season ``s`` the models are trained only on seasons ``< s``:
  1. an inner split (train ``< s-1``, validate ``s-1``) picks the number of trees
     by early stopping and provides out-of-fold lambdas to fit Dixon-Coles ``rho``;
  2. the final models are refit on all seasons ``< s`` with that tree count.
"""
from __future__ import annotations

import argparse
import json
from typing import Sequence

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src import backtest, config, evaluate, features, ingestion, probability

TRAIN_FLOOR_SEASON: int = max(config.WARMUP_SEASONS) + 1  # first season used for training rows

DEFAULT_PARAMS: dict = {
    "learning_rate": 0.03,
    "num_leaves": 6,
    "max_depth": 3,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_lambda": 5.0,
}
MAX_TREES: int = 800


def _make(params: dict, n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="poisson", n_estimators=n_estimators, random_state=config.SEED,
        verbosity=-1, n_jobs=1, **params,
    )


def _fit_pair(
    train: pd.DataFrame, cols: Sequence[str], params: dict, n_home: int, n_away: int
) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor]:
    mh = _make(params, n_home).fit(train[list(cols)], train["home_goals"])
    ma = _make(params, n_away).fit(train[list(cols)], train["away_goals"])
    return mh, ma


def _best_trees(
    train: pd.DataFrame, val: pd.DataFrame, cols: Sequence[str], params: dict
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Early-stopped tree counts and out-of-fold lambdas on ``val``."""
    res = []
    for target in ("home_goals", "away_goals"):
        m = _make(params, MAX_TREES).fit(
            train[list(cols)], train[target],
            eval_set=[(val[list(cols)], val[target])], eval_metric="poisson",
            callbacks=[lgb.early_stopping(60, verbose=False)],
        )
        res.append((max(int(m.best_iteration_ or 50), 20), m.predict(val[list(cols)])))
    return res[0][0], res[1][0], res[0][1], res[1][1]


def run_gbm_backtest(
    feat: pd.DataFrame,
    cols: Sequence[str],
    params: dict | None = None,
    name: str = "gbm_poisson",
    seasons: Sequence[int] | None = None,
    n_trees: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Walk-forward predictions of the GBM (same long schema as ``backtest``).

    Parameters
    ----------
    feat : pd.DataFrame
        Output of :func:`src.features.build_features`.
    cols : sequence of str
        Model input columns (their order is the saved feature schema).
    params : dict, optional
        LightGBM hyper-parameters (defaults to :data:`DEFAULT_PARAMS`).
    seasons : sequence of int, optional
        Seasons to predict (default: every season from the first validation season).
    n_trees : tuple of int, optional
        Fixed ``(home, away)`` tree counts; when ``None`` early stopping on the
        previous season decides.
    """
    params = params or DEFAULT_PARAMS
    seasons = list(seasons or [s for s in sorted(feat["season"].unique())
                               if s >= config.FIRST_VALIDATION_SEASON])
    out = []
    for s in seasons:
        prior = feat[(feat["season"] < s) & (feat["season"] >= TRAIN_FLOOR_SEASON)]
        test = feat[feat["season"] == s]
        last = prior["season"].max()
        inner_train, inner_val = prior[prior["season"] < last], prior[prior["season"] == last]
        nh, na, oof_h, oof_a = _best_trees(inner_train, inner_val, cols, params)
        rho = probability.fit_rho(oof_h, oof_a, inner_val["home_goals"].to_numpy(),
                                  inner_val["away_goals"].to_numpy())
        if n_trees is not None:
            nh, na = n_trees
        mh, ma = _fit_pair(prior, cols, params, int(nh * 1.15), int(na * 1.15))
        lh, la = mh.predict(test[list(cols)]), ma.predict(test[list(cols)])
        out.append(backtest._from_lambdas(name, test, lh, la, rho))
    return pd.concat(out, ignore_index=True)


def tune(
    feat: pd.DataFrame, cols: Sequence[str],
    fold_seasons: Sequence[int] = (2018, 2019, 2020, 2021, 2022), n_trials: int = 40,
) -> dict:
    """Optuna search minimizing mean validation Poisson deviance on walk-forward folds.

    Only ``fold_seasons`` are ever scored, so seasons after them stay untouched
    for an honest hold-out.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        p = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 16),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        }
        n = trial.suggest_int("n_estimators", 40, 500)
        devs = []
        for s in fold_seasons:
            tr = feat[(feat["season"] < s) & (feat["season"] >= TRAIN_FLOOR_SEASON)]
            va = feat[feat["season"] == s]
            mh, ma = _fit_pair(tr, cols, p, n, n)
            devs.append(
                evaluate.poisson_deviance(va["home_goals"], mh.predict(va[list(cols)]))
                + evaluate.poisson_deviance(va["away_goals"], ma.predict(va[list(cols)]))
            )
        return float(np.mean(devs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    n_estimators = best.pop("n_estimators")
    best["subsample_freq"] = 1
    return {"params": best, "n_estimators": n_estimators, "value": study.best_value}


def _summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n== {title} ==")
    print(df.groupby("model")[["rps", "logloss", "brier", "score_ll"]].mean().sort_values("rps").to_string())


def _diffs(df: pd.DataFrame, ref: str, models: Sequence[str]) -> None:
    base = df[df["model"] == ref].set_index("match_id")["rps"]
    print(f"-- diferenca de RPS vs {ref} (IC 95%) --")
    for m in models:
        cur = df[df["model"] == m].set_index("match_id")["rps"]
        idx = base.index.intersection(cur.index)
        mean, lo, hi = evaluate.paired_bootstrap_diff(cur.loc[idx].to_numpy(), base.loc[idx].to_numpy())
        print(f"{m:16s} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  (n={len(idx)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()
    pd.options.display.float_format = "{:.4f}".format

    matches = ingestion.build_interim()
    feat = features.build_features(matches)
    cols = features.feature_columns(feat)
    elo_cols = ["elo_home", "elo_away", "elo_diff"]

    base_preds = backtest.run_backtest(matches)
    gbm = run_gbm_backtest(feat, cols, name="gbm_default")
    gbm_elo = run_gbm_backtest(feat, elo_cols, name="gbm_elo_only")
    gbm_odds = run_gbm_backtest(feat, features.feature_columns(feat, include_odds=True),
                                name="EXP_gbm_with_odds")

    print("Otimizando hiperparametros (folds 2018-2022; 2023-2025 intocados)...")
    best = tune(feat, cols, n_trials=args.trials)
    print("melhores:", json.dumps(best, indent=1))
    holdout = [2023, 2024, 2025]
    tuned = run_gbm_backtest(feat, cols, best["params"], name="gbm_tuned", seasons=holdout,
                             n_trees=(best["n_estimators"], best["n_estimators"]))

    allp = pd.concat([base_preds, gbm, gbm_elo, gbm_odds, tuned], ignore_index=True)
    df = backtest.score_table(allp, matches)

    full = df[(df["season"] <= config.TEST_SEASON) & (df["model"] != "gbm_tuned")]
    _summary(full, f"2016-{config.TEST_SEASON} walk-forward (params padrao)")
    print(df[df["model"].isin(["odds_avg", "dixon_coles", "gbm_default", "gbm_elo_only"])]
          .pivot_table(index="season", columns="model", values="rps").to_string())
    _diffs(full, "dixon_coles", ["gbm_default", "gbm_elo_only", "elo_poisson"])
    _diffs(full, "odds_avg", ["gbm_default", "dixon_coles"])

    hold = df[df["season"].isin(holdout)]
    _summary(hold, "Hold-out 2023-2025 (hiperparametros tunados so em 2018-2022)")
    _diffs(hold, "dixon_coles", ["gbm_default", "gbm_tuned"])
    _diffs(hold, "odds_avg", ["gbm_default", "gbm_tuned", "dixon_coles"])

    # final artifact: model trained on every complete season, with saved schema
    final_train = feat[(feat["season"] >= TRAIN_FLOOR_SEASON) & (feat["season"] <= config.TEST_SEASON)]
    mh, ma = _fit_pair(final_train, cols, best["params"], best["n_estimators"], best["n_estimators"])
    config.MODELS_DIR.mkdir(exist_ok=True)
    mh.booster_.save_model(str(config.MODELS_DIR / "gbm_home.txt"))
    ma.booster_.save_model(str(config.MODELS_DIR / "gbm_away.txt"))
    (config.MODELS_DIR / "schema.json").write_text(json.dumps({"features": cols, **best}, indent=1))
    imp = pd.Series(mh.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("\nTop features (gain, modelo mandante):")
    print((imp / imp.sum()).head(10).to_string())


if __name__ == "__main__":
    main()
