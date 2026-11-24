"""Walk-forward backtest shared by all models.

Every prediction for a match uses only matches with an earlier kickoff than the
first match of its refit block. The latest completed season is the test season;
the in-progress season is reported separately as forward evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import baselines, config, evaluate, ingestion, probability

MODELS = ("odds_avg", "odds_pinnacle", "climatology", "elo_poisson", "dixon_coles")


def _rows(model: str, df: pd.DataFrame, probs: np.ndarray,
          lh: np.ndarray | None = None, la: np.ndarray | None = None,
          score_ll: np.ndarray | None = None) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame({
        "match_id": df["match_id"].to_numpy(),
        "season": df["season"].to_numpy(),
        "model": model,
        "p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2],
        "lam_home": np.full(n, np.nan) if lh is None else lh,
        "lam_away": np.full(n, np.nan) if la is None else la,
        "score_ll": np.full(n, np.nan) if score_ll is None else score_ll,
    })


def _from_lambdas(model: str, df: pd.DataFrame, lh: np.ndarray, la: np.ndarray,
                  rho: float) -> pd.DataFrame:
    mat = probability.scoreline_matrix(lh, la, rho)
    hg = np.minimum(df["home_goals"].to_numpy(), config.MAX_GOALS)
    ag = np.minimum(df["away_goals"].to_numpy(), config.MAX_GOALS)
    ll = np.log(mat[np.arange(len(df)), hg, ag])
    return _rows(model, df, probability.outcome_probs(mat), lh, la, ll)


def run_backtest(
    matches: pd.DataFrame,
    first_val_season: int = config.FIRST_VALIDATION_SEASON,
    block: int = 20,
    half_life_days: float = 730.0,
) -> pd.DataFrame:
    """Run all baselines walk-forward and return a long table of predictions.

    Parameters
    ----------
    matches : pd.DataFrame
        Clean matches from :func:`src.ingestion.build_interim`.
    first_val_season : int
        First season to predict; earlier seasons are training only.
    block : int
        Dixon-Coles is refit every ``block`` matches (about two rounds).
    half_life_days : float
        Time-decay half-life of the Dixon-Coles weights.
    """
    elo = baselines.compute_elo(matches)
    m = matches.join(elo)
    out: list[pd.DataFrame] = []
    train_floor = max(config.WARMUP_SEASONS)

    for season in sorted(m["season"].unique()):
        if season < first_val_season:
            continue
        sdf = m[m["season"] == season]
        season_start = sdf["kickoff"].iloc[0]
        prior = m[m["kickoff"] < season_start]

        # bookmaker benchmarks
        out.append(_rows("odds_avg", sdf, baselines.odds_probs(sdf, "AvgC")))
        pin = sdf.dropna(subset=["PSCH"])
        out.append(_rows("odds_pinnacle", pin, baselines.odds_probs(pin, "PSC")))

        # climatology
        freq = np.bincount(
            evaluate.outcome_index(prior["home_goals"].to_numpy(), prior["away_goals"].to_numpy()),
            minlength=3) / len(prior)
        out.append(_rows("climatology", sdf, np.tile(freq, (len(sdf), 1))))

        # Elo -> Poisson (params and rho fitted on prior seasons only)
        tr = prior[prior["season"] > train_floor]
        ep = baselines.EloPoisson().fit(tr["elo_diff"].to_numpy(),
                                        tr["home_goals"].to_numpy(), tr["away_goals"].to_numpy())
        lh_tr, la_tr = ep.predict(tr["elo_diff"].to_numpy())
        rho_ep = probability.fit_rho(lh_tr, la_tr, tr["home_goals"].to_numpy(), tr["away_goals"].to_numpy())
        lh, la = ep.predict(sdf["elo_diff"].to_numpy())
        out.append(_from_lambdas("elo_poisson", sdf, lh, la, rho_ep))

        # Dixon-Coles, refit every `block` matches on strictly earlier kickoffs
        rho_dc: float | None = None
        for start in range(0, len(sdf), block):
            chunk = sdf.iloc[start:start + block]
            ref = chunk["kickoff"].iloc[0]
            train = m[m["kickoff"] < ref]
            dc = baselines.DixonColes(half_life_days).fit(train, ref)
            if rho_dc is None:
                lh_t, la_t = dc.predict(train)
                rho_dc = probability.fit_rho(lh_t, la_t, train["home_goals"].to_numpy(),
                                             train["away_goals"].to_numpy())
            lh, la = dc.predict(chunk)
            out.append(_from_lambdas("dixon_coles", chunk, lh, la, rho_dc))

    return pd.concat(out, ignore_index=True)


def score_table(preds: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-season metrics (RPS, log-loss, Brier, scoreline log-lik)."""
    df = preds.merge(matches[["match_id", "home_goals", "away_goals"]], on="match_id")
    y = evaluate.outcome_index(df["home_goals"].to_numpy(), df["away_goals"].to_numpy())
    p = df[["p_home", "p_draw", "p_away"]].to_numpy()
    df["rps"] = evaluate.rps_per_match(p, y)
    df["logloss"] = evaluate.log_loss_per_match(p, y)
    df["brier"] = evaluate.brier_per_match(p, y)
    return df


def main() -> None:
    matches = ingestion.build_interim()
    preds = run_backtest(matches)
    df = score_table(preds, matches)
    test = config.TEST_SEASON
    complete = df[df["season"] <= test]

    pd.options.display.float_format = "{:.4f}".format
    print("\n== RPS por temporada (menor = melhor) ==")
    print(df.pivot_table(index="season", columns="model", values="rps").to_string())
    print(f"\n== Resumo 2016-{test} (walk-forward) ==")
    print(complete.groupby("model")[["rps", "logloss", "brier", "score_ll"]].mean().sort_values("rps").to_string())
    print(f"\n== Teste {test} ==")
    print(df[df["season"] == test].groupby("model")[["rps", "logloss", "brier", "score_ll"]].mean().sort_values("rps").to_string())

    # paired bootstrap vs. odds on the same matches (all validation seasons)
    base = complete[complete["model"] == "odds_avg"].set_index("match_id")["rps"]
    print("\n== Diferenca de RPS vs odds_avg (IC 95%, bootstrap pareado) ==")
    for model in ("elo_poisson", "dixon_coles", "climatology"):
        cur = complete[complete["model"] == model].set_index("match_id")["rps"]
        mean, lo, hi = evaluate.paired_bootstrap_diff(cur.loc[base.index].to_numpy(), base.to_numpy())
        print(f"{model:12s} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
