"""Poisson vs Negative Binomial goal marginals on top of Dixon-Coles lambdas.

Uses out-of-sample Dixon-Coles lambdas (production settings). For every season
``s >= 2016`` the dispersion ``alpha`` (``Var = lam + alpha*lam^2``) and ``rho``
are fitted ONLY on lambdas/results of earlier seasons, then both models are
scored on season ``s``. ``alpha`` is a single scalar, so selection risk is tiny.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, evaluate, ingestion, probability, stacking


def score_season(cur: pd.DataFrame, alpha: float, rho: float) -> pd.DataFrame:
    """Per-match metrics for one season under a given dispersion and rho."""
    lh, la = cur["dc_lh"].to_numpy(), cur["dc_la"].to_numpy()
    mat = probability.scoreline_matrix(lh, la, rho, alpha=alpha)
    p = probability.outcome_probs(mat)
    y = evaluate.outcome_index(cur["home_goals"].to_numpy(), cur["away_goals"].to_numpy())
    hg = np.minimum(cur["home_goals"].to_numpy(), config.MAX_GOALS)
    ag = np.minimum(cur["away_goals"].to_numpy(), config.MAX_GOALS)
    over = probability.market_probs(mat)["over_2_5"]
    hit = ((cur["home_goals"] + cur["away_goals"]) >= 3).to_numpy().astype(float)
    return pd.DataFrame({
        "match_id": cur["match_id"].to_numpy(), "season": cur["season"].to_numpy(),
        "rps": evaluate.rps_per_match(p, y), "logloss": evaluate.log_loss_per_match(p, y),
        "score_ll": np.log(mat[np.arange(len(cur)), hg, ag]),
        "over_brier": (over - hit) ** 2, "p_over": over, "over_hit": hit,
    })


def main() -> None:
    pd.options.display.float_format = "{:.5f}".format
    matches = ingestion.build_interim()
    lam = stacking.dc_oos_lambdas(matches)
    df = lam.merge(matches[["match_id", "season", "home_goals", "away_goals"]], on="match_id")

    # raw overdispersion diagnostic on all out-of-sample predictions
    y = np.r_[df["home_goals"], df["away_goals"]].astype(float)
    mu = np.r_[df["dc_lh"], df["dc_la"]]
    print(f"Diagnostico (n={len(y)} gols de time): media prevista {mu.mean():.3f} | "
          f"media observada {y.mean():.3f}")
    print(f"  E[(y-mu)^2]/E[mu] = {np.mean((y - mu) ** 2) / mu.mean():.3f}  (Poisson perfeito = 1.000)")
    print(f"  alpha de Pearson ~ {np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2):.4f}")

    pois, nb, fitted = [], [], []
    for s in range(2016, config.TEST_SEASON + 1):
        prior, cur = df[df["season"] < s], df[df["season"] == s]
        args = (prior["dc_lh"].to_numpy(), prior["dc_la"].to_numpy(),
                prior["home_goals"].to_numpy(), prior["away_goals"].to_numpy())
        alpha = probability.fit_alpha(*args)
        rho_p = probability.fit_rho(*args)
        rho_n = probability.fit_rho(*args, alpha=alpha)
        fitted.append((s, alpha, rho_p, rho_n))
        pois.append(score_season(cur, 0.0, rho_p))
        nb.append(score_season(cur, alpha, rho_n))
    pois, nb = pd.concat(pois, ignore_index=True), pd.concat(nb, ignore_index=True)

    print("\nalpha e rho ajustados so com temporadas anteriores:")
    print(pd.DataFrame(fitted, columns=["season", "alpha", "rho_poisson", "rho_negbin"]).to_string(index=False))

    print(f"\n== Walk-forward 2016-{config.TEST_SEASON} (n={len(pois)}) ==")
    cols = ["rps", "logloss", "score_ll", "over_brier"]
    print(pd.DataFrame({"poisson": pois[cols].mean(), "negbin": nb[cols].mean()}).T.to_string())
    print("\nDiferenca NegBin - Poisson (IC 95%, bootstrap pareado):")
    for c in cols:
        better = "negativo = NegBin melhor" if c != "score_ll" else "positivo = NegBin melhor"
        m, lo, hi = evaluate.paired_bootstrap_diff(nb[c].to_numpy(), pois[c].to_numpy())
        print(f"  {c:11s} {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]   ({better})")

    print("\nCalibracao do Over 2.5 (previsto >= 55%):")
    for name, r in (("poisson", pois), ("negbin", nb)):
        hi = r[r["p_over"] >= 0.55]
        print(f"  {name:8s} n={len(hi):4d} previsto={hi['p_over'].mean():.3f} observado={hi['over_hit'].mean():.3f}")


if __name__ == "__main__":
    main()
