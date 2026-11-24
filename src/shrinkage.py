"""Does shrinking the Dixon-Coles lambdas help?

Two shrinkage mechanisms are compared in a causal walk-forward setting:

* **Ridge** on attack/defence parameters (shrinks team strengths to the mean).
* **Log-lambda shrinkage** ``log lam' = m + alpha * (log lam - m)`` towards the mean
  log-lambda of earlier seasons (shrinks extreme matchups, ``alpha < 1``).

Configurations are *selected* on validation seasons 2016-2022 and judged on the
untouched hold-out 2023-2025. ``rho`` for season ``s`` is fitted only on
out-of-sample lambdas of earlier seasons.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src import config, evaluate, ingestion, probability, stacking

HALF_LIVES: tuple[float, ...] = (365.0, 730.0, 1460.0)
RIDGES: tuple[float, ...] = (0.5, 2.0, 5.0, 10.0, 20.0)
ALPHAS: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7)
VALIDATION = range(2016, 2023)
HOLDOUT = (2023, 2024, 2025)
BASELINE = (730.0, 0.5, 1.0)  # (half_life, ridge, alpha) used so far


def _shrink(lh: np.ndarray, la: np.ndarray, mh: float, ma: float, alpha: float
            ) -> tuple[np.ndarray, np.ndarray]:
    return (np.exp(mh + alpha * (np.log(lh) - mh)), np.exp(ma + alpha * (np.log(la) - ma)))


def evaluate_config(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per-match metrics for one set of OOS lambdas (columns dc_lh, dc_la) and ``alpha``."""
    out = []
    for s in range(2016, config.TEST_SEASON + 1):
        prior, cur = df[df["season"] < s], df[df["season"] == s]
        mh, ma = np.log(prior["dc_lh"]).mean(), np.log(prior["dc_la"]).mean()
        plh, pla = _shrink(prior["dc_lh"].to_numpy(), prior["dc_la"].to_numpy(), mh, ma, alpha)
        rho = probability.fit_rho(plh, pla, prior["home_goals"].to_numpy(), prior["away_goals"].to_numpy())
        lh, la = _shrink(cur["dc_lh"].to_numpy(), cur["dc_la"].to_numpy(), mh, ma, alpha)
        mat = probability.scoreline_matrix(lh, la, rho)
        p = probability.outcome_probs(mat)
        y = evaluate.outcome_index(cur["home_goals"].to_numpy(), cur["away_goals"].to_numpy())
        hg = np.minimum(cur["home_goals"].to_numpy(), config.MAX_GOALS)
        ag = np.minimum(cur["away_goals"].to_numpy(), config.MAX_GOALS)
        over = probability.market_probs(mat)["over_2_5"]
        hit = ((cur["home_goals"] + cur["away_goals"]) >= 3).to_numpy().astype(float)
        out.append(pd.DataFrame({
            "match_id": cur["match_id"].to_numpy(), "season": s,
            "rps": evaluate.rps_per_match(p, y), "logloss": evaluate.log_loss_per_match(p, y),
            "score_ll": np.log(mat[np.arange(len(cur)), hg, ag]),
            "over_brier": (over - hit) ** 2, "p_home": p[:, 0], "p_over": over,
            "home_win": (y == 0).astype(float), "over_hit": hit,
        }))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    pd.options.display.float_format = "{:.4f}".format
    matches = ingestion.build_interim()
    results: dict[tuple, pd.DataFrame] = {}
    for hl, ridge in itertools.product(HALF_LIVES, RIDGES):
        lam = stacking.dc_oos_lambdas(matches, half_life_days=hl, ridge=ridge)
        df = lam.merge(matches[["match_id", "season", "home_goals", "away_goals"]], on="match_id")
        for alpha in ALPHAS:
            results[(hl, ridge, alpha)] = evaluate_config(df, alpha)
        print(f"half-life {hl:.0f}d ridge {ridge}: done", flush=True)

    rows = []
    for cfg, r in results.items():
        v, h = r[r["season"].isin(list(VALIDATION))], r[r["season"].isin(HOLDOUT)]
        rows.append({"half_life": cfg[0], "ridge": cfg[1], "alpha": cfg[2],
                     "val_rps": v["rps"].mean(), "val_over_brier": v["over_brier"].mean(),
                     "hold_rps": h["rps"].mean(), "hold_logloss": h["logloss"].mean(),
                     "hold_over_brier": h["over_brier"].mean()})
    tab = pd.DataFrame(rows).sort_values("val_rps")
    print("\n== Top 8 por RPS de validacao (2016-2022) | hold-out 2023-2025 ==")
    print(tab.head(8).to_string(index=False))
    base = tab[(tab.half_life == BASELINE[0]) & (tab.ridge == BASELINE[1]) & (tab.alpha == BASELINE[2])]
    print("\n== Configuracao atual (baseline) ==")
    print(base.to_string(index=False))

    best = tuple(tab.iloc[0][["half_life", "ridge", "alpha"]])
    print(f"\nSelecionada em 2016-2022: half_life={best[0]:.0f} ridge={best[1]} alpha={best[2]}")
    for name, cfg in (("selecionada", best), ("so alpha=0.8 (ridge/half-life atuais)", (730.0, 0.5, 0.8))):
        rb = results[BASELINE].query("season in @HOLDOUT").set_index("match_id")
        rs = results[cfg].query("season in @HOLDOUT").set_index("match_id")
        print(f"\n-- {name} vs baseline no hold-out 2023-2025 (n={len(rb)}) --")
        for col in ("rps", "logloss", "over_brier"):
            m, lo, hi = evaluate.paired_bootstrap_diff(rs[col].to_numpy(), rb.loc[rs.index, col].to_numpy())
            print(f"{col:11s} {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]   (negativo = melhor)")

    print("\n== Calibracao dos extremos (2016-2025, todas as temporadas) ==")
    for name, cfg in (("baseline", BASELINE), ("selecionada", best), ("alpha=0.8", (730.0, 0.5, 0.8))):
        r = results[cfg]
        hi = r[r["p_home"] >= 0.65]
        ov = r[r["p_over"] >= 0.55]
        print(f"{name:12s} mandante>=65%: n={len(hi):4d} previsto={hi['p_home'].mean():.3f} "
              f"observado={hi['home_win'].mean():.3f} | over>=55%: n={len(ov):4d} "
              f"previsto={ov['p_over'].mean():.3f} observado={ov['over_hit'].mean():.3f}")


if __name__ == "__main__":
    main()
