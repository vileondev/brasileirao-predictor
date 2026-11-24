"""Calibration report for the walk-forward Dixon-Coles predictions (1X2 and Over 2.5)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import backtest, config, evaluate, ingestion, probability

REPORT_DIR = config.ROOT / "reports"


def ece(table: np.ndarray) -> float:
    """Expected calibration error from a :func:`evaluate.calibration_table` result."""
    w = table[:, 2] / table[:, 2].sum()
    return float((w * np.abs(table[:, 0] - table[:, 1])).sum())


def main() -> None:
    matches = ingestion.build_interim()
    preds = backtest.run_backtest(matches)
    preds = preds[preds["season"] <= config.TEST_SEASON]
    df = preds.merge(matches[["match_id", "home_goals", "away_goals"]], on="match_id")
    y = evaluate.outcome_index(df["home_goals"].to_numpy(), df["away_goals"].to_numpy())

    dc = df[df["model"] == "dixon_coles"].reset_index(drop=True)
    y_dc = evaluate.outcome_index(dc["home_goals"].to_numpy(), dc["away_goals"].to_numpy())
    rho = probability.fit_rho(dc["lam_home"].to_numpy(), dc["lam_away"].to_numpy(),
                              dc["home_goals"].to_numpy(), dc["away_goals"].to_numpy())
    over = probability.market_probs(
        probability.scoreline_matrix(dc["lam_home"].to_numpy(), dc["lam_away"].to_numpy(), rho))["over_2_5"]
    over_hit = ((dc["home_goals"] + dc["away_goals"]) >= 3).to_numpy().astype(float)

    REPORT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    panels = [("Mandante", dc["p_home"].to_numpy(), (y_dc == 0)),
              ("Empate", dc["p_draw"].to_numpy(), (y_dc == 1)),
              ("Visitante", dc["p_away"].to_numpy(), (y_dc == 2)),
              ("Over 2.5", over, over_hit.astype(bool))]
    print(f"Calibracao Dixon-Coles walk-forward 2016-{config.TEST_SEASON} (n={len(dc)})")
    for ax, (title, p, hit) in zip(axes, panels):
        tab = evaluate.calibration_table(p, hit.astype(float), bins=8)
        print(f"\n{title}: ECE={ece(tab):.4f}  prob media prevista={p.mean():.3f}  freq observada={hit.mean():.3f}")
        print("  prev   obs    n")
        for a, b, n in tab:
            print(f"  {a:.3f}  {b:.3f}  {int(n)}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(tab[:, 0], tab[:, 1], "o-")
        ax.set_title(f"{title} (ECE {ece(tab):.3f})")
        ax.set_xlabel("probabilidade prevista")
        ax.set_ylabel("frequencia observada")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "calibration_dc.png", dpi=110)

    odds = df[df["model"] == "odds_avg"].reset_index(drop=True)
    y_o = evaluate.outcome_index(odds["home_goals"].to_numpy(), odds["away_goals"].to_numpy())
    for name, frame, yy in (("Dixon-Coles", dc, y_dc), ("Odds", odds, y_o)):
        e = np.mean([ece(evaluate.calibration_table(frame[c].to_numpy(), (yy == k).astype(float), 8))
                     for k, c in enumerate(["p_home", "p_draw", "p_away"])])
        print(f"\nECE medio 1X2 {name}: {e:.4f}")
    print(f"\nrho (ajustado nas previsoes walk-forward): {rho:.3f}")
    print(f"grafico: {REPORT_DIR / 'calibration_dc.png'}")


if __name__ == "__main__":
    main()
