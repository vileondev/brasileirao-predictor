"""Forward test: freeze predictions before kickoff, score them once results arrive.

Workflow
--------
1. ``python -m src.forward log``    -> fit on all played matches, predict every
   still-unplayed matchup of the season and save an immutable CSV in ``predictions/``
   (commit it: the git timestamp is the proof it was made before the games).
2. refresh ``data/raw/BRA_footballdata.csv`` (football-data.co.uk updates it)
3. ``python -m src.forward score``  -> compare logged predictions with the played
   results and with the bookmaker closing odds on the same matches.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import baselines, config, evaluate, fixtures, ingestion, predict

PRED_DIR: Path = config.ROOT / "predictions"
MODEL_TAG = "dixon_coles_hl730_ridge0.5"


def make_log(matches: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Predict every unplayed matchup of the latest season using only ``matches``.

    Parameters
    ----------
    matches : pd.DataFrame
        All played matches known at prediction time (from ``ingestion.build_interim``).
    now : pd.Timestamp, optional
        Wall-clock stamp stored in the log (default: now).
    """
    art = predict.fit_production(matches)
    rem = fixtures.remaining_fixtures(matches)
    res = predict.predict_fixtures(art, rem)
    res.insert(0, "season", int(matches["season"].max()))
    res["model"] = MODEL_TAG
    res["trained_through"] = art["trained_through"]
    res["predicted_at"] = str(now or pd.Timestamp.now().floor("s"))
    return res


def save_log(log: pd.DataFrame, directory: Path = PRED_DIR, force: bool = False) -> Path:
    """Write the log to ``<directory>/<date>_remaining.csv`` (never overwrites unless ``force``)."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = str(log["predicted_at"].iloc[0])[:10]
    path = directory / f"{stamp}_remaining.csv"
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; predictions are immutable (use --force to replace)")
    log.to_csv(path, index=False)
    return path


def load_logs(directory: Path = PRED_DIR) -> pd.DataFrame:
    """All logged predictions; if a matchup was logged twice the earliest one wins."""
    files = sorted(directory.glob("*_remaining.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("predicted_at").drop_duplicates(["season", "home", "away"], keep="first")
    return df.reset_index(drop=True)


def score_log(log: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Join logged predictions with played results; return per-match scores.

    Only matches that were NOT in the training data at log time can appear here
    (the log only contains unplayed matchups), so there is no look-ahead.
    """
    played = matches.rename(columns={"home_team": "home", "away_team": "away"})
    cols = ["season", "home", "away", "kickoff", "home_goals", "away_goals",
            *[c for c in ("AvgCH", "AvgCD", "AvgCA") if c in played.columns]]
    df = log.merge(played[cols], on=["season", "home", "away"], how="inner")
    if df.empty:
        return df
    y = evaluate.outcome_index(df["home_goals"].to_numpy(), df["away_goals"].to_numpy())
    p = df[["p_home", "p_draw", "p_away"]].to_numpy()
    df["rps"] = evaluate.rps_per_match(p, y)
    df["logloss"] = evaluate.log_loss_per_match(p, y)
    df["correct_1x2"] = (p.argmax(axis=1) == y).astype(float)
    actual = df["home_goals"].astype(int).astype(str) + "-" + df["away_goals"].astype(int).astype(str)
    for k in (1, 2, 3):
        df[f"in_top{k}"] = (actual == df[f"top{k}"].str.split(" ").str[0]).astype(float)
    df["in_top3"] = df[["in_top1", "in_top2", "in_top3"]].max(axis=1)
    hit = ((df["home_goals"] + df["away_goals"]) >= 3).astype(float)
    df["over_brier"] = (df["p_over_2_5"] - hit) ** 2
    if {"AvgCH", "AvgCD", "AvgCA"} <= set(df.columns):
        op = baselines.odds_probs(df, "AvgC")
        df["odds_rps"] = evaluate.rps_per_match(op, y)
        df["odds_logloss"] = evaluate.log_loss_per_match(op, y)
    return df


def _report(scored: pd.DataFrame, total: int) -> None:
    pd.options.display.float_format = "{:.4f}".format
    n = len(scored)
    print(f"{n} de {total} jogos previstos ja foram disputados.")
    if n == 0:
        return
    show = scored[["kickoff", "home", "away", "home_goals", "away_goals", "p_home", "p_draw",
                   "p_away", "top1", "rps"]].sort_values("kickoff")
    print(show.to_string(index=False))
    print(f"\nRPS modelo: {scored['rps'].mean():.4f} | log-loss: {scored['logloss'].mean():.4f} | "
          f"acerto 1X2: {scored['correct_1x2'].mean():.1%} | placar exato (top1): "
          f"{scored['in_top1'].mean():.1%} | (top3): {scored['in_top3'].mean():.1%}")
    if "odds_rps" in scored:
        print(f"RPS odds (mesmos jogos): {scored['odds_rps'].mean():.4f} | log-loss odds: "
              f"{scored['odds_logloss'].mean():.4f}")
    if n < 30:
        print(f"\nAVISO: n={n} e pequeno demais para concluir qualquer coisa; use so como acompanhamento.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["log", "score"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    matches = ingestion.build_interim()
    if args.action == "log":
        log = make_log(matches)
        path = save_log(log, force=args.force)
        print(f"{len(log)} previsoes salvas em {path} (modelo ate {log['trained_through'].iloc[0]})")
    else:
        log = load_logs()
        if log.empty:
            print("Nenhuma previsao registrada. Rode: python -m src.forward log")
            return
        _report(score_log(log, matches), len(log))


if __name__ == "__main__":
    main()
