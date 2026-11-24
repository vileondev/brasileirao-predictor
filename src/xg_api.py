"""Match statistics (xG, shots, possession) from API-Football, within the free-plan limits.

The free plan serves seasons 2022-2024 and 100 requests/day, one request per match
(``/fixtures/statistics``). This module keeps a resumable store under the git-ignored
``data/external/api_football/`` so a run can stop at the daily budget and continue the
next day without spending requests twice.

    python -m src.xg_api probe --n 8              # spread sample per season (coverage check)
    python -m src.xg_api download --budget 60     # fetch missing matches, newest season first
    python -m src.xg_api report                   # coverage and sanity checks
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src import config, fixtures, predict

FINISHED = ("FT", "AET", "PEN")
STAT_TYPES = {"expected_goals": "xg", "Total Shots": "shots", "Shots on Goal": "shots_on", "Ball Possession": "possession",
              "Corner Kicks": "corners", "Fouls": "fouls", "Goalkeeper Saves": "saves", "Total passes": "passes"}


def _stats_dir(store: Path) -> Path:
    return store / "stats"


def load_fixtures(season: int, store: Path = config.EXTERNAL_API_DIR) -> pd.DataFrame:
    """Finished fixtures of a season (1 request the first time, then read from the store)."""
    path = store / f"fixtures_{season}.json"
    if path.exists():
        body = json.loads(path.read_text(encoding="utf-8"))
    else:
        body = fixtures.api_get("/fixtures", {"league": fixtures.LEAGUE_ID_SERIE_A, "season": season}, write_cache=False)
        store.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body), encoding="utf-8")
    rows = []
    for it in body.get("response", []):
        ft = it["score"]["fulltime"]
        if it["fixture"]["status"]["short"] not in FINISHED or ft["home"] is None:
            continue
        rows.append({"fixture_id": it["fixture"]["id"], "season": season,
                     "date": pd.to_datetime(it["fixture"]["date"]).tz_convert("UTC").tz_localize(None),
                     "round": it["league"].get("round"),
                     "home": predict.resolve_team(it["teams"]["home"]["name"]), "away": predict.resolve_team(it["teams"]["away"]["name"]),
                     "home_goals": int(ft["home"]), "away_goals": int(ft["away"])})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def downloaded_ids(store: Path = config.EXTERNAL_API_DIR) -> set[int]:
    """Fixture ids whose statistics are already in the store."""
    d = _stats_dir(store)
    return {int(p.stem) for p in d.glob("*.json")} if d.exists() else set()


def spread_sample(fx: pd.DataFrame, n: int) -> list[int]:
    """``n`` fixture ids evenly spread over the season (by date)."""
    if n >= len(fx):
        return fx["fixture_id"].tolist()
    idx = np.linspace(0, len(fx) - 1, n).round().astype(int)
    return fx["fixture_id"].iloc[sorted(set(idx))].tolist()


def download(
    ids: Sequence[int], budget: int, store: Path = config.EXTERNAL_API_DIR, sleep: float = config.API_SLEEP_SECONDS,
    min_remaining: int = config.API_MIN_REMAINING, fetch: Callable[..., dict] = fixtures.api_get,
    pause: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Fetch statistics for ``ids`` not yet stored, stopping at ``budget`` or when the quota is low.

    Parameters
    ----------
    ids : sequence of int
        Fixture ids in priority order.
    budget : int
        Maximum number of requests to spend in this call.
    min_remaining : int
        Stop when the API reports this many requests (or fewer) left for the day.

    Returns
    -------
    dict
        ``fetched``, ``skipped`` (already stored), ``remaining`` (last reported) and ``stopped``.
    """
    d = _stats_dir(store)
    d.mkdir(parents=True, exist_ok=True)
    have = downloaded_ids(store)
    todo = [i for i in ids if i not in have]
    out: dict[str, object] = {"fetched": 0, "skipped": len(ids) - len(todo), "remaining": None, "stopped": "done"}
    for fid in todo:
        if out["fetched"] >= budget:
            out["stopped"] = "budget"
            break
        try:
            body = fetch("/fixtures/statistics", {"fixture": fid}, use_cache=False, write_cache=False)
        except fixtures.ApiError as e:  # rate limit or plan error: stop cleanly, keep what we have
            out["stopped"] = f"api error: {e}"
            break
        (d / f"{fid}.json").write_text(json.dumps({"fixture": fid, "response": body.get("response", [])}), encoding="utf-8")
        out["fetched"] += 1
        rem = body.get("_remaining_requests")
        out["remaining"] = int(rem) if rem is not None else None
        if out["remaining"] is not None and out["remaining"] <= min_remaining:
            out["stopped"] = f"quota low ({out['remaining']} left)"
            break
        pause(sleep)
    return out


def _num(v: object) -> float:
    if v is None:
        return np.nan
    try:
        return float(str(v).replace("%", ""))
    except ValueError:
        return np.nan


def parse_store(store: Path = config.EXTERNAL_API_DIR, seasons: Sequence[int] = config.API_FREE_SEASONS) -> pd.DataFrame:
    """One row per stored match: score from the fixture list, xG/shots/possession per side."""
    d = _stats_dir(store)
    fx = pd.concat([load_fixtures(s, store) for s in seasons if (store / f"fixtures_{s}.json").exists()], ignore_index=True)
    rows = []
    for fid in sorted(downloaded_ids(store)):
        body = json.loads((d / f"{fid}.json").read_text(encoding="utf-8"))
        sides = body["response"]
        rec: dict[str, object] = {"fixture_id": fid}
        for k, team in zip(("h", "a"), sides[:2]):
            stats = {s["type"]: s["value"] for s in team["statistics"]}
            for raw, short in STAT_TYPES.items():
                rec[f"{short}_{k}"] = _num(stats.get(raw))
        rows.append(rec)
    st = pd.DataFrame(rows)
    return fx.merge(st, on="fixture_id", how="inner") if len(st) else pd.DataFrame()


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Share of stored matches with xG / shots by season."""
    g = df.groupby("season")
    return pd.DataFrame({"baixados": g.size(), "com_xg": g["xg_h"].apply(lambda s: int(s.notna().sum())),
                         "com_chutes": g["shots_h"].apply(lambda s: int(s.notna().sum()))}).assign(
        pct_xg=lambda t: t["com_xg"] / t["baixados"] * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["probe", "download", "report"])
    ap.add_argument("--n", type=int, default=8, help="matches per season for the probe")
    ap.add_argument("--budget", type=int, default=60, help="max requests for this run")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2023, 2022])
    args = ap.parse_args()
    pd.options.display.float_format = "{:.2f}".format

    if args.action == "report":
        df = parse_store()
        print(coverage(df).to_string() if len(df) else "nada baixado ainda")
        return
    ids: list[int] = []
    for s in args.seasons:
        fx = load_fixtures(s)
        ids += spread_sample(fx, args.n) if args.action == "probe" else fx["fixture_id"].tolist()
    res = download(ids, args.budget)
    print(f"baixados agora: {res['fetched']} | ja existiam: {res['skipped']} | cota restante: {res['remaining']} | parou: {res['stopped']}")
    df = parse_store()
    if len(df):
        print(coverage(df).to_string())


if __name__ == "__main__":
    main()
