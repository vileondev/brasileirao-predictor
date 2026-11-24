"""Pre-match feature engineering (results-only, strictly no look-ahead).

Every per-team rolling statistic is computed with ``shift(1)`` inside the
team's own chronological history, so the features of match ``t`` depend only on
matches of the two teams that kicked off before ``t``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import baselines, config

EWM_HALFLIFE: float = 4.0
VENUE_WINDOWS: tuple[int, ...] = (5, 10)
H2H_MEETINGS: int = 5
H2H_DECAY: float = 0.7
REST_CAP_DAYS: float = 30.0


def _team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) from that team's perspective."""
    def side(team: str, opp: str, gf: str, ga: str, venue: str) -> pd.DataFrame:
        return pd.DataFrame({
            "match_id": matches["match_id"], "season": matches["season"],
            "kickoff": matches["kickoff"], "team": matches[team],
            "venue": venue, "gf": matches[gf].astype(float), "ga": matches[ga].astype(float),
        })

    long = pd.concat([
        side("home_team", "away_team", "home_goals", "away_goals", "H"),
        side("away_team", "home_team", "away_goals", "home_goals", "A"),
    ], ignore_index=True)
    long["pts"] = np.where(long["gf"] > long["ga"], 3.0, np.where(long["gf"] == long["ga"], 1.0, 0.0))
    return long.sort_values(["team", "kickoff", "match_id"], kind="mergesort").reset_index(drop=True)


def _roll(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=min(window, 3)).mean()


def _team_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Per (match, team) pre-match features."""
    long = _team_long(matches)
    g = long.groupby("team", sort=False)

    for n in config.ROLLING_WINDOWS:
        for col in ("gf", "ga", "pts"):
            long[f"{col}_r{n}"] = g[col].transform(lambda s, n=n: _roll(s, n))
    for col in ("gf", "ga"):
        long[f"{col}_ewm"] = g[col].transform(
            lambda s: s.shift(1).ewm(halflife=EWM_HALFLIFE, min_periods=3).mean())

    gv = long.groupby(["team", "venue"], sort=False)
    for n in VENUE_WINDOWS:
        for col in ("gf", "ga", "pts"):
            long[f"{col}_v{n}"] = gv[col].transform(lambda s, n=n: _roll(s, n))

    long["games_played"] = g.cumcount().astype(float)
    long["games_in_season"] = long.groupby(["team", "season"], sort=False).cumcount().astype(float)
    prev_kick = g["kickoff"].shift(1)
    long["rest_days"] = ((long["kickoff"] - prev_kick).dt.total_seconds() / 86400.0).clip(upper=REST_CAP_DAYS)

    # promoted: team absent from the previous season's table (known before kickoff)
    by_season = long.groupby("season")["team"].agg(lambda x: set(x))
    first = long["season"].min()
    long["promoted"] = [
        0.0 if s == first else float(t not in by_season.get(s - 1, set()))
        for s, t in zip(long["season"], long["team"])
    ]
    return long


def _h2h(matches: pd.DataFrame) -> pd.Series:
    """Decayed goal difference (home perspective) over the last meetings of the pair."""
    hist: dict[frozenset, list[tuple[str, int]]] = {}
    out = np.full(len(matches), np.nan)
    for n, row in enumerate(matches.itertuples(index=False)):
        key = frozenset((row.home_team, row.away_team))
        past = hist.get(key, [])[-H2H_MEETINGS:]
        if past:
            num = den = 0.0
            for age, (home_then, gd_then) in enumerate(reversed(past)):
                gd = gd_then if home_then == row.home_team else -gd_then
                w = H2H_DECAY ** age
                num += w * gd
                den += w
            out[n] = num / den
        hist.setdefault(key, []).append((row.home_team, row.home_goals - row.away_goals))
    return pd.Series(out, index=matches.index, name="h2h_gd")


TEAM_FEATURES: tuple[str, ...] = (
    *(f"{c}_r{n}" for n in config.ROLLING_WINDOWS for c in ("gf", "ga", "pts")),
    "gf_ewm", "ga_ewm",
    *(f"{c}_v{n}" for n in VENUE_WINDOWS for c in ("gf", "ga", "pts")),
    "games_played", "games_in_season", "rest_days", "promoted",
)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build one pre-match feature row per match.

    Parameters
    ----------
    matches : pd.DataFrame
        Clean matches sorted by kickoff (output of ``ingestion.build_interim``).

    Returns
    -------
    pd.DataFrame
        Columns: identifiers (``match_id, season, kickoff, home_team, away_team``),
        targets (``home_goals, away_goals``), all ``h_*`` / ``a_*`` team features,
        Elo, rest/attack-vs-defence interactions and H2H. No target-match
        information appears in any feature column.
    """
    long = _team_features(matches)
    cols = list(TEAM_FEATURES)
    home = long[long["venue"] == "H"].set_index("match_id")[cols].add_prefix("h_")
    away = long[long["venue"] == "A"].set_index("match_id")[cols].add_prefix("a_")

    base = matches.set_index("match_id")
    elo = baselines.compute_elo(matches).set_axis(matches["match_id"])
    odds_cols = [c for c in ("AvgCH", "AvgCD", "AvgCA") if c in base.columns]
    df = base[["season", "kickoff", "home_team", "away_team", "home_goals", "away_goals", *odds_cols]] \
        .join(home).join(away).join(elo)

    df["h2h_gd"] = _h2h(matches).set_axis(matches["match_id"])
    df["rest_diff"] = df["h_rest_days"] - df["a_rest_days"]
    df["home_att_edge"] = df["h_gf_v5"] - df["a_ga_v5"]
    df["away_att_edge"] = df["a_gf_v5"] - df["h_ga_v5"]
    df["home_att_edge_ewm"] = df["h_gf_ewm"] - df["a_ga_ewm"]
    df["away_att_edge_ewm"] = df["a_gf_ewm"] - df["h_ga_ewm"]
    df["form_diff_r5"] = df["h_pts_r5"] - df["a_pts_r5"]
    df["matchday"] = df["h_games_in_season"] + 1.0

    if odds_cols:  # de-vigged closing odds: kept apart, used only in the labelled odds experiment
        inv = 1.0 / df[odds_cols].to_numpy(dtype=float)
        inv = inv / inv.sum(axis=1, keepdims=True)
        df["odds_p_home"], df["odds_p_draw"], df["odds_p_away"] = inv.T
        df = df.drop(columns=odds_cols)
    return df.reset_index()


ODDS_FEATURES: tuple[str, ...] = ("odds_p_home", "odds_p_draw", "odds_p_away")
ID_COLUMNS: tuple[str, ...] = ("match_id", "season", "kickoff", "home_team", "away_team",
                               "home_goals", "away_goals")


def feature_columns(df: pd.DataFrame, include_odds: bool = False) -> list[str]:
    """Model input columns in a fixed, saved order (odds only if explicitly requested)."""
    skip = set(ID_COLUMNS) | set(ODDS_FEATURES)
    cols = [c for c in df.columns if c not in skip]
    return cols + list(ODDS_FEATURES) if include_odds else cols
