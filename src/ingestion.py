"""Load, validate and normalize the raw football-data.co.uk Brasileirão CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Season", "Date", "Time", "Home", "Away", "HG", "AG", "Res",
)
ODDS_COLUMNS: tuple[str, ...] = (
    "PSCH", "PSCD", "PSCA", "AvgCH", "AvgCD", "AvgCA", "MaxCH", "MaxCD", "MaxCA",
)


class SchemaError(ValueError):
    """Raised when the raw data violates the expected schema."""


def load_raw(path: Path = config.RAW_MATCHES_FILE) -> pd.DataFrame:
    """Read the raw CSV, dropping fully empty columns.

    Parameters
    ----------
    path : Path
        Location of the raw CSV.

    Returns
    -------
    pd.DataFrame
        Raw dataframe, untouched apart from empty-column removal.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw matches: canonical teams, kickoff timestamp, ordering.

    The kickoff time is stored as provided by the source (timezone not
    documented); it is only used for ordering matches within a day.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_raw`.

    Returns
    -------
    pd.DataFrame
        Clean matches sorted by kickoff, with a stable ``match_id``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    # Unplayed fixtures (no score) are dropped. Known case: Chapecoense vs
    # Atletico-MG, 2016-12-11, never played after the Nov 2016 plane crash.
    out = df.dropna(subset=["HG", "AG"]).copy()
    unknown = (set(out["Home"]) | set(out["Away"])) - set(config.TEAM_MAP)
    if unknown:
        raise SchemaError(f"Teams missing from TEAM_MAP: {sorted(unknown)}")

    out["home_team"] = out["Home"].map(config.TEAM_MAP)
    out["away_team"] = out["Away"].map(config.TEAM_MAP)
    out["kickoff"] = pd.to_datetime(
        out["Date"].astype(str) + " " + out["Time"].astype(str),
        format="%d/%m/%Y %H:%M",
    )
    out = out.rename(
        columns={"Season": "season", "HG": "home_goals", "AG": "away_goals", "Res": "result"}
    )
    out["season"] = out["season"].astype(int)
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)

    odds = [c for c in ODDS_COLUMNS if c in out.columns]
    keep = ["season", "kickoff", "home_team", "away_team",
            "home_goals", "away_goals", "result", *odds]
    out = out[keep].sort_values(["kickoff", "home_team"], kind="mergesort")
    out = out.reset_index(drop=True)
    out.insert(0, "match_id", range(len(out)))
    return out


def validate(df: pd.DataFrame) -> None:
    """Validate a normalized matches table; raise :class:`SchemaError` on problems.

    Checks: no null goals, non-negative goals, ``result`` consistent with
    goals, no duplicate fixtures, no team playing itself, 20 teams per full season,
    sorted by kickoff.
    """
    if df[["home_goals", "away_goals"]].isna().any().any():
        raise SchemaError("Null goals found")
    if (df[["home_goals", "away_goals"]] < 0).any().any():
        raise SchemaError("Negative goals found")

    expected = pd.Series("D", index=df.index)
    expected[df["home_goals"] > df["away_goals"]] = "H"
    expected[df["home_goals"] < df["away_goals"]] = "A"
    if (expected != df["result"]).any():
        raise SchemaError("`result` inconsistent with goals")

    if (df["home_team"] == df["away_team"]).any():
        raise SchemaError("A team plays itself")
    if df.duplicated(["season", "home_team", "away_team"]).any():
        raise SchemaError("Duplicate fixture within a season")
    if not df["kickoff"].is_monotonic_increasing:
        raise SchemaError("Matches not sorted by kickoff")

    for season, g in df.groupby("season"):
        n_teams = len(set(g["home_team"]) | set(g["away_team"]))
        if n_teams != 20:
            raise SchemaError(f"Season {season}: expected 20 teams, found {n_teams}")


def build_interim(
    raw_path: Path = config.RAW_MATCHES_FILE,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Run load -> normalize -> validate; optionally persist to parquet/CSV.

    Parameters
    ----------
    raw_path : Path
        Raw CSV path.
    out_path : Path or None
        If given, the clean table is written there (``.csv`` or ``.parquet``).

    Returns
    -------
    pd.DataFrame
        Validated matches table.
    """
    df = normalize(load_raw(raw_path))
    validate(df)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".csv":
            df.to_csv(out_path, index=False)
        else:
            df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    matches = build_interim(out_path=config.DATA_INTERIM / "matches.csv")
    print(f"{len(matches)} matches, seasons {matches.season.min()}-{matches.season.max()}")
