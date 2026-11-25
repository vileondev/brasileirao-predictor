"""Load, validate and normalize the raw football-data.co.uk Brasileirão CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
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


def validate(df: pd.DataFrame, n_teams: int = config.TEAMS_PER_SEASON) -> None:
    """Validate a normalized matches table; raise :class:`SchemaError` on problems.

    Checks: no null goals, non-negative goals, ``result`` consistent with
    goals, no duplicate fixtures, no team playing itself, ``n_teams`` teams per
    full season, sorted by kickoff.
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
        found = len(set(g["home_team"]) | set(g["away_team"]))
        # a season only just started has not shown all its teams yet
        if found > n_teams or (found != n_teams and len(g) >= 5 * n_teams):
            raise SchemaError(f"Season {season}: expected {n_teams} teams, found {found}")


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


FD_STAT_COLUMNS: tuple[str, ...] = ("HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR")


def load_football_data(files: list[Path]) -> pd.DataFrame:
    """Read football-data.co.uk season files (``<code>_YYZZ.csv``) and tag each row with its season.

    The season is the year the season *starts* (``E0_1213.csv`` -> 2012); it is taken from the
    file name, not from the dates. Rows without both team names (blank lines) are dropped.
    """
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, encoding="utf-8-sig")
        except UnicodeDecodeError:
            d = pd.read_csv(f, encoding="latin-1")
        d.columns = [str(c).strip() for c in d.columns]
        missing = [c for c in ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR") if c not in d.columns]
        if missing:
            raise SchemaError(f"{f.name}: missing columns {missing}")
        d = d.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
        yy = int(f.stem.split("_")[-1][:2])
        d["season"] = (1900 if yy >= 90 else 2000) + yy
        frames.append(d)
    if not frames:
        raise SchemaError("no football-data files found")
    return pd.concat(frames, ignore_index=True)


def normalize_football_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize football-data season files into the pipeline schema.

    Benchmark columns: ``PSC*`` is Pinnacle closing; ``AvgC*`` is the market-average closing odds
    when the file has them (2019/20+) and Pinnacle closing otherwise (earlier seasons), so a single
    benchmark exists for every match. Kickoff is ``Date`` + ``Time`` (15:00 when the file has no
    time), only used to order matches. Unplayed rows (no score) are dropped.
    """
    d = raw.dropna(subset=["FTHG", "FTAG"]).copy()
    date = pd.to_datetime(d["Date"], dayfirst=True, format="mixed")  # files mix dd/mm/yy and dd/mm/yyyy
    time = d["Time"].astype(str).str.strip().where(d["Time"].notna(), "15:00") if "Time" in d.columns else "15:00"
    out = pd.DataFrame({
        "season": d["season"].astype(int),
        "kickoff": pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + time),
        "home_team": d["HomeTeam"].astype(str).str.strip(), "away_team": d["AwayTeam"].astype(str).str.strip(),
        "home_goals": d["FTHG"].astype(int), "away_goals": d["FTAG"].astype(int), "result": d["FTR"].astype(str).str.strip(),
    }, index=d.index)
    for side in "HDA":
        out[f"PSC{side}"] = d[f"PSC{side}"] if f"PSC{side}" in d.columns else np.nan
        avg = d[f"AvgC{side}"] if f"AvgC{side}" in d.columns else pd.Series(np.nan, index=d.index)
        out[f"AvgC{side}"] = avg.fillna(out[f"PSC{side}"])
    for c in FD_STAT_COLUMNS:
        if c in d.columns:
            out[c] = pd.to_numeric(d[c], errors="coerce")
    out = out.sort_values(["kickoff", "home_team"], kind="mergesort").reset_index(drop=True)
    out.insert(0, "match_id", range(len(out)))
    return out


def build_league(league: "object", out_path: Path | None = None) -> pd.DataFrame:
    """Load, normalize and validate the matches of any registered league.

    Brazil keeps its own parser (:func:`build_interim`); football-data "main" leagues use
    :func:`load_football_data` / :func:`normalize_football_data`.
    """
    from src import leagues

    lg = leagues.get(league)  # type: ignore[arg-type]
    if lg.source == "football-data-new":
        return build_interim(out_path=out_path)
    df = normalize_football_data(load_football_data(lg.data_files()))
    validate(df, n_teams=lg.n_teams)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    matches = build_interim(out_path=config.DATA_INTERIM / "matches.csv")
    print(f"{len(matches)} matches, seasons {matches.season.min()}-{matches.season.max()}")
