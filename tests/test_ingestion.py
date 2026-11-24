"""Ingestion tests on the real raw CSV."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config, ingestion


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return ingestion.build_interim()


def test_loads_all_seasons(matches: pd.DataFrame) -> None:
    assert matches["season"].min() == 2012
    assert matches["season"].max() >= 2025
    complete = matches[matches["season"] <= 2025].groupby("season").size()
    # 2016: Chapecoense-Atletico-MG was never played (plane crash) -> 379.
    expected = pd.Series(380, index=complete.index)
    expected.loc[2016] = 379
    assert (complete == expected).all()


def test_no_duplicates_or_null_goals(matches: pd.DataFrame) -> None:
    assert not matches.duplicated(["season", "home_team", "away_team"]).any()
    assert matches[["home_goals", "away_goals"]].notna().all().all()


def test_sorted_and_ids_stable(matches: pd.DataFrame) -> None:
    assert matches["kickoff"].is_monotonic_increasing
    assert matches["match_id"].tolist() == list(range(len(matches)))


def test_team_map_covers_source() -> None:
    raw = ingestion.load_raw()
    assert (set(raw["Home"]) | set(raw["Away"])) <= set(config.TEAM_MAP)


def test_validate_catches_bad_result(matches: pd.DataFrame) -> None:
    bad = matches.copy()
    bad.loc[0, "result"] = "A" if bad.loc[0, "result"] != "A" else "H"
    with pytest.raises(ingestion.SchemaError):
        ingestion.validate(bad)


def test_validate_catches_self_match(matches: pd.DataFrame) -> None:
    bad = matches.copy()
    bad.loc[0, "away_team"] = bad.loc[0, "home_team"]
    with pytest.raises(ingestion.SchemaError):
        ingestion.validate(bad)
