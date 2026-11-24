"""Offline tests for the API-Football integration (no network, no key needed)."""
from __future__ import annotations

import pandas as pd
import pytest

from src import fixtures, predict

SAMPLE = {"response": [
    {"fixture": {"id": 1, "date": "2026-11-29T21:00:00+00:00", "status": {"short": "NS"}},
     "league": {"round": "Regular Season - 37"},
     "teams": {"home": {"name": "Flamengo"}, "away": {"name": "Vasco DA Gama"}}},
    {"fixture": {"id": 2, "date": "2026-11-29T19:00:00+00:00", "status": {"short": "NS"}},
     "league": {"round": "Regular Season - 37"},
     "teams": {"home": {"name": "Atlético-MG"}, "away": {"name": "RB Bragantino"}}},
]}


def test_parse_fixtures() -> None:
    df = fixtures.parse_fixtures(SAMPLE)
    assert list(df["fixture_id"]) == [1, 2]
    assert df["kickoff"].dtype.kind == "M" and (df["status"] == "NS").all()


@pytest.mark.parametrize("raw,expected", [
    ("Vasco DA Gama", "Vasco"), ("Atlético-MG", "Atletico-MG"), ("RB Bragantino", "Bragantino"),
    ("São Paulo", "Sao-Paulo"), ("Grêmio", "Gremio"), ("Athletico Paranaense", "Athletico-PR"),
    ("Sport Recife", "Sport"), ("Flamengo RJ", "Flamengo"), ("Botafogo", "Botafogo"),
])
def test_resolve_team_variants(raw: str, expected: str) -> None:
    assert predict.resolve_team(raw) == expected


def test_unrelated_name_is_left_alone() -> None:
    assert predict.resolve_team("Time Novo FC") == "Time Novo FC"


def test_api_fixture_names_all_resolve() -> None:
    df = fixtures.parse_fixtures(SAMPLE)
    names = pd.concat([df["home_api"], df["away_api"]]).map(predict.resolve_team)
    assert set(names) <= predict.CANONICAL_TEAMS


def test_missing_key_error_does_not_leak(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setattr(fixtures.config, "ROOT", tmp_path)
    with pytest.raises(fixtures.ApiError, match="not set"):
        fixtures.load_key()


def test_cache_path_is_stable_and_daily() -> None:
    a = fixtures._cache_path("/fixtures", {"league": 71, "season": 2026})
    b = fixtures._cache_path("/fixtures", {"season": 2026, "league": 71})
    assert a == b and pd.Timestamp.today().strftime("%Y-%m-%d") in a.name


def test_remaining_fixtures_on_real_data() -> None:
    from src import ingestion
    m = ingestion.build_interim()
    rem = fixtures.remaining_fixtures(m)
    played = int((m["season"] == m["season"].max()).sum())
    assert len(rem) == 380 - played
    s = m[m["season"] == m["season"].max()]
    assert not (set(zip(rem["home"], rem["away"])) & set(zip(s["home_team"], s["away_team"])))
    assert (rem["home"] != rem["away"]).all()


def test_remaining_fixtures_small_league() -> None:
    m = pd.DataFrame({"season": [2026, 2026], "home_team": ["A", "B"], "away_team": ["B", "C"]})
    rem = fixtures.remaining_fixtures(m)
    assert len(rem) == 4  # 3 teams -> 6 ordered pairs, 2 already played
    assert ("A", "B") not in set(zip(rem["home"], rem["away"]))


def test_schedule_covers_every_remaining_matchup() -> None:
    from src import ingestion
    rem = fixtures.remaining_fixtures(ingestion.build_interim())
    sched = fixtures.load_schedule()
    assert not sched.duplicated(["home", "away"]).any() and sched["kickoff"].notna().all()
    merged = fixtures.attach_schedule(rem, sched)
    assert merged["kickoff"].notna().all(), "a remaining matchup is missing from the schedule"
    assert merged["kickoff"].is_monotonic_increasing and len(merged) == len(rem)


def test_schedule_teams_are_canonical() -> None:
    sched = fixtures.load_schedule()
    assert set(sched["home"]) | set(sched["away"]) <= predict.CANONICAL_TEAMS
