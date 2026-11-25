"""Offline tests for the API-Football statistics store (no network, no key)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import fixtures, xg_api


def _fixture_list(season: int, n: int = 12) -> dict:
    resp = []
    for i in range(n):
        resp.append({"fixture": {"id": 1000 + i, "date": f"{season}-05-{i + 1:02d}T20:00:00+00:00", "status": {"short": "FT"}},
                     "league": {"round": f"Regular Season - {i + 1}"},
                     "teams": {"home": {"name": "Flamengo"}, "away": {"name": "Vasco DA Gama"}},
                     "score": {"fulltime": {"home": i % 3, "away": 1}}})
    resp.append({"fixture": {"id": 9999, "date": f"{season}-12-01T20:00:00+00:00", "status": {"short": "NS"}},
                 "league": {"round": "x"}, "teams": {"home": {"name": "Santos"}, "away": {"name": "Bahia"}},
                 "score": {"fulltime": {"home": None, "away": None}}})
    return {"response": resp}


def _stats_body(xg: tuple[str | None, str | None]) -> dict:
    def mk(x: str | None, shots: int) -> dict:
        return {"statistics": [{"type": "expected_goals", "value": x}, {"type": "Total Shots", "value": shots},
                               {"type": "Ball Possession", "value": "63%"}]}
    return {"response": [mk(xg[0], 14), mk(xg[1], 7)], "_remaining_requests": "50"}


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    (tmp_path / "fixtures_2024.json").write_text(json.dumps(_fixture_list(2024)))
    return tmp_path


def test_load_fixtures_keeps_finished_only_and_canonical_names(store: Path) -> None:
    fx = xg_api.load_fixtures(2024, store)
    assert len(fx) == 12 and fx["fixture_id"].is_unique and fx["date"].is_monotonic_increasing
    assert set(fx["away"]) == {"Vasco"}  # "Vasco DA Gama" resolved to the canonical name


def test_spread_sample_is_ordered_and_spread(store: Path) -> None:
    fx = xg_api.load_fixtures(2024, store)
    ids = xg_api.spread_sample(fx, 4)
    assert len(ids) == 4 and ids[0] == fx.fixture_id.iloc[0] and ids[-1] == fx.fixture_id.iloc[-1]
    assert xg_api.spread_sample(fx, 50) == fx.fixture_id.tolist()


def test_download_respects_budget_and_is_resumable(store: Path) -> None:
    calls: list[int] = []

    def fake(path: str, params: dict, **kw: object) -> dict:
        calls.append(params["fixture"])
        return _stats_body(("1.2", "0.8"))

    ids = xg_api.load_fixtures(2024, store).fixture_id.tolist()
    r1 = xg_api.download(ids, budget=5, store=store, fetch=fake, pause=lambda s: None)
    assert r1["fetched"] == 5 and r1["stopped"] == "budget" and len(calls) == 5
    r2 = xg_api.download(ids, budget=100, store=store, fetch=fake, pause=lambda s: None)
    assert r2["skipped"] == 5 and r2["fetched"] == 7 and len(set(calls)) == len(calls) == 12  # never twice
    r3 = xg_api.download(ids, budget=100, store=store, fetch=fake, pause=lambda s: None)
    assert r3["fetched"] == 0 and r3["skipped"] == 12


def test_download_stops_when_quota_low_or_api_error(store: Path) -> None:
    ids = xg_api.load_fixtures(2024, store).fixture_id.tolist()

    def low(path: str, params: dict, **kw: object) -> dict:
        b = _stats_body(("1", "1"))
        b["_remaining_requests"] = "9"
        return b

    r = xg_api.download(ids, budget=50, store=store, fetch=low, pause=lambda s: None)
    assert r["fetched"] == 1 and str(r["stopped"]).startswith("quota low")

    def boom(path: str, params: dict, **kw: object) -> dict:
        raise fixtures.ApiError("rate limit")

    r = xg_api.download(ids, budget=50, store=store, fetch=boom, pause=lambda s: None)
    assert r["fetched"] == 0 and str(r["stopped"]).startswith("api error")


def test_parse_store_and_coverage(store: Path) -> None:
    ids = xg_api.load_fixtures(2024, store).fixture_id.tolist()

    def fake(path: str, params: dict, **kw: object) -> dict:
        return _stats_body((None, None) if params["fixture"] % 2 else ("1.38", "0.5"))

    xg_api.download(ids, budget=100, store=store, fetch=fake, pause=lambda s: None)
    df = xg_api.parse_store(store, seasons=(2024,))
    assert len(df) == 12 and df["possession_h"].eq(63.0).all()
    assert df["xg_h"].notna().sum() == 6 and df.loc[df.xg_h.notna(), "xg_h"].eq(1.38).all()
    cov = xg_api.coverage(df)
    assert int(cov.loc[2024, "com_xg"]) == 6 and cov.loc[2024, "pct_xg"] == pytest.approx(50.0)
    assert np.isfinite(df["shots_h"]).all()


def _injury_body() -> dict:
    mk = lambda pid, name, typ, reason, fid, team: {
        "player": {"id": pid, "name": name, "type": typ, "reason": reason}, "team": {"name": team},
        "fixture": {"id": fid, "date": "2024-08-16T19:00:00+00:00"}}
    return {"response": [mk(1, "A. One", "Missing Fixture", "Knee Injury", 10, "Manchester United"),
                         mk(2, "B. Two", "Questionable", "Muscle Injury", 10, "Fulham"),
                         mk(1, "A. One", "Missing Fixture", "Knee Injury", 10, "Manchester United")]}  # duplicate row


def test_fetch_injuries_is_idempotent_and_parsed(tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake(path: str, params: dict, **kw: object) -> dict:
        calls.append(params)
        return _injury_body()

    assert xg_api.fetch_injuries(39, 2024, tmp_path, fetch=fake) is True
    assert xg_api.fetch_injuries(39, 2024, tmp_path, fetch=fake) is False  # second call costs nothing
    assert calls == [{"league": 39, "season": 2024}]
    inj = xg_api.load_injuries(39, 2024, tmp_path)
    assert len(inj) == 2 and set(inj["type"]) == {"Missing Fixture", "Questionable"}
    assert inj["date"].dtype.kind == "M" and (inj["fixture_id"] == 10).all()
