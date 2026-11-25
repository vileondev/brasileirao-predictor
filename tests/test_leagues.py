"""Tests for the per-league parametrisation (Brazil regression + Premier League)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtest, fixtures, forward, ingestion, leagues, predict

E0 = leagues.get("E0")
needs_e0 = pytest.mark.skipif(not E0.data_files(), reason="Premier League files not present")


def test_registry_and_lookup() -> None:
    assert leagues.get("bra") is leagues.BRA and leagues.get(E0) is E0
    assert {"BRA", "E0", "SP1", "I1", "D1", "F1"} <= set(leagues.LEAGUES)
    with pytest.raises(KeyError, match="Unknown league"):
        leagues.get("XX9")
    assert E0.games_per_team == 38 and leagues.get("D1").games_per_team == 34 and leagues.get("D1").matches_per_season == 306
    assert E0.season_tag(2012) == "1213" and E0.season_tag(1999) == "9900"
    assert E0.model_file.name == "dc_E0.json" and leagues.BRA.model_file.name == "dc_production.json"  # Brazil path unchanged
    assert E0.pred_dir.name == "E0" and leagues.BRA.pred_dir.name == "predictions"


def test_current_season_labels() -> None:
    from datetime import date
    assert E0.current_season(date(2026, 11, 25)) == 2026 and E0.current_season(date(2027, 3, 1)) == 2026
    assert E0.current_season(date(2026, 6, 30)) == 2025 and leagues.BRA.current_season(date(2026, 11, 25)) == 2026


def test_brazil_default_is_unchanged() -> None:
    a, b = ingestion.build_league("BRA"), ingestion.build_interim()
    pd.testing.assert_frame_equal(a, b)
    assert predict.resolve_team("Flamengo RJ") == "Flamengo" and predict.resolve_team("Vasco DA Gama") == "Vasco"
    assert predict.resolve_team("Flamengo RJ", "BRA") == "Flamengo"


@needs_e0
def test_premier_league_ingestion() -> None:
    m = ingestion.build_league(E0)
    assert m["kickoff"].is_monotonic_increasing and m["match_id"].tolist() == list(range(len(m)))
    full = m[m.season <= 2025].groupby("season").size()
    assert (full == 380).all() and m.season.min() == 2012
    assert m[["AvgCH", "AvgCD", "AvgCA"]].notna().all().all()  # benchmark resolver: AvgC, else Pinnacle closing
    assert {"HS", "AS", "HST", "AST"} <= set(m.columns) and m["HS"].notna().all()
    assert not m.duplicated(["season", "home_team", "away_team"]).any()
    assert E0.test_season(m) == 2025


@needs_e0
def test_premier_league_team_resolution() -> None:
    r = leagues.resolver(E0)
    assert r("Manchester United") == "Man United" and r("Nottingham Forest") == "Nott'm Forest"
    assert r("Wolves") == "Wolves" and r("Tottenham Hotspur") == "Tottenham" and r("Man City") == "Man City"
    assert r("Time Novo FC") == "Time Novo FC"  # unknown stays unchanged
    assert predict.resolve_team("Newcastle United", E0) == "Newcastle"


@needs_e0
def test_premier_league_prediction_and_forward_log(tmp_path: Path) -> None:
    m = ingestion.build_league(E0)
    art = predict.fit_production(m[m.season <= 2025])
    res = predict.predict_fixtures(art, pd.DataFrame({"home": ["Manchester City"], "away": ["Ipswich Town"]}), league=E0)
    r = res.iloc[0]
    assert r.warning == "" and r.p_home + r.p_draw + r.p_away == pytest.approx(1.0, abs=1e-6) and r.p_home > r.p_away
    log = forward.make_log(m, now=pd.Timestamp("2026-11-25 12:00:00"), league=E0)
    assert (log["league"] == "E0").all() and len(log) == E0.matches_per_season - int((m.season == 2026).sum())
    assert np.allclose(log[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-6)
    path = forward.save_log(log, tmp_path)
    assert path.exists()


@needs_e0
def test_backtest_runs_with_league_parameters() -> None:
    m = ingestion.build_league(E0)
    m = m[m.season <= 2017].reset_index(drop=True)
    preds = backtest.run_backtest(m, first_val_season=E0.first_validation_season, warmup_seasons=E0.warmup_seasons)
    assert set(preds["season"]) == {2016, 2017} and set(preds["model"]) >= {"dixon_coles", "elo_poisson", "odds_avg"}
    p = preds[preds.model == "dixon_coles"][["p_home", "p_draw", "p_away"]].to_numpy()
    assert len(p) == 760 and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


@needs_e0
def test_fd_fixtures_loader_filters_the_league() -> None:
    f = fixtures.load_fd_fixtures(E0)
    assert list(f.columns) == ["round", "kickoff", "home", "away"]
    assert f["kickoff"].dtype.kind == "M" or f.empty
    raw = pd.read_csv(E0.fixtures_file, encoding="utf-8-sig")
    assert len(f) == int((raw["Div"] == "E0").sum())  # only this league's rows are kept


def test_football_data_download_offline(tmp_path: Path) -> None:
    """Serve football-data-style files through file:// URLs and check the downloader."""
    from datetime import date
    server = tmp_path / "server"
    for tag in ("1213", "1314"):
        (server / "mmz4281" / tag).mkdir(parents=True)
        (server / "mmz4281" / tag / "E0.csv").write_text("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,18/08/2012,A,B,1,0,H\n")
    (server / "fixtures.csv").write_text("Div,Date,Time,HomeTeam,AwayTeam\nE0,01/12/2026,15:00,A,B\n")
    lg = leagues.League(code="E0", name="t", n_teams=2, source="football-data-main", raw_dir=tmp_path / "raw", first_season=2012,
                        model_file=tmp_path / "m.json", pred_dir=tmp_path / "p", warmup_seasons=(2012,), first_validation_season=2013,
                        div="E0", fixtures_file=tmp_path / "raw" / "fixtures.csv")
    res = leagues.download_football_data(lg, base_url=server.as_uri(), today=date(2013, 11, 1))
    assert sorted(res["changed"]) == ["E0_1213.csv", "E0_1314.csv", "fixtures.csv"] and res["missing"] == []
    res2 = leagues.download_football_data(lg, base_url=server.as_uri(), today=date(2013, 11, 1))
    assert "E0_1213.csv" in res2["unchanged"] and "E0_1314.csv" in res2["unchanged"]  # history kept, current unchanged
    assert (lg.raw_dir / "MANIFEST.json").exists()
    bad = server / "mmz4281" / "1314" / "E0.csv"
    bad.write_text("not,a,football,file\n1,2,3,4\n")
    before = (lg.raw_dir / "E0_1314.csv").read_bytes()
    with pytest.raises(RuntimeError, match="unexpected header"):
        leagues.download_football_data(lg, base_url=server.as_uri(), today=date(2013, 11, 1))
    assert (lg.raw_dir / "E0_1314.csv").read_bytes() == before  # an invalid download never replaces a good file
