"""Offline tests for the refresh command (downloads use file:// URLs)."""
from __future__ import annotations

import shutil

import pandas as pd
import pytest

from src import config, forward, ingestion, update


def test_download_adds_new_rows(tmp_path) -> None:
    src = config.RAW_MATCHES_FILE
    lines = src.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    dest = tmp_path / "BRA.csv"
    dest.write_text("".join(lines[:-5]), encoding="utf-8")  # old copy lacks the last 5 matches
    added = update.download_raw(src.as_uri(), dest)
    assert added == 5
    assert len(ingestion.build_interim(dest)) == len(ingestion.build_interim(src))
    assert not dest.with_suffix(".download").exists()


def test_invalid_download_keeps_current_file(tmp_path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("Season,Date,Time,Home,Away,HG,AG,Res\n2027,01/02/2027,20:00,Nowhere FC,Santos,1,0,H\n")
    dest = tmp_path / "BRA.csv"
    shutil.copy(config.RAW_MATCHES_FILE, dest)
    before = dest.read_bytes()
    with pytest.raises(ingestion.SchemaError, match="TEAM_MAP"):
        update.download_raw(bad.as_uri(), dest)
    assert dest.read_bytes() == before
    assert not dest.with_suffix(".download").exists()


def test_unreachable_url_raises_runtime_error(tmp_path) -> None:
    dest = tmp_path / "BRA.csv"
    shutil.copy(config.RAW_MATCHES_FILE, dest)
    with pytest.raises(RuntimeError):
        update.download_raw((tmp_path / "missing.csv").as_uri(), dest)


def test_partial_new_season_passes_validation() -> None:
    m = ingestion.build_interim()
    partial = pd.concat([m[m["season"] < 2026], m[m["season"] == 2026].iloc[:15]], ignore_index=True)
    ingestion.validate(partial)  # must not demand 20 teams after only 15 matches


def test_new_matchups_filters_already_logged(tmp_path) -> None:
    m = ingestion.build_interim()
    log = forward.make_log(m, now=pd.Timestamp("2026-11-24 12:00:00"))
    forward.save_log(log.iloc[:15], tmp_path)
    fresh = update.new_matchups(log, tmp_path)
    assert len(fresh) == len(log) - 15
    assert update.new_matchups(log, tmp_path / "empty").shape == log.shape
