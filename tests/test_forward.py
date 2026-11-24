"""Forward-test tests: logs are causal, immutable, and scoring only sees later results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import forward, ingestion


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return ingestion.build_interim()


def test_simulated_forward_run(matches: pd.DataFrame) -> None:
    held = 10
    train = matches.iloc[:-held]
    log = forward.make_log(train, now=pd.Timestamp("2026-11-20 12:00:00"))
    assert len(log) == 21 + held
    assert np.allclose(log[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-6)

    scored = forward.score_log(log, matches)  # full table includes the 10 held-out results
    assert len(scored) == held
    assert set(scored["home"] + scored["away"]) <= set(matches.iloc[-held:]["home_team"]
                                                         + matches.iloc[-held:]["away_team"])
    assert scored["rps"].between(0, 1).all() and scored["odds_rps"].notna().all()
    assert scored["in_top3"].ge(scored["in_top1"]).all()


def test_log_ignores_unplayed_and_training_matches(matches: pd.DataFrame) -> None:
    log = forward.make_log(matches, now=pd.Timestamp("2026-11-24 12:00:00"))
    scored = forward.score_log(log, matches)
    assert scored.empty  # nothing in the log has been played yet


def test_log_is_immutable(matches: pd.DataFrame, tmp_path) -> None:
    log = forward.make_log(matches, now=pd.Timestamp("2026-11-24 12:00:00"))
    path = forward.save_log(log, tmp_path)
    with pytest.raises(FileExistsError):
        forward.save_log(log, tmp_path)
    assert len(forward.load_logs(tmp_path)) == len(log) and path.exists()


def test_earliest_log_wins(matches: pd.DataFrame, tmp_path) -> None:
    a = forward.make_log(matches, now=pd.Timestamp("2026-11-24 12:00:00"))
    b = a.copy()
    b["predicted_at"] = "2026-11-25 12:00:00"
    b["p_home"] = 0.99
    forward.save_log(a, tmp_path)
    forward.save_log(b, tmp_path)
    loaded = forward.load_logs(tmp_path)
    assert len(loaded) == len(a) and (loaded["p_home"] < 0.99).all()
