"""Inference tests for the production Dixon-Coles pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import ingestion, predict


@pytest.fixture(scope="module")
def artifact() -> dict:
    return predict.fit_production(ingestion.build_interim())


def test_single_mock_fixture(artifact: dict) -> None:
    res = predict.predict_fixtures(artifact, pd.DataFrame({"home": ["Flamengo"], "away": ["Palmeiras"]}))
    r = res.iloc[0]
    assert r.p_home + r.p_draw + r.p_away == pytest.approx(1.0, abs=1e-6)
    assert 0 < r.p_over_2_5 < 1 and 0 < r.p_btts < 1
    assert r.lam_home > 0 and r.lam_away > 0 and r.warning == ""
    assert r.top1.count("-") == 1


def test_source_and_canonical_spellings_agree(artifact: dict) -> None:
    a = predict.predict_fixtures(artifact, pd.DataFrame({"home": ["Flamengo RJ"], "away": ["Sao Paulo"]}))
    b = predict.predict_fixtures(artifact, pd.DataFrame({"home": ["flamengo"], "away": ["Sao-Paulo"]}))
    assert np.allclose(a[["p_home", "p_draw", "p_away"]], b[["p_home", "p_draw", "p_away"]])


def test_unknown_team_is_flagged_not_crashing(artifact: dict) -> None:
    res = predict.predict_fixtures(artifact, pd.DataFrame({"home": ["Time Novo FC"], "away": ["Santos"]}))
    assert "Time Novo FC" in res.iloc[0].warning
    assert res.iloc[0].p_home + res.iloc[0].p_draw + res.iloc[0].p_away == pytest.approx(1.0, abs=1e-6)


def test_artifact_roundtrip(artifact: dict, tmp_path) -> None:
    path = tmp_path / "m.json"
    predict.save_artifact(artifact, path)
    fx = pd.DataFrame({"home": ["Gremio", "Vasco"], "away": ["Cruzeiro", "Bahia"]})
    a = predict.predict_fixtures(artifact, fx)
    b = predict.predict_fixtures(predict.load_artifact(path), fx)
    assert np.allclose(a[["lam_home", "lam_away", "p_home"]], b[["lam_home", "lam_away", "p_home"]])


def test_stronger_home_team_is_favoured(artifact: dict) -> None:
    res = predict.predict_fixtures(artifact, pd.DataFrame({"home": ["Palmeiras", "Palmeiras"],
                                                           "away": ["Parana", "Palmeiras"]}))
    assert res.iloc[0].p_home > res.iloc[0].p_away
