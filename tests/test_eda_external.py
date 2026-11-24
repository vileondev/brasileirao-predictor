"""Tests for the external-table audit (skipped when data/external is absent: it is git-ignored)."""
from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import config, eda_external

pytestmark = pytest.mark.skipif(
    not (config.EXTERNAL_STATS_FILE.exists() and config.EXTERNAL_MATCHES_FILE.exists()),
    reason="external tables not downloaded (data/external is git-ignored)")


def _hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("eda_ext")
    before = _hashes(config.EXTERNAL_DIR)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = eda_external.run(tmp / "eda_external.md", tmp / "figures")
    result["unchanged"] = before == _hashes(config.EXTERNAL_DIR)
    return result


def test_report_and_figures(out: dict) -> None:
    assert out["report"].exists() and out["report"].stat().st_size > 5000
    assert len(out["figures"]) == 4
    for p in out["figures"].values():
        assert p.exists() and p.stat().st_size > 10_000
        assert round(Image.open(p).info.get("dpi", (0, 0))[0]) == config.FIG_DPI
    text = out["report"].read_text(encoding="utf-8")
    assert not re.search(r"\|\s*-?(nan|inf)\s*\|", text, flags=re.IGNORECASE)
    for h in ("## Veredito", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."):
        assert h in text


def test_external_files_are_never_modified(out: dict) -> None:
    assert out["unchanged"]


def test_known_defects_are_detected(out: dict) -> None:
    s = out["sections"]
    checks = s["checks"].set_index("Verificacao")
    assert checks.loc["Linhas duplicadas (temporada, mandante, visitante, data)", "Status"] == "ATENCAO"
    assert checks.loc["id_match unico", "Status"] == "ATENCAO"
    rules = s["rules"].set_index("Regra")
    shots = rules.loc["Chutes fora > chutes (se 'chutes' fosse o total, nunca ocorreria)"]
    assert shots.Violacoes > 200 and shots["Linhas avaliaveis"] == 753
    assert rules.loc["Gols do 1o tempo > gols finais", "Violacoes"] == 1
    assert (rules.Violacoes >= 0).all() and (rules["Linhas avaliaveis"] > 0).all()


def test_scores_agree_with_our_base(out: dict) -> None:
    cs = out["sections"]["crosscheck"]
    assert cs["score_diff"] == 0 and cs["paired"] > 3300
    mc = out["sections"]["matches"]
    assert len(mc["diff"]) == 3 and mc["bad_format"] == 0 and mc["paired"] > 5600


def test_missingness_is_structural_by_season(out: dict) -> None:
    mech = out["sections"]["missing"]["mechanism"].set_index("Coluna")
    for col in ("valor_equipe_titular_man", "escanteios_man", "publico"):
        assert mech.loc[col, "p Season"] < config.MISSING_ALPHA_STRICT
        assert mech.loc[col, "Mecanismo"].startswith("MAR")


def test_univariate_identities_and_values(out: dict) -> None:
    u = out["sections"]["univariate"]
    core = u[["N", "Mean", "Median", "Min", "Max", "Range", "Variance", "SD", "Q1", "Q3", "IQR"]]
    assert np.isfinite(core.to_numpy(dtype=float)).all()
    assert np.allclose(u["IQR"], u["Q3"] - u["Q1"]) and np.allclose(u["Range"], u["Max"] - u["Min"])
    ages = u[u.Variavel.str.startswith("Idade")]
    assert ages["Mean"].between(25, 30).all()


def test_value_signal_and_normalisation(out: dict) -> None:
    sig = out["sections"]["signal"]
    assert 0.2 < sig["corr_raw"] < 0.45 and 0.2 < sig["corr_norm"] < 0.45
    vs = out["sections"]["valor_season"]
    # nominal medians jump between seasons (e.g. 2014 -> 2015), so the season level must not be ignored
    assert vs["Mediana (M)"].max() / vs["Mediana (M)"].min() > 1.5
