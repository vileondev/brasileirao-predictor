"""EDA module tests: outputs exist, tables are consistent, raw data is never touched."""
from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import config, crosscheck, eda


def _hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("eda")
    before = _hashes(config.DATA_RAW)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the audit must run without any warning
        result = eda.run(tmp / "eda_report.md", tmp / "figures")
    result["raw_unchanged"] = before == _hashes(config.DATA_RAW)
    return result


def test_report_and_four_figures_generated(out: dict) -> None:
    assert out["report"].exists() and out["report"].stat().st_size > 5000
    names = {p.name for p in out["figures"].values()}
    assert names == {"missing_values_heatmap.png", "goals_distribution_poisson_fit.png",
                     "team_goals_boxplot.png", "team_goals_comparison_bar.png"}
    for p in out["figures"].values():
        assert p.exists() and p.stat().st_size > 10_000
        dpi = Image.open(p).info.get("dpi", (0, 0))
        assert round(dpi[0]) == config.FIG_DPI


def test_report_has_no_nan_or_inf(out: dict) -> None:
    text = out["report"].read_text(encoding="utf-8")
    assert not re.search(r"\|\s*-?(nan|inf)\s*\|", text, flags=re.IGNORECASE)  # table cells only
    assert not re.search(r"=\s*-?(nan|inf)\b", text, flags=re.IGNORECASE)
    for h in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert h in text


def test_summary_numbers_are_finite(out: dict) -> None:
    for key in ("league", "team_scored", "team_conceded"):
        df = out[key]
        core = df[["N", "Mean", "Median", "Min", "Max", "Range", "Variance", "SD", "Q1", "Q3", "IQR"]]
        assert np.isfinite(core.to_numpy(dtype=float)).all(), key


def test_iqr_and_range_identities(out: dict) -> None:
    for key in ("league", "team_scored", "team_conceded"):
        df = out[key]
        assert np.allclose(df["IQR"], df["Q3"] - df["Q1"], atol=1e-12)
        assert np.allclose(df["Range"], df["Max"] - df["Min"], atol=1e-12)


def test_implied_probabilities_sum_to_one(out: dict) -> None:
    p = out["implied"]
    assert len(p) > 5000 and np.allclose(p.sum(axis=1), 1.0, atol=1e-6) and (p > 0).all().all()


def test_integrity_checks_and_champions(out: dict) -> None:
    checks = out["checks"]
    unexpected = checks[checks.Status == "ATENCAO"]["Verificacao"].tolist()
    # known source defects in the (unused) Max* odds columns are the only accepted warnings
    assert all("Odds" in v for v in unexpected), unexpected
    assert out["champions"]["Confere"].dropna().all()


def test_raw_data_is_never_modified(out: dict) -> None:
    assert out["raw_unchanged"]


def test_describe_known_values() -> None:
    d = eda.describe([0, 1, 1, 2, 4])
    assert d["Mean"] == pytest.approx(1.6) and d["Median"] == 1 and d["Mode"] == 1
    assert d["Variance"] == pytest.approx(2.3) and d["SD"] == pytest.approx(np.sqrt(2.3))
    assert d["Q1"] == 1 and d["Q3"] == 2 and d["IQR"] == 1 and d["Range"] == 4
    assert d["VMR"] == pytest.approx(2.3 / 1.6) and d["Poisson_skew"] == pytest.approx(1 / np.sqrt(1.6))
    assert np.isnan(eda.describe([0, 0, 0, 0])["VMR"])  # undefined, shown as n/d


def test_dispersion_test_detects_overdispersion() -> None:
    rng = np.random.default_rng(0)
    pois = rng.poisson(1.4, 5000)
    nb = rng.negative_binomial(2, 2 / (2 + 1.4), 5000)
    assert eda.dispersion_test(pois)[1] > 0.01
    assert eda.dispersion_test(nb)[0] > 1.3 and eda.dispersion_test(nb)[1] < 1e-6


def test_crosscheck_compare_offline() -> None:
    ours = pd.DataFrame({"season": [2022, 2022], "home_team": ["A", "B"], "away_team": ["B", "A"],
                         "kickoff": pd.to_datetime(["2022-05-01 20:00", "2022-05-08 20:00"]),
                         "home_goals": [1, 2], "away_goals": [0, 2]})
    api = pd.DataFrame({"season": [2022, 2022], "home": ["A", "B"], "away": ["B", "A"],
                        "api_home_goals": [1, 2], "api_away_goals": [0, 1],
                        "api_date": pd.to_datetime(["2022-05-01 20:00", "2022-05-08 20:00"])})
    res = crosscheck.compare(ours, api)
    assert res["matched"] == 2 and len(res["score_mismatch"]) == 1 and res["date_off"] == 0
    assert "Placares divergentes: 1" in crosscheck.build_report(res)
