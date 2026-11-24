"""Data-quality audit and exploratory data analysis of the raw Brasileirão CSV.

Run with ``python -m src.eda``. Produces ``reports/eda_report.md`` and four
300-DPI figures in ``reports/figures/``. The module is strictly read-only on
``data/raw/``.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src import baselines, config, ingestion

ODDS_PREFIXES: tuple[str, ...] = ("PSC", "MaxC", "AvgC", "BFEC", "B365C")
MAX_OVER_AVG_RATIO: float = 1.5  # best odds above 1.5x the market average are implausible

#: raw column -> (standardized name, domain type, definition, modeling role)
DATA_DICTIONARY: dict[str, tuple[str, str, str, str]] = {
    "Country": ("(dropped)", "Nominal Categorical", "Country of the league (constant 'Brazil')", "Metadata/Drop"),
    "League": ("(dropped)", "Nominal Categorical", "League name (constant 'Serie A')", "Metadata/Drop"),
    "Season": ("season", "Ordinal Categorical", "Season label; 2020 runs Aug/2020-Feb/2021 (COVID)", "Feature Source"),
    "Date": ("kickoff (date)", "Timestamp", "Match date, dd/mm/yyyy (same clock as Time)", "Feature Source"),
    "Time": ("kickoff (time)", "Timestamp", "Kickoff time HH:MM; evidence points to UTC (see 1.3)", "Feature Source"),
    "Home": ("home_team", "Nominal Categorical", "Home team (source spelling, mapped to canonical)", "Feature Source"),
    "Away": ("away_team", "Nominal Categorical", "Away team (source spelling, mapped to canonical)", "Feature Source"),
    "HG": ("home_goals", "Discrete Numeric", "Full-time goals scored by the home team", "Target"),
    "AG": ("away_goals", "Discrete Numeric", "Full-time goals scored by the away team", "Target"),
    "Res": ("result", "Nominal Categorical", "Full-time result: H (home), D (draw), A (away)", "Target"),
}
_ODDS_DEFS = {
    "PSC": "Pinnacle closing decimal odds", "MaxC": "Highest closing odds across bookmakers",
    "AvgC": "Market-average closing odds", "BFEC": "Betfair Exchange closing odds",
    "B365C": "Bet365 closing odds",
}
_ODDS_ROLE = {"PSC": "Benchmark Only", "MaxC": "Benchmark Only", "AvgC": "Benchmark Only",
              "BFEC": "Metadata/Drop", "B365C": "Metadata/Drop"}
_SIDE = {"H": "home win", "D": "draw", "A": "away win"}


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #

def _fmt(v: object, decimals: int = 2) -> str:
    """Format a cell: exact integers without decimals, floats fixed, undefined as ``n/d``."""
    if v is None:
        return "n/d"
    if isinstance(v, (bool, np.bool_)):
        return "sim" if v else "nao"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.{decimals}f}" if np.isfinite(v) else "n/d"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v) if str(v) != "nan" else "n/d"


def md_table(df: pd.DataFrame, decimals: dict[str, int] | int = 2) -> str:
    """Render a DataFrame as a Markdown table (``decimals`` per column or global)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for i in range(len(df)):
        cells = []
        for c in cols:
            d = decimals.get(c, 2) if isinstance(decimals, dict) else decimals
            cells.append(_fmt(df[c].iloc[i], d).replace("|", "\\|"))  # per-cell: keeps integer dtypes intact
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


# --------------------------------------------------------------------------- #
# descriptive statistics
# --------------------------------------------------------------------------- #

def describe(x: pd.Series | np.ndarray) -> dict[str, float]:
    """Full univariate profile (sample variance/SD with ``ddof=1``, linear-interpolated quartiles)."""
    s = pd.Series(np.asarray(x, dtype=float)).dropna()
    n = len(s)
    mean, med = s.mean(), s.median()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    sd = s.std(ddof=1) if n > 1 else np.nan
    var = s.var(ddof=1) if n > 1 else np.nan
    iqr = q3 - q1
    g1 = float(stats.skew(s, bias=False)) if n > 2 and sd > 0 else np.nan
    return {
        "N": n, "Mean": mean, "Median": med, "Mode": s.round(config.MODE_DECIMALS).mode().iloc[0],
        "Min": s.min(), "Max": s.max(), "Range": s.max() - s.min(), "Variance": var, "SD": sd,
        "Q1": q1, "Q3": q3, "IQR": iqr, "Skew_g1": g1,
        "Bowley": (q3 + q1 - 2 * med) / iqr if iqr > 0 else np.nan,
        "Poisson_skew": 1 / np.sqrt(mean) if mean > 0 else np.nan,
        "CV_pct": sd / mean * 100 if mean != 0 else np.nan,
        "IQR_over_median_pct": iqr / med * 100 if med != 0 else np.nan,
        "VMR": var / mean if mean > 0 else np.nan,
    }


def skew_label(g1: float) -> str:
    """Direction and magnitude of Fisher-Pearson skewness."""
    if not np.isfinite(g1):
        return "n/d"
    mag = ("simetrica" if abs(g1) < config.SKEW_SYMMETRIC
           else "moderada" if abs(g1) < config.SKEW_STRONG else "forte")
    return mag if mag == "simetrica" else f"{mag} ({'direita' if g1 > 0 else 'esquerda'})"


def dispersion_test(x: np.ndarray) -> tuple[float, float]:
    """Two-sided chi-square test of equidispersion: ``(n-1)*VMR ~ chi2(n-1)`` under Poisson."""
    x = np.asarray(x, dtype=float)
    n, mean = len(x), x.mean()
    vmr = x.var(ddof=1) / mean
    cdf = stats.chi2.cdf((n - 1) * vmr, n - 1)
    return float(vmr), float(2 * min(cdf, 1 - cdf))


def poisson_gof(x: np.ndarray) -> dict[str, float]:
    """Pearson chi-square goodness of fit against Poisson(mean), pooling counts >= POISSON_TAIL_POOL."""
    x = np.asarray(x)
    kmax, n, lam = config.POISSON_TAIL_POOL, len(x), float(np.mean(x))
    obs = np.array([(x == k).sum() for k in range(kmax)] + [(x >= kmax).sum()], dtype=float)
    p = stats.poisson.pmf(np.arange(kmax), lam)
    exp = n * np.append(p, 1 - p.sum())
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = len(obs) - 2
    return {"chi2": chi2, "dof": dof, "p_value": float(stats.chi2.sf(chi2, dof))}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw CSV untouched and the validated/cleaned table (no file is written)."""
    raw = pd.read_csv(config.RAW_MATCHES_FILE, encoding="utf-8-sig")
    clean = ingestion.build_interim(config.RAW_MATCHES_FILE)
    return raw, clean


def _named(raw: pd.DataFrame) -> list[str]:
    return [c for c in raw.columns if not str(c).startswith("Unnamed")]


def implied_probs(df: pd.DataFrame, prefix: str = "PSC") -> pd.DataFrame:
    """Vig-removed implied probabilities (rows with all three odds present)."""
    cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
    sub = df.dropna(subset=cols)
    p = baselines.odds_probs(sub, prefix)
    return pd.DataFrame(p, index=sub.index, columns=["p_home", "p_draw", "p_away"])


# --------------------------------------------------------------------------- #
# step 1 - schema, volumetry, dictionary
# --------------------------------------------------------------------------- #

def _odds_meta(col: str) -> tuple[str, str, str, str] | None:
    for pre in ODDS_PREFIXES:
        if col.startswith(pre) and col[len(pre):] in _SIDE:
            side = col[len(pre):]
            return (col if pre in ("PSC", "MaxC", "AvgC") else "(dropped)", "Continuous Numeric",
                    f"{_ODDS_DEFS[pre]}, {_SIDE[side]}", _ODDS_ROLE[pre])
    return None


def schema_table(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Data dictionary with raw/clean dtypes, domain type, definition and modeling role."""
    rows = []
    for col in _named(raw):
        meta = DATA_DICTIONARY.get(col) or _odds_meta(col)
        std, dtype_dom, definition, role = meta if meta else (col, "Unknown", "", "Metadata/Drop")
        clean_name = {"Season": "season", "Home": "home_team", "Away": "away_team", "HG": "home_goals",
                      "AG": "away_goals", "Res": "result", "Date": "kickoff", "Time": "kickoff"}.get(col, col)
        rows.append({"Coluna original": col, "Nome padronizado": std,
                     "Dtype bruto": str(raw[col].dtype),
                     "Dtype limpo": str(clean[clean_name].dtype) if clean_name in clean.columns else "—",
                     "Tipo de dominio": dtype_dom, "Definicao": definition, "Papel": role})
    return pd.DataFrame(rows)


def datetime_audit(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, object]:
    """Date/time parsing facts and timezone evidence."""
    t = raw["Time"].astype(str)
    dates = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    hours = pd.to_datetime(t.str.strip(), format="%H:%M", errors="coerce").dt.hour
    night = float(((hours >= 0) & (hours < 6)).mean())
    return {
        "time_missing": int(raw["Time"].isna().sum()),
        "time_whitespace": int((t != t.str.strip()).sum()),
        "date_unparsed": int(dates.isna().sum()),
        "time_unparsed": int(hours.isna().sum()),
        "share_00_to_06": night,
        "hour_counts": hours.value_counts().sort_index(),
        "date_min": dates.min(), "date_max": dates.max(),
        "raw_sorted": bool(pd.to_datetime(raw["Date"], format="%d/%m/%Y").is_monotonic_increasing),
        "clean_sorted": bool(clean["kickoff"].is_monotonic_increasing),
    }


# --------------------------------------------------------------------------- #
# step 2 - integrity
# --------------------------------------------------------------------------- #

def _team_long(clean: pd.DataFrame) -> pd.DataFrame:
    h = clean.rename(columns={"home_team": "team", "home_goals": "gf", "away_goals": "ga"})[["season", "team", "gf", "ga"]]
    a = clean.rename(columns={"away_team": "team", "away_goals": "gf", "home_goals": "ga"})[["season", "team", "gf", "ga"]]
    h, a = h.assign(venue="H"), a.assign(venue="A")
    long = pd.concat([h, a], ignore_index=True)
    long["pts"] = np.where(long["gf"] > long["ga"], 3, np.where(long["gf"] == long["ga"], 1, 0))
    return long


def integrity_checks(raw: pd.DataFrame, clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Domain/structure checks and the champion cross-check against the public record."""
    checks: list[tuple[str, str, str]] = []

    def add(name: str, ok: bool, detail: str, info: bool = False) -> None:
        checks.append((name, "INFO" if info else ("OK" if ok else "ATENCAO"), detail))

    named = raw[_named(raw)]
    dup = int(named.duplicated(["Season", "Date", "Home", "Away"]).sum())
    add("Duplicatas exatas (Season, Date, Home, Away)", dup == 0, f"{dup} duplicatas")
    names = sorted(set(raw["Home"]) | set(raw["Away"]))
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(_norm(n), []).append(n)
    drift = {k: v for k, v in groups.items() if len(v) > 1}
    add("Deriva de nomes de times (variantes que normalizam igual)", not drift,
        f"{len(names)} nomes distintos, {len(drift)} colisoes" + (f": {drift}" if drift else ""))
    add("Todos os nomes mapeados em TEAM_MAP", set(names) <= set(config.TEAM_MAP),
        f"{len(set(names) - set(config.TEAM_MAP))} sem mapa")
    played = raw.dropna(subset=["HG", "AG"])
    add("Gols negativos", int((played[["HG", "AG"]] < 0).sum().sum()) == 0,
        f"{int((played[['HG', 'AG']] < 0).sum().sum())} ocorrencias")
    add("Gols nao inteiros", bool((played[["HG", "AG"]] % 1 == 0).all().all()),
        f"max mandante={played.HG.max():.0f}, max visitante={played.AG.max():.0f}")
    odds_cols = [c for c in raw.columns if any(c.startswith(p) for p in ODDS_PREFIXES) and c in raw]
    odds = raw[odds_cols]
    low_cells = [(c, i) for c in odds_cols for i in raw.index[raw[c] <= 1.0]]
    where = "; ".join(f"{c}={raw.at[i, c]:.2f} em {raw.at[i, 'Date']} {raw.at[i, 'Home']} x {raw.at[i, 'Away']}"
                      for c, i in low_cells)
    valid = odds.where(odds > 1.0)
    add("Odds invalidas (<= 1.0)", not low_cells,
        f"{len(low_cells)} ocorrencia(s)" + (f": {where}" if where else "")
        + f"; faixa valida {np.nanmin(valid.to_numpy()):.2f} a {np.nanmax(valid.to_numpy()):.2f}")
    mx, av = ["MaxCH", "MaxCD", "MaxCA"], ["AvgCH", "AvgCD", "AvgCA"]
    below = raw.index[(raw[mx].to_numpy() < raw[av].to_numpy()).any(axis=1)]
    ratio = (raw[mx].to_numpy() / raw[av].to_numpy())
    spread = raw.index[(ratio > MAX_OVER_AVG_RATIO).any(axis=1)]
    bad_rows = sorted(set(below) | set(spread))
    desc_rows = "; ".join(f"{raw.at[i, 'Date']} {raw.at[i, 'Home']} x {raw.at[i, 'Away']}" for i in bad_rows)
    add("Odds maximas coerentes com a media (Max >= Avg e Max/Avg <= 1.5)", not bad_rows,
        f"{len(bad_rows)} jogo(s)" + (f": {desc_rows} (Max* atipico: o 0.00 e erro certo, os demais podem ser precos extremos de uma casa; o projeto nao usa Max*)" if bad_rows else ""))
    inv = implied_probs(raw, "PSC")
    ov = (1 / raw.loc[inv.index, ["PSCH", "PSCD", "PSCA"]]).sum(axis=1)
    lo, hi = config.OVERROUND_RANGE
    bad = int(((ov < lo) | (ov > hi)).sum())
    add("Overround Pinnacle dentro de [1.00, 1.20]", bad == 0,
        f"{bad} fora; media={ov.mean():.3f}, min={ov.min():.3f}, max={ov.max():.3f}")
    unplayed = raw[raw[["HG", "AG", "Res"]].isna().any(axis=1)]
    desc = "; ".join(f"{r.Date} {r.Home} x {r.Away}" for r in unplayed.itertuples())
    add("Jogos sem placar (nao disputados)", True, f"{len(unplayed)} linha(s): {desc} — removida(s) na ingestao", info=True)
    exp = pd.Series("D", index=played.index)
    exp[played.HG > played.AG], exp[played.HG < played.AG] = "H", "A"
    mism = int((exp != played["Res"]).sum())
    add("Consistencia Res x (HG, AG)", mism == 0, f"{mism} inconsistencias em {len(played)} jogos")
    cnt = clean.groupby("season").apply(lambda g: len(set(g.home_team) | set(g.away_team)), include_groups=False)
    complete = clean.groupby("season").size()
    full = complete[complete >= config.TEAMS_PER_SEASON * (config.TEAMS_PER_SEASON - 1) - 1].index
    add("20 times por temporada", bool((cnt == config.TEAMS_PER_SEASON).all()), f"min={cnt.min()}, max={cnt.max()}")
    long = _team_long(clean)
    gpt = long[long.season.isin(full)].groupby(["season", "team"]).size()
    short = gpt[gpt != config.GAMES_PER_TEAM_SEASON]
    explained = {(int(r.Season), config.TEAM_MAP.get(t)) for r in unplayed.itertuples() for t in (r.Home, r.Away)}
    unexplained = [(s, t) for (s, t) in short.index if (int(s), t) not in explained]
    add("38 jogos por time em temporada completa", not unexplained,
        "todos com 38" if short.empty else "; ".join(f"{s}/{t}={n}" for (s, t), n in short.items())
        + (" — explicado pelo jogo nao disputado" if not unexplained else ""),
        info=bool(len(short)) and not unexplained)
    pair_dup = int(clean.duplicated(["season", "home_team", "away_team"]).sum())
    add("Cada par (mandante, visitante) aparece uma vez por temporada", pair_dup == 0, f"{pair_dup} repetidos")
    add("Ordenacao cronologica", bool(clean["kickoff"].is_monotonic_increasing),
        "arquivo bruto " + ("ordenado" if bool(pd.to_datetime(raw.Date, format='%d/%m/%Y').is_monotonic_increasing) else "fora de ordem") + "; tabela limpa ordenada")

    champs = []
    for s, g in long.groupby("season"):
        pts = g.groupby("team")["pts"].sum().sort_values(ascending=False)
        known = config.KNOWN_CHAMPIONS.get(int(s))
        champs.append({"Temporada": int(s), "Jogos": int(len(clean[clean.season == s])),
                       "Lider nos dados": pts.index[0], "Pontos": int(pts.iloc[0]),
                       "Campeao oficial": known[0] if known else "(em andamento)",
                       "Pontos oficiais": known[1] if known and known[1] is not None else None,
                       "Confere": (pts.index[0] == known[0] and (known[1] is None or int(pts.iloc[0]) == known[1])) if known else None})
    ch = pd.DataFrame(champs)
    ch["Pontos oficiais"] = pd.Series([int(v) if pd.notna(v) else None for v in ch["Pontos oficiais"]], dtype=object)
    ch["Confere"] = pd.Series([None if pd.isna(v) else bool(v) for v in ch["Confere"]], dtype=object)
    ok = ch["Confere"].dropna()
    add("Campeoes e pontos conferem com o registro publico", bool(ok.all()), f"{int(ok.sum())} de {len(ok)} temporadas")
    return pd.DataFrame(checks, columns=["Verificacao", "Status", "Detalhe"]), ch


# --------------------------------------------------------------------------- #
# step 3 - missing data
# --------------------------------------------------------------------------- #

def missing_tables(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Missingness per column/season and mechanism diagnostics (Rubin taxonomy)."""
    cols = _named(raw)
    n = len(raw)
    per_col = pd.DataFrame({"Coluna": cols, "Ausentes": [int(raw[c].isna().sum()) for c in cols]})
    per_col["Pct"] = per_col["Ausentes"] / n * 100
    by_season = raw.groupby("Season")[cols].apply(lambda g: g.isna().mean() * 100)
    by_season = by_season.loc[:, by_season.max() > 0]

    played = raw.dropna(subset=["HG", "AG", "Res"]).copy()
    played["tg"] = played.HG + played.AG
    long = _team_long(clean)
    ppg = long.groupby("team")["pts"].mean()
    tier = pd.qcut(ppg, 3, labels=["baixo", "medio", "alto"])
    played["tier_home"] = played["Home"].map(config.TEAM_MAP).map(tier)

    rows = []
    for c in per_col.loc[per_col.Ausentes > 0, "Coluna"]:
        if c in ("HG", "AG", "Res"):
            rows.append({"Coluna": c, "Ausentes": int(raw[c].isna().sum()), "p Season": np.nan, "p Resultado": np.nan,
                         "p Gols totais": np.nan, "p Nivel do mandante": np.nan,
                         "Mecanismo": "MNAR por construcao (jogo cancelado apos o acidente aereo da Chapecoense; "
                                      "ausencia depende do proprio evento) — estrutural, nao imputar"})
            continue
        ind = played[c].isna()
        k = int(ind.sum())

        def chi_p(by: pd.Series) -> float:
            tab = pd.crosstab(by, ind)
            tab = tab.loc[:, tab.sum() > 0]
            return float(stats.chi2_contingency(tab)[1]) if tab.shape[1] > 1 and tab.shape[0] > 1 else np.nan

        p_season, p_res, p_tier = chi_p(played["Season"]), chi_p(played["Res"]), chi_p(played["tier_home"])
        a_, b_ = played.loc[ind, "tg"], played.loc[~ind, "tg"]
        p_goals = float(stats.mannwhitneyu(a_, b_).pvalue) if min(len(a_), len(b_)) >= config.MIN_GROUP_FOR_TESTS else np.nan
        full_gap = int((by_season[c] >= 99.9).sum()) if c in by_season else 0
        if p_season < config.MISSING_ALPHA_STRICT:
            mech = f"MAR (depende de Season; {full_gap} temporada(s) 100% ausentes: cobertura da fonte)" if full_gap else "MAR (depende de Season)"
        elif min(x for x in (p_res, p_goals) if np.isfinite(x)) < config.MISSING_ALPHA:
            mech = "MAR/MNAR candidato (associado ao resultado); MNAR nao e distinguivel de MAR so com dados observados"
        else:
            mech = "MCAR nao rejeitado (sem associacao com Season, resultado, gols ou nivel do time)"
        rows.append({"Coluna": c, "Ausentes": k, "p Season": p_season, "p Resultado": p_res,
                     "p Gols totais": p_goals, "p Nivel do mandante": p_tier, "Mecanismo": mech})
    return {"per_col": per_col, "by_season": by_season, "mechanism": pd.DataFrame(rows)}


# --------------------------------------------------------------------------- #
# step 4 - outliers
# --------------------------------------------------------------------------- #

def tukey_z_summary(variables: dict[str, pd.Series]) -> pd.DataFrame:
    """Tukey (1.5 / 3.0 IQR), Z-score and modified Z-score (MAD) flags per variable."""
    rows = []
    for name, x in variables.items():
        x = x.dropna().astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - config.IQR_FENCE_STANDARD * iqr, q3 + config.IQR_FENCE_STANDARD * iqr
        lo_e, hi_e = q1 - config.IQR_FENCE_EXTREME * iqr, q3 + config.IQR_FENCE_EXTREME * iqr
        z = (x - x.mean()) / x.std(ddof=1)
        med = x.median()
        mad = float(np.median(np.abs(x - med)))
        scale = mad / config.MAD_NORMAL_CONSTANT if mad > 0 else config.MEAN_AD_NORMAL_CONSTANT * float(np.mean(np.abs(x - med)))
        mz = (x - med) / scale if scale > 0 else pd.Series(0.0, index=x.index)
        rows.append({"Variavel": name, "N": len(x), "Cerca inf.": lo, "Cerca sup.": hi,
                     "Tukey 1.5": int(((x < lo) | (x > hi)).sum()), "Tukey 3.0": int(((x < lo_e) | (x > hi_e)).sum()),
                     "|Z|>3": int((z.abs() > config.Z_THRESHOLD).sum()), "MAD": mad,
                     "|Z mod|>3.5": int((mz.abs() > config.MODIFIED_Z_THRESHOLD).sum())})
    return pd.DataFrame(rows)


def _verdict(r: pd.Series) -> str:
    lo, hi = config.OVERROUND_RANGE
    problems = []
    if max(r.home_goals, r.away_goals) > config.MAX_PLAUSIBLE_TEAM_GOALS or r.home_goals + r.away_goals > config.MAX_PLAUSIBLE_TOTAL_GOALS:
        problems.append("placar implausivel")
    if not lo <= r.overround <= hi:
        problems.append("overround fora da faixa")
    exp = "H" if r.home_goals > r.away_goals else "A" if r.home_goals < r.away_goals else "D"
    if exp != r.result:
        problems.append("Res inconsistente")
    return "Suspeito: " + ", ".join(problems) if problems else "Evento genuino de cauda (reter)"


def anomaly_table(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Isolation Forest + LOF on (HG, AG, p_home, p_away); returns the top extreme matches."""
    p = implied_probs(clean, "PSC")
    sub = clean.loc[p.index].join(p)
    sub["overround"] = (1 / sub[["PSCH", "PSCD", "PSCA"]]).sum(axis=1)
    X = StandardScaler().fit_transform(sub[["home_goals", "away_goals", "p_home", "p_away"]])
    iso = IsolationForest(n_estimators=config.ISOLATION_TREES, contamination=config.ISOLATION_CONTAMINATION,
                          random_state=config.SEED).fit(X)
    sub["if_score"] = -iso.score_samples(X)
    lof = LocalOutlierFactor(n_neighbors=config.LOF_NEIGHBORS, contamination=config.ISOLATION_CONTAMINATION)
    flags = lof.fit_predict(X)
    sub["lof_score"] = -lof.negative_outlier_factor_
    k = int((iso.predict(X) == -1).sum())
    top_if = set(sub.nlargest(k, "if_score").index)
    top_lof = set(sub.nlargest(k, "lof_score").index)
    real_p = np.where(sub.result == "H", sub.p_home, np.where(sub.result == "A", sub.p_away, 1 - sub.p_home - sub.p_away))
    sub["p_outcome"] = real_p
    top = sub.nlargest(config.TOP_EXTREME_MATCHES, "if_score")
    out = pd.DataFrame({
        "Data": top.kickoff.dt.strftime("%Y-%m-%d"), "Jogo": top.home_team + " x " + top.away_team,
        "Placar": top.home_goals.astype(int).astype(str) + "-" + top.away_goals.astype(int).astype(str),
        "P casa": top.p_home, "P empate": 1 - top.p_home - top.p_away, "P fora": top.p_away,
        "P do resultado": top.p_outcome, "Score IF": top.if_score, "Veredito": top.apply(_verdict, axis=1)})
    info = {"flagged_if": k, "overlap_lof": len(top_if & top_lof), "lof_flagged": int((flags == -1).sum())}
    sub["total"] = sub.home_goals + sub.away_goals
    sub["margin"] = (sub.home_goals - sub.away_goals).abs()
    picks = pd.concat([
        sub.nlargest(4, ["total", "margin"]).assign(Motivo="mais gols no jogo"),
        sub.nlargest(4, ["margin", "total"]).assign(Motivo="maior diferenca de gols"),
        sub.nsmallest(4, "p_outcome").assign(Motivo="maior zebra (menor prob. do resultado)"),
    ]).drop_duplicates(subset="match_id")
    info["extremes"] = pd.DataFrame({
        "Motivo": picks.Motivo, "Data": picks.kickoff.dt.strftime("%Y-%m-%d"),
        "Jogo": picks.home_team + " x " + picks.away_team,
        "Placar": picks.home_goals.astype(int).astype(str) + "-" + picks.away_goals.astype(int).astype(str),
        "P do resultado": picks.p_outcome, "Veredito": picks.apply(_verdict, axis=1)}).reset_index(drop=True)
    return out.reset_index(drop=True), info


# --------------------------------------------------------------------------- #
# step 5 - descriptive statistics
# --------------------------------------------------------------------------- #

def league_stats(clean: pd.DataFrame) -> pd.DataFrame:
    """League-wide profile of goals, goal difference and Pinnacle implied probabilities."""
    p = implied_probs(clean, "PSC")
    vars_ = {"HG (gols mandante)": clean.home_goals, "AG (gols visitante)": clean.away_goals,
             "Gols totais": clean.home_goals + clean.away_goals, "Saldo (HG-AG)": clean.home_goals - clean.away_goals,
             "P implicita casa": p.p_home, "P implicita empate": p.p_draw, "P implicita fora": p.p_away}
    rows = []
    for name, x in vars_.items():
        d = describe(x)
        rows.append({"Variavel": name, **d, "Assimetria": skew_label(d["Skew_g1"])})
    return pd.DataFrame(rows)


def team_stats(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-team profile of goals scored/conceded (teams with >= MIN_SEASONS_PER_TEAM seasons)."""
    long = _team_long(clean)
    seasons = long.groupby("team")["season"].nunique()
    teams = seasons[seasons >= config.MIN_SEASONS_PER_TEAM].index
    scored, conceded = [], []
    for t in sorted(teams):
        g = long[long.team == t]
        for tgt, col in ((scored, "gf"), (conceded, "ga")):
            d = describe(g[col])
            tgt.append({"Time": t, "Temporadas": int(seasons[t]), **d, "Assimetria": skew_label(d["Skew_g1"])})
    sc, cd = pd.DataFrame(scored), pd.DataFrame(conceded)
    season_means = long[long.team.isin(teams)].groupby(["team", "season"])[["gf", "ga"]].mean()
    spread = season_means.groupby("team").std(ddof=1).rename(columns={"gf": "SD_entre_temporadas_gf", "ga": "SD_entre_temporadas_ga"})
    return sc, cd, spread.reset_index().rename(columns={"team": "Time"})


def model_justification(clean: pd.DataFrame) -> dict[str, object]:
    """Evidence for the Poisson assumption, temporal decay and home-advantage handling."""
    hg, ag = clean.home_goals.to_numpy(), clean.away_goals.to_numpy()
    out: dict[str, object] = {"gof_home": poisson_gof(hg), "gof_away": poisson_gof(ag)}
    out["vmr_home"], out["vmr_p_home"] = dispersion_test(hg)
    out["vmr_away"], out["vmr_p_away"] = dispersion_test(ag)
    lam = baselines.dc_oos_lambdas(clean)
    m = clean.merge(lam, on="match_id")
    out["cond_vmr_home"] = float(((m.home_goals - m.dc_lh) ** 2).mean() / m.dc_lh.mean())
    out["cond_vmr_away"] = float(((m.away_goals - m.dc_la) ** 2).mean() / m.dc_la.mean())

    gd = clean.assign(d=clean.home_goals - clean.away_goals).groupby("season")["d"]
    adv = pd.DataFrame({"Temporada": gd.mean().index.astype(int), "Jogos": gd.size().to_numpy(),
                        "Vantagem (HG-AG)": gd.mean().to_numpy(),
                        "IC95 +/-": (1.96 * gd.std(ddof=1) / np.sqrt(gd.size())).to_numpy()})
    out["home_adv"] = adv
    lr = stats.linregress(adv["Temporada"], adv["Vantagem (HG-AG)"])
    out["home_adv_slope"], out["home_adv_p"] = float(lr.slope), float(lr.pvalue)

    long = _team_long(clean)
    n_games = long.groupby(["team", "season"]).size()
    full = n_games[n_games >= config.GAMES_PER_TEAM_SEASON - 1].index
    sm = long.set_index(["team", "season"]).loc[full].groupby(level=[0, 1])[["gf", "ga"]].mean()
    pairs = sm.reset_index().sort_values(["team", "season"])
    for col in ("gf", "ga"):
        pairs[f"{col}_prev"] = pairs.groupby("team")[col].shift(1)
        pairs[f"{col}_prev_season"] = pairs.groupby("team")["season"].shift(1)
    pairs = pairs[pairs["season"] - pairs["gf_prev_season"] == 1].dropna()
    for col in ("gf", "ga"):
        r = stats.pearsonr(pairs[col], pairs[f"{col}_prev"])
        out[f"persist_{col}"], out[f"persist_{col}_p"], out["persist_n"] = float(r[0]), float(r[1]), len(pairs)
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)


def fig_missing(by_season: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * by_season.shape[1] + 3), 6))
    im = ax.imshow(by_season.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_xticks(range(by_season.shape[1]), by_season.columns, rotation=60, ha="right")
    ax.set_yticks(range(by_season.shape[0]), by_season.index.astype(int))
    for i in range(by_season.shape[0]):
        for j in range(by_season.shape[1]):
            v = by_season.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white" if v > 55 else "black")
    ax.set_title("Valores ausentes por temporada e coluna (somente colunas com alguma ausencia)")
    ax.set_xlabel("Coluna do arquivo bruto")
    ax.set_ylabel("Temporada")
    fig.colorbar(im, ax=ax, label="% de jogos com valor ausente")
    _save(fig, path)


def fig_poisson(clean: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, title in ((axes[0], "home_goals", "Gols do mandante (HG)"), (axes[1], "away_goals", "Gols do visitante (AG)")):
        x = clean[col].to_numpy()
        lam, vmr = x.mean(), x.var(ddof=1) / x.mean()
        ks = np.arange(0, int(x.max()) + 1)
        ax.hist(x, bins=np.arange(-0.5, ks.max() + 1.5), density=True, color="#9ecae1", edgecolor="white", label="Empirico (histograma)")
        grid = np.linspace(-0.5, ks.max() + 0.5, 300)
        ax.plot(grid, stats.gaussian_kde(x, bw_method=0.5)(grid), color="#08519c", lw=1.8, label="KDE (largura 0,5)")
        ax.plot(ks, stats.poisson.pmf(ks, lam), "o-", color="#d94801", lw=1.2, label=f"Poisson(media={lam:.2f})")
        ax.set_title(f"{title}: N={len(x)}, VMR={vmr:.3f}")
        ax.set_xlabel("Gols no jogo (unidades)")
        ax.set_ylabel("Densidade / probabilidade")
        ax.legend()
    _save(fig, path)


def fig_boxplot(clean: pd.DataFrame, path: Path) -> list[str]:
    long = _team_long(clean)
    teams = long.groupby("team").size().sort_values(ascending=False).head(config.BOXPLOT_TEAMS).index.tolist()
    data = [long.loc[long.team == t, "gf"].to_numpy() for t in teams]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.boxplot(data, showmeans=True, whis=config.IQR_FENCE_STANDARD,
               meanprops={"marker": "D", "markerfacecolor": "#d94801", "markeredgecolor": "black", "markersize": 7},
               flierprops={"marker": "o", "markerfacecolor": "none", "markeredgecolor": "#636363", "markersize": 5},
               medianprops={"color": "#08519c", "lw": 2})
    top = max(d.max() for d in data)
    for i, d in enumerate(data, start=1):
        s = pd.Series(d)
        mode = s.mode().iloc[0]
        ax.plot(i, mode, "s", color="#31a354", markersize=7, zorder=5)
        ax.text(i, top + 0.7, f"Q1={s.quantile(.25):.2f}\nMed={s.median():.2f}\nMedia={s.mean():.2f}\nModa={mode:.0f}\nQ3={s.quantile(.75):.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(1, len(teams) + 1), teams, rotation=20)
    ax.set_ylim(-0.5, top + 3.2)
    ax.set_ylabel("Gols marcados por jogo (gols)")
    ax.set_xlabel("Time (os 8 com mais jogos na base)")
    ax.set_title("Gols marcados por jogo: caixa Q1-Q3, bigodes 1,5xIQR, outliers e medidas de centro")
    ax.plot([], [], "D", color="#d94801", label="Media")
    ax.plot([], [], "s", color="#31a354", label="Moda")
    ax.plot([], [], "-", color="#08519c", lw=2, label="Mediana")
    ax.plot([], [], "o", mfc="none", mec="#636363", label="Outlier (fora de 1,5xIQR)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    _save(fig, path)
    return teams


def fig_team_bars(sc: pd.DataFrame, cd: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, 0.28 * len(sc))), sharey=False)
    for ax, df, title, color in ((axes[0], sc, "Gols marcados por jogo", "#3182bd"), (axes[1], cd, "Gols sofridos por jogo", "#e6550d")):
        d = df.sort_values("Mean")
        mu, sd = d["Mean"].mean(), d["Mean"].std(ddof=1)
        z = (d["Mean"] - mu) / sd
        y = np.arange(len(d))
        ax.axvspan(mu - sd, mu + sd, color="#bdbdbd", alpha=0.35, label="Media das medias +/- 1 DP")
        ax.axvline(mu, color="black", lw=1, label=f"Media entre times ({mu:.2f})")
        ax.barh(y, d["Mean"], xerr=d["SD"], color=color, alpha=0.85, error_kw={"lw": 0.6, "ecolor": "#252525"})
        for yi, (m_, zi) in enumerate(zip(d["Mean"], z)):
            ax.text(m_ + d["SD"].iloc[yi] + 0.03, yi, f"z={zi:+.2f}", va="center", fontsize=6)
        ax.set_yticks(y, d["Time"], fontsize=7)
        ax.set_xlabel("Gols por jogo (media +/- 1 desvio-padrao do jogo)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Times com 3 ou mais temporadas: faixa cinza = +/-1 DP entre as medias dos times; z = escore-z da media")
    _save(fig, path)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def _stat_cols(df: pd.DataFrame, first: str) -> pd.DataFrame:
    cols = [first, "N", "Mean", "Median", "Mode", "Min", "Max", "Range", "Variance", "SD", "Q1", "Q3", "IQR"]
    return df[cols]


def build_report(raw: pd.DataFrame, clean: pd.DataFrame, figs: dict[str, Path], boxplot_teams: list[str],
                 sections: dict[str, object]) -> str:
    s = sections
    checks, champs = s["checks"], s["champions"]
    dt, mt, ls = s["datetime"], s["missing"], s["league"]
    sc, cd, spread = s["team_scored"], s["team_conceded"], s["team_spread"]
    mj, an, an_info, ts = s["justification"], s["anomalies"], s["anomaly_info"], s["tukey"]
    n_ok = int((checks.Status == "OK").sum())
    named = _named(raw)
    empty_cols = len(raw.columns) - len(named)
    f2 = lambda x: _fmt(x, 2)
    f3 = lambda x: _fmt(x, 3)
    champ_ok = champs["Confere"].dropna()

    L: list[str] = []
    L.append("# Relatorio de EDA e auditoria de qualidade dos dados\n")
    L.append(f"Fonte: `{config.RAW_MATCHES_FILE.name}` (football-data.co.uk). Gerado por `python -m src.eda`; "
             "o modulo so le `data/raw/`. Contagens exatas aparecem como inteiros; estatisticas com 2 casas e "
             "assimetria/CV/probabilidades com 3 casas. `n/d` = indefinido (ex.: divisao por zero).\n")
    L.append("## Veredito\n")
    attn = checks[checks.Status == "ATENCAO"]["Verificacao"].tolist()
    L.append(f"- **A base e real e consistente nos campos que o projeto usa:** {n_ok} de {len(checks)} verificacoes OK "
             f"({int((checks.Status == 'INFO').sum())} informativas); campeao e pontos batem com o registro publico em "
             f"{int(champ_ok.sum())} de {len(champ_ok)} temporadas; sem duplicatas, gols negativos nem `Res` inconsistente.")
    if attn:
        L.append("- **Atencao (sem impacto no modelo):** " + "; ".join(attn)
                 + ". Afetam so a coluna `MaxCA` (um valor 0.00 em Cruzeiro x Fortaleza, 27/10/2019) e precos maximos atipicos em `Max*`, "
                 "colunas que o projeto nao usa; placares, times e odds Pinnacle/media estao integros.")
    L.append(f"- **Volume:** {len(raw)} linhas brutas ({len(clean)} jogos disputados), {len(raw.columns)} colunas "
             f"({len(named)} com dados, {empty_cols} vazias), {raw.Season.nunique()} temporadas, de {dt['date_min']:%Y-%m-%d} a {dt['date_max']:%Y-%m-%d}.")
    L.append("- **Faltantes:** so Pinnacle (poucos jogos), Bet365/Betfair (cobertura parcial) e o jogo nao disputado de 2016; "
             "placares, times, datas e horas estao completos.")
    L.append(f"- **Poisson e adequada, sem sobredispersao:** razao variancia/media do mandante {f3(mj['vmr_home'])} (p={f3(mj['vmr_p_home'])}) "
             f"e do visitante {f3(mj['vmr_away'])} (p={f3(mj['vmr_p_away'])}); condicional aos times (λ fora da amostra) "
             f"{f3(mj['cond_vmr_home'])} e {f3(mj['cond_vmr_away'])}. O qui-quadrado de ajuste rejeita Poisson para HG "
             f"(p={f3(mj['gof_home']['p_value'])}, por leve *sub*dispersao) e nao para AG (p={f3(mj['gof_away']['p_value'])}); "
             "como a variancia nunca excede a media, a Binomial Negativa nao se justifica.")
    L.append("- **Limite:** esta auditoria verifica consistencia interna e confere campeoes; a conferencia contra uma segunda fonte "
             "independente esta em `reports/crosscheck.md`.\n")

    L.append("## 1. Esquema, volumetria e dicionario de dados\n")
    L.append(f"- Linhas: **{len(raw)}** brutas, **{len(clean)}** apos a ingestao. Colunas: **{len(raw.columns)}** brutas "
             f"(**{len(named)}** com nome, {empty_cols} colunas finais vazias do CSV), {clean.shape[1]} apos a limpeza.")
    L.append(f"- Cobertura temporal: {dt['date_min']:%Y-%m-%d} a {dt['date_max']:%Y-%m-%d}; temporadas {raw.Season.min()}-{raw.Season.max()}.")
    L.append("- A limpeza descarta `Country`, `League`, as colunas Bet365/Betfair e o jogo sem placar; padroniza times, "
             "junta `Date`+`Time` em `kickoff` e ordena cronologicamente.\n")
    L.append("### 1.1 Dicionario de dados\n")
    L.append(md_table(s["schema"]) + "\n")
    L.append("### 1.2 Regras de parsing de data e hora\n")
    L.append(f"- `Date` no formato dd/mm/yyyy: {len(raw) - dt['date_unparsed']} de {len(raw)} validas. `Time` HH:MM: "
             f"{len(raw) - dt['time_unparsed']} de {len(raw)} validas; **{dt['time_missing']} ausentes**.")
    L.append(f"- **{dt['time_whitespace']} valores de `Time` com espaco a esquerda** (ex.: `\" 22:30\"`, jogos de 30/10 a 01/11/2019); "
             "tratados com `strip` antes do parsing. Nenhuma linha legada sem `Time`.")
    L.append(f"- Arquivo bruto {'ja em ordem cronologica' if dt['raw_sorted'] else 'fora de ordem'}; tabela limpa ordenada por `kickoff` e `home_team`.\n")
    L.append("### 1.3 Fuso horario\n")
    L.append(f"A fonte nao documenta o fuso. **{f3(dt['share_00_to_06'])}** dos horarios caem entre 00:00 e 05:59 — impossivel como horario local "
             "do Brasil — e os mais comuns sao 22:30 e 00:30. Isso indica **UTC** (jogos da noite de Brasilia aparecem ja no dia seguinte). "
             "O horario so e usado para ordenar partidas dentro do mesmo dia, portanto o fuso nao afeta o modelo.\n")
    hc = dt["hour_counts"].rename_axis("Hora").reset_index(name="Jogos")
    L.append(md_table(hc.sort_values("Jogos", ascending=False).head(8)) + "\n")

    L.append("## 2. Integridade, valores especiais e entidades\n")
    L.append(md_table(checks) + "\n")
    L.append("### 2.1 Conferencia de campeoes por temporada\n")
    L.append(md_table(champs) + "\n")
    L.append("O jogo Chapecoense x Atletico-MG de 11/12/2016 esta na fonte **sem placar** (nao foi disputado apos o acidente aereo de 28/11/2016). "
             "Nao e um WO duplo registrado: nao ha resultado a atribuir. A ingestao o remove, e ambos os times terminam 2016 com 37 jogos.\n")

    L.append("## 3. Dados faltantes e mecanismo\n")
    L.append(f"Colunas finais vazias do CSV: **{empty_cols}** (100% ausentes, sem nome — ignoradas).\n")
    L.append(md_table(mt["per_col"], {"Pct": 2}) + "\n")
    L.append("![Ausencias](figures/missing_values_heatmap.png)\n")
    L.append("### 3.1 Classificacao do mecanismo (taxonomia de Rubin)\n")
    L.append("Testes qui-quadrado (ausencia x Season, x resultado, x nivel do mandante por pontos/jogo em tercis) e Mann-Whitney "
             "(gols totais). Valores-p com 3 casas; limiar de 0,05 (0,01 para Season). "
             "Com os dados observados nao se prova MNAR: o teste so distingue associacao com variaveis observadas.\n")
    L.append(md_table(mt["mechanism"], 3) + "\n")

    L.append("## 4. Anomalias e outliers\n")
    L.append(md_table(ts, {"Cerca inf.": 3, "Cerca sup.": 3, "MAD": 3}) + "\n")
    t = ts.set_index("Variavel")
    L.append("**Por que o Z-score distorce em contagens assimetricas:** media e desvio-padrao sao puxados pela cauda longa de uma "
             "variavel discreta e limitada em zero, e o desvio-padrao de uma contagem pequena e sensivel a poucos valores altos. "
             "O MAD (mediana dos desvios absolutos) nao sofre isso, mas em contagens com MAD pequeno qualquer valor acima da mediana "
             "pode ficar longe em escala robusta; se o MAD fosse 0 usariamos o desvio medio absoluto como escala. "
             f"Nos dados: em HG, |Z|>3 marca {t.loc['HG', '|Z|>3']} jogos, Tukey 1.5 marca {t.loc['HG', 'Tukey 1.5']} e o Z modificado "
             f"{t.loc['HG', '|Z mod|>3.5']}; em AG, {t.loc['AG', '|Z|>3']}, {t.loc['AG', 'Tukey 1.5']} e {t.loc['AG', '|Z mod|>3.5']}. "
             "Os criterios discordam muito porque medem coisas diferentes numa distribuicao discreta assimetrica; "
             "por isso a decisao final depende de checagens de dominio (secao 4.1), e nao de um unico limiar.\n")
    L.append(f"### 4.1 Os {config.TOP_EXTREME_MATCHES} jogos mais extremos (Isolation Forest sobre HG, AG, P casa, P fora)\n")
    L.append(f"Isolation Forest sinalizou {an_info['flagged_if']} jogos (contaminacao {config.ISOLATION_CONTAMINATION:.2f}); "
             f"o LOF concorda em {an_info['overlap_lof']} deles.\n")
    L.append(md_table(an, {"P casa": 3, "P empate": 3, "P fora": 3, "P do resultado": 3, "Score IF": 3}) + "\n")
    L.append("### 4.2 Placares extremos e maiores zebras\n")
    L.append(md_table(an_info["extremes"], {"P do resultado": 3}) + "\n")
    L.append("Veredito automatico: *genuino* se o placar e plausivel (<=10 gols por time, <=12 no total), o overround esta em [1.00, 1.20] "
             "e `Res` e consistente; caso contrario *suspeito*. Nenhuma linha foi alterada ou removida por ser outlier: caudas assim "
             "sao esperadas em dados de gols.\n")

    L.append("## 5. Estatisticas univariadas\n")
    L.append("### 5.1 Liga inteira\n")
    L.append(md_table(_stat_cols(ls, "Variavel"), {"N": 0}) + "\n")
    L.append(md_table(ls[["Variavel", "Skew_g1", "Bowley", "Poisson_skew", "Assimetria", "CV_pct", "IQR_over_median_pct", "VMR"]]
                      .rename(columns={"Skew_g1": "g1", "Poisson_skew": "Assimetria Poisson", "CV_pct": "CV %",
                                       "IQR_over_median_pct": "IQR/Mediana %"}), 3) + "\n")
    L.append("### 5.2 Teste de equidispersao e ajuste Poisson\n")
    gh, ga = mj["gof_home"], mj["gof_away"]
    L.append(md_table(pd.DataFrame([
        {"Variavel": "HG", "VMR": mj["vmr_home"], "p (dispersao)": mj["vmr_p_home"], "chi2 Poisson": gh["chi2"], "gl": gh["dof"], "p (ajuste)": gh["p_value"], "VMR condicional": mj["cond_vmr_home"]},
        {"Variavel": "AG", "VMR": mj["vmr_away"], "p (dispersao)": mj["vmr_p_away"], "chi2 Poisson": ga["chi2"], "gl": ga["dof"], "p (ajuste)": ga["p_value"], "VMR condicional": mj["cond_vmr_away"]},
    ]), 3) + "\n")
    L.append("O VMR bruto mistura times fortes e fracos (λ diferente por jogo), o que o inflaria; o **VMR condicional** usa o λ de cada jogo "
             "previsto fora da amostra pelo Dixon-Coles e mede a dispersao em torno do que o modelo espera. Valores abaixo de 1 indicam que a "
             "variancia nao excede a de Poisson; **nao ha base para a Binomial Negativa**.\n")
    L.append("![Distribuicao](figures/goals_distribution_poisson_fit.png)\n")
    L.append("### 5.3 Por time (>= 3 temporadas)\n")
    L.append("**Gols marcados por jogo**\n")
    L.append(md_table(_stat_cols(sc, "Time"), {"N": 0}) + "\n")
    L.append(md_table(sc[["Time", "Skew_g1", "Bowley", "Assimetria", "CV_pct", "VMR"]].rename(columns={"Skew_g1": "g1", "CV_pct": "CV %"}), 3) + "\n")
    L.append("**Gols sofridos por jogo**\n")
    L.append(md_table(_stat_cols(cd, "Time"), {"N": 0}) + "\n")
    L.append(md_table(cd[["Time", "Skew_g1", "Bowley", "Assimetria", "CV_pct", "VMR"]].rename(columns={"Skew_g1": "g1", "CV_pct": "CV %"}), 3) + "\n")
    L.append("![Boxplot](figures/team_goals_boxplot.png)\n")
    L.append("![Comparacao](figures/team_goals_comparison_bar.png)\n")
    L.append("### 5.4 Regularidade entre temporadas\n")
    both = sc[["Time", "CV_pct"]].merge(cd[["Time", "CV_pct"]], on="Time", suffixes=(" marcados", " sofridos")).merge(spread, on="Time")
    L.append("Mais regulares e mais volateis (CV dos gols por jogo; desvio entre as medias de cada temporada):\n")
    for col, label in (("CV_pct marcados", "marcar"), ("CV_pct sofridos", "sofrer")):
        a = both.nsmallest(3, col)["Time"].tolist()
        b = both.nlargest(3, col)["Time"].tolist()
        L.append(f"- {label.capitalize()} gols — mais regulares: {', '.join(a)}; mais volateis: {', '.join(b)}.")
    L.append("")
    L.append(md_table(both.rename(columns={"SD_entre_temporadas_gf": "DP entre temporadas (marc.)",
                                           "SD_entre_temporadas_ga": "DP entre temporadas (sofr.)",
                                           "CV_pct marcados": "CV % marcados", "CV_pct sofridos": "CV % sofridos"}), 3) + "\n")

    L.append("## 6. O que a base justifica para a modelagem\n")
    ha = mj["home_adv"]
    L.append(f"- **Poisson:** ver 5.2. Sem sobredispersao condicional; manter Poisson com correcao de Dixon-Coles.")
    L.append(f"- **Decaimento temporal:** a correlacao entre a media de gols de um time numa temporada e na anterior e r={f3(mj['persist_gf'])} "
             f"(marcados, p={f3(mj['persist_gf_p'])}) e r={f3(mj['persist_ga'])} (sofridos, p={f3(mj['persist_ga_p'])}), n={mj['persist_n']} pares. "
             "Persistencia relevante mas bem abaixo de 1: a forca muda de uma temporada para outra, o que justifica pesar mais os jogos recentes.")
    L.append(f"- **Vantagem de mandante:** media de {f3(ha['Vantagem (HG-AG)'].mean())} gols por jogo; tendencia por temporada "
             f"{f3(mj['home_adv_slope'])} gol/ano (p={f3(mj['home_adv_p'])}).\n")
    L.append(md_table(ha, {"Jogos": 0, "Vantagem (HG-AG)": 3, "IC95 +/-": 3}) + "\n")
    L.append("## 7. Limitacoes\n")
    L.append("- Sem xG, chutes, escalacoes ou lesoes; a base so permite modelos de resultados e odds.\n"
             "- Odds de fechamento da Pinnacle ausentes em parte dos jogos; Bet365/Betfair tem cobertura parcial e nao sao usadas.\n"
             "- Fuso horario inferido (UTC), nao documentado pela fonte.\n"
             "- A conferencia de campeoes usa o registro publico; a comparacao jogo a jogo com outra fonte esta em `reports/crosscheck.md`.\n")
    return "\n".join(L)


def run(report_path: Path = config.EDA_REPORT_FILE, figures_dir: Path = config.FIGURES_DIR) -> dict[str, object]:
    """Run the full audit and write the report and figures.

    Returns
    -------
    dict
        Summary tables (``league``, ``team_scored``, ``team_conceded``, ``implied``) and output paths.
    """
    raw, clean = load()
    schema = schema_table(raw, clean)
    datetime = datetime_audit(raw, clean)
    checks, champs = integrity_checks(raw, clean)
    missing = missing_tables(raw, clean)
    p = implied_probs(clean, "PSC")
    gols = {"HG": clean.home_goals, "AG": clean.away_goals, "Gols totais": clean.home_goals + clean.away_goals,
            "Saldo": clean.home_goals - clean.away_goals, "P implicita casa": p.p_home,
            "P implicita empate": p.p_draw, "P implicita fora": p.p_away}
    tukey = tukey_z_summary(gols)
    anomalies, an_info = anomaly_table(clean)
    league = league_stats(clean)
    sc, cd, spread = team_stats(clean)
    justification = model_justification(clean)

    figs = {"missing": figures_dir / "missing_values_heatmap.png",
            "poisson": figures_dir / "goals_distribution_poisson_fit.png",
            "boxplot": figures_dir / "team_goals_boxplot.png",
            "bars": figures_dir / "team_goals_comparison_bar.png"}
    fig_missing(missing["by_season"], figs["missing"])
    fig_poisson(clean, figs["poisson"])
    teams = fig_boxplot(clean, figs["boxplot"])
    fig_team_bars(sc, cd, figs["bars"])

    sections = {"schema": schema, "datetime": datetime, "checks": checks, "champions": champs, "missing": missing,
                "tukey": tukey, "anomalies": anomalies, "anomaly_info": an_info, "league": league,
                "team_scored": sc, "team_conceded": cd, "team_spread": spread, "justification": justification}
    report = build_report(raw, clean, figs, teams, sections)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {"league": league, "team_scored": sc, "team_conceded": cd, "implied": p, "checks": checks,
            "champions": champs, "report": report_path, "figures": figs, "sections": sections}


def main() -> None:
    out = run()
    print(f"Relatorio: {out['report']}")
    for name, path in out["figures"].items():
        print(f"Figura ({name}): {path}")
    checks = out["checks"]
    print(f"Verificacoes OK: {int((checks.Status == 'OK').sum())} de {len(checks)}")


if __name__ == "__main__":
    main()
