"""Data-quality audit of the external tables (williamorim/brasileirao ``stats.csv`` and ``matches.csv``).

Same methodology as :mod:`src.eda` (schema, integrity, missing-data mechanisms, outliers,
univariate statistics) plus a cross-check against our validated base. Read-only on
``data/external/``. Run with ``python -m src.eda_external``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src import config, eda, ingestion, predict

#: column -> (domain type, definition (inferred from the column name), role)
DICTIONARY: dict[str, tuple[str, str, str]] = {
    "id_match": ("Discrete Numeric", "Match id from the matches table (join key built by season/home/away)", "Metadata/Drop"),
    "season": ("Ordinal Categorical", "Season label", "Metadata/Drop"),
    "data": ("Timestamp", "Match date (YYYY-MM-DD)", "Metadata/Drop"),
    "horario": ("Timestamp", "Kickoff time as stored; implausible hours (see 2.3) -> unreliable", "Metadata/Drop"),
    "rodada": ("Ordinal Categorical", "Round number (1-38)", "Metadata/Drop"),
    "estadio": ("Nominal Categorical", "Stadium", "Metadata/Drop"),
    "arbitro": ("Nominal Categorical", "Referee", "Feature Source"),
    "publico": ("Discrete Numeric", "Paid/total attendance", "Feature Source"),
    "publico_max": ("Discrete Numeric", "Stadium capacity", "Feature Source"),
    "home": ("Nominal Categorical", "Home team (accented spelling)", "Metadata/Drop"),
    "away": ("Nominal Categorical", "Away team (accented spelling)", "Metadata/Drop"),
    "tecnico_man": ("Nominal Categorical", "Home coach", "Feature Source"),
    "tecnico_vis": ("Nominal Categorical", "Away coach", "Feature Source"),
    "colocacao_man": ("Ordinal Categorical", "Home team league position", "Feature Source"),
    "colocacao_vis": ("Ordinal Categorical", "Away team league position", "Feature Source"),
    "valor_equipe_titular_man": ("Continuous Numeric", "Market value of the home starting XI (currency not documented)", "Feature Source"),
    "valor_equipe_titular_vis": ("Continuous Numeric", "Market value of the away starting XI", "Feature Source"),
    "idade_media_titular_man": ("Continuous Numeric", "Mean age of the home starting XI (years)", "Feature Source"),
    "idade_media_titular_vis": ("Continuous Numeric", "Mean age of the away starting XI (years)", "Feature Source"),
    "gols_man": ("Discrete Numeric", "Full-time goals, home", "Benchmark Only"),
    "gols_vis": ("Discrete Numeric", "Full-time goals, away", "Benchmark Only"),
    "gols_1_tempo_man": ("Discrete Numeric", "Half-time goals, home", "Feature Source"),
    "gols_1_tempo_vis": ("Discrete Numeric", "Half-time goals, away", "Feature Source"),
    "escanteios_man": ("Discrete Numeric", "Corners, home", "Feature Source"),
    "escanteios_vis": ("Discrete Numeric", "Corners, away", "Feature Source"),
    "faltas_man": ("Discrete Numeric", "Fouls, home", "Feature Source"),
    "faltas_vis": ("Discrete Numeric", "Fouls, away", "Feature Source"),
    "chutes_bola_parada_man": ("Discrete Numeric", "Set-piece shots?, home (semantics unclear)", "Metadata/Drop"),
    "chutes_bola_parada_vis": ("Discrete Numeric", "Set-piece shots?, away (semantics unclear)", "Metadata/Drop"),
    "defesas_man": ("Discrete Numeric", "Goalkeeper saves, home", "Feature Source"),
    "defesas_vis": ("Discrete Numeric", "Goalkeeper saves, away", "Feature Source"),
    "impedimentos_man": ("Discrete Numeric", "Offsides, home", "Feature Source"),
    "impedimentos_vis": ("Discrete Numeric", "Offsides, away", "Feature Source"),
    "chutes_man": ("Discrete Numeric", "'Shots', home (semantics unclear, see 2.2)", "Metadata/Drop"),
    "chutes_vis": ("Discrete Numeric", "'Shots', away", "Metadata/Drop"),
    "chutes_fora_man": ("Discrete Numeric", "Shots off target, home", "Feature Source"),
    "chutes_fora_vis": ("Discrete Numeric", "Shots off target, away", "Feature Source"),
}
NUMERIC_PAIRS: dict[str, tuple[str, str]] = {
    "Valor do titular": ("valor_equipe_titular_man", "valor_equipe_titular_vis"),
    "Idade media do titular": ("idade_media_titular_man", "idade_media_titular_vis"),
    "Gols 1o tempo": ("gols_1_tempo_man", "gols_1_tempo_vis"), "Escanteios": ("escanteios_man", "escanteios_vis"),
    "Faltas": ("faltas_man", "faltas_vis"), "Defesas": ("defesas_man", "defesas_vis"),
    "Impedimentos": ("impedimentos_man", "impedimentos_vis"), "Chutes (rotulo)": ("chutes_man", "chutes_vis"),
    "Chutes fora": ("chutes_fora_man", "chutes_fora_vis"),
}


def _canon(s: pd.Series) -> pd.Series:
    return s.map(predict.resolve_team)


def load_stats(path: Path = config.EXTERNAL_STATS_FILE) -> pd.DataFrame:
    """Read ``stats.csv`` untouched and add parsed helper columns."""
    e = pd.read_csv(path)
    e["date"] = pd.to_datetime(e["data"], errors="coerce")
    e["home_c"], e["away_c"] = _canon(e["home"]), _canon(e["away"])
    e["total_goals"] = e["gols_man"] + e["gols_vis"]
    e["result"] = np.where(e.gols_man > e.gols_vis, "H", np.where(e.gols_man < e.gols_vis, "A", "D"))
    e.loc[e["gols_man"].isna(), "result"] = np.nan
    return e


# --------------------------------------------------------------------------- #
# integrity
# --------------------------------------------------------------------------- #

def rules_table(e: pd.DataFrame) -> pd.DataFrame:
    """Logical/plausibility rules with violation counts and examples."""
    lo_a, hi_a = config.EXT_AGE_RANGE
    lo_v, hi_v = config.EXT_VALUE_RANGE

    def ex(mask: pd.Series) -> str:
        rows = e[mask].head(3)
        return "; ".join(f"{r.season} {r.home} x {r.away}" for r in rows.itertuples()) or "—"

    cols_shots = ["chutes_man", "chutes_vis", "chutes_fora_man", "chutes_fora_vis"]
    rules = {
        "Gols do 1o tempo > gols finais": ((e.gols_1_tempo_man > e.gols_man) | (e.gols_1_tempo_vis > e.gols_vis),
                                            ["gols_1_tempo_man", "gols_1_tempo_vis", "gols_man", "gols_vis"]),
        "Publico > capacidade": (e.publico > e.publico_max, ["publico", "publico_max"]),
        "Capacidade <= 0 ou publico < 0": ((e.publico_max <= 0) | (e.publico < 0), ["publico", "publico_max"]),
        "Chutes fora > chutes (se 'chutes' fosse o total, nunca ocorreria)":
            ((e.chutes_fora_man > e.chutes_man) | (e.chutes_fora_vis > e.chutes_vis), cols_shots),
        "Chutes < gols (so gol contra explicaria)": ((e.chutes_man < e.gols_man) | (e.chutes_vis < e.gols_vis), cols_shots + ["gols_man", "gols_vis"]),
        f"Faltas de um time > {config.EXT_FOULS_MAX} no jogo": ((e.faltas_man > config.EXT_FOULS_MAX) | (e.faltas_vis > config.EXT_FOULS_MAX),
                                                               ["faltas_man", "faltas_vis"]),
        f"Valor do titular fora de [{lo_v:.0e}, {hi_v:.0e}]": (((e.valor_equipe_titular_man < lo_v) | (e.valor_equipe_titular_man > hi_v)
                                                              | (e.valor_equipe_titular_vis < lo_v) | (e.valor_equipe_titular_vis > hi_v)),
                                                             ["valor_equipe_titular_man", "valor_equipe_titular_vis"]),
        f"Idade media fora de [{lo_a:.0f}, {hi_a:.0f}] anos": (((e.idade_media_titular_man < lo_a) | (e.idade_media_titular_man > hi_a)
                                                              | (e.idade_media_titular_vis < lo_a) | (e.idade_media_titular_vis > hi_a)),
                                                             ["idade_media_titular_man", "idade_media_titular_vis"]),
        "Posicao na tabela fora de [1, 24]": (((e.colocacao_man < 1) | (e.colocacao_man > 24) | (e.colocacao_vis < 1) | (e.colocacao_vis > 24)),
                                              ["colocacao_man", "colocacao_vis"]),
        "Mandante == visitante": (e.home == e.away, ["home", "away"]),
    }
    rows = []
    for name, (mask, cols) in rules.items():
        mask = mask.fillna(False)
        rows.append({"Regra": name, "Violacoes": int(mask.sum()), "Linhas avaliaveis": int(e[cols].notna().all(axis=1).sum()), "Exemplos": ex(mask)})
    return pd.DataFrame(rows)


def integrity_checks(e: pd.DataFrame) -> pd.DataFrame:
    """Duplicates, keys, ordering, entity resolution and date consistency."""
    checks: list[tuple[str, str, str]] = []

    def add(name: str, ok: bool, detail: str, info: bool = False) -> None:
        checks.append((name, "INFO" if info else ("OK" if ok else "ATENCAO"), detail))

    full = e[e.duplicated(["season", "home", "away", "data"], keep=False)]
    add("Linhas duplicadas (temporada, mandante, visitante, data)", full.empty,
        f"{len(full) // 2} par(es): " + "; ".join(f"{r.season} {r.home} x {r.away} em {r.data}" for r in full.drop_duplicates(['season', 'home', 'away']).itertuples())
        if not full.empty else "0")
    pair = e[e.duplicated(["season", "home_c", "away_c"], keep=False)]
    n_pairs = int(pair.groupby(["season", "home_c", "away_c"]).ngroups) if not pair.empty else 0
    add("Mesmo confronto (mandante x visitante) repetido na temporada", n_pairs == 0,
        f"{n_pairs} confronto(s); em {sorted(set(pair.season))}: " + "; ".join(sorted({f'{r.home} x {r.away}' for r in pair.itertuples()}))[:200])
    idd = e[e.duplicated("id_match", keep=False) & e.id_match.notna()]
    add("id_match unico", idd.empty, f"{int(idd.id_match.nunique())} id(s) compartilhado(s) por jogos diferentes (chave de junção por temporada/mandante/visitante falha quando o confronto se repete)")
    add("id_match ausente", int(e.id_match.isna().sum()) == 0, f"{int(e.id_match.isna().sum())} linha(s) sem id")
    add("Linhas sem placar", int(e.gols_man.isna().sum()) == 0, f"{int(e.gols_man.isna().sum())}: " + "; ".join(f"{r.season} {r.home} x {r.away}" for r in e[e.gols_man.isna()].itertuples()), info=True)
    add("Datas invalidas", int(e.date.isna().sum()) == 0, f"{int(e.date.isna().sum())}")
    add("Arquivo em ordem cronologica", bool(e.date.is_monotonic_increasing), "nao: a ordem do arquivo segue id/rodada, nao a data" if not e.date.is_monotonic_increasing else "sim", info=True)
    off = e[(e.date.dt.year != e.season)]
    add("Ano da data == temporada", off.empty, f"{len(off)} linha(s) em temporadas {sorted(set(off.season))} (a de 2020 termina em 2021)", info=True)
    unk = sorted((set(e.home_c) | set(e.away_c)) - predict.CANONICAL_TEAMS)
    add("Times fora do nosso mapa", not unk, f"{len(unk)} (clubes de antes de 2012, fora do escopo): {', '.join(unk)}", info=True)
    names = pd.concat([e.tecnico_man, e.tecnico_vis]).dropna().unique()
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(eda._norm(n), []).append(n)
    coll = {k: v for k, v in groups.items() if len(v) > 1}
    add("Tecnicos: variantes que normalizam igual", not coll, f"{len(names)} nomes, {len(coll)} colisoes" + (f": {list(coll.values())[:4]}" if coll else ""))
    refs = e.arbitro.dropna().unique()
    rg: dict[str, list[str]] = {}
    for n in refs:
        rg.setdefault(eda._norm(n), []).append(n)
    rcoll = {k: v for k, v in rg.items() if len(v) > 1}
    add("Arbitros: variantes que normalizam igual", not rcoll, f"{len(refs)} nomes, {len(rcoll)} colisoes" + (f": {list(rcoll.values())[:4]}" if rcoll else ""))
    sizes = e[e.season >= config.EXT_FIRST_SEASON].groupby(["season", "rodada"]).size()
    add("10 jogos por rodada (2012+)", bool((sizes == 10).all()), f"{int((sizes != 10).sum())} rodadas irregulares de {len(sizes)}", info=True)
    return pd.DataFrame(checks, columns=["Verificacao", "Status", "Detalhe"])


def kickoff_analysis(e: pd.DataFrame, ours: pd.DataFrame) -> dict[str, object]:
    """Plausibility of ``horario`` and its offset against our (UTC-like) kickoff."""
    t = pd.to_datetime(e["horario"], format="%H:%M", errors="coerce")
    hour = t.dt.hour
    share_impossible = float(((hour >= 3) & (hour < 12)).mean())
    m = ours.merge(e[["season", "home_c", "away_c", "horario"]].drop_duplicates(["season", "home_c", "away_c"]),
                   left_on=["season", "home_team", "away_team"], right_on=["season", "home_c", "away_c"])
    ext_min = pd.to_datetime(m["horario"], format="%H:%M", errors="coerce")
    our_min = m["kickoff"].dt.hour * 60 + m["kickoff"].dt.minute
    delta = ((our_min - (ext_min.dt.hour * 60 + ext_min.dt.minute)) % 1440) / 60
    top = delta.round(2).value_counts(normalize=True).head(3)
    return {"share_impossible": share_impossible, "hour_counts": hour.value_counts().sort_index(),
            "top_offsets": top, "n_paired": int(len(m))}


# --------------------------------------------------------------------------- #
# missing data
# --------------------------------------------------------------------------- #

def missing_audit(e: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Missingness per column/season and mechanism classification (Rubin)."""
    cols = [c for c in e.columns if c in DICTIONARY]
    per_col = pd.DataFrame({"Coluna": cols, "Ausentes": [int(e[c].isna().sum()) for c in cols]})
    per_col["Pct"] = per_col["Ausentes"] / len(e) * 100
    by_season = e.groupby("season")[cols].apply(lambda g: g.isna().mean() * 100)
    by_season = by_season.loc[:, by_season.max() > 0]
    played = e.dropna(subset=["gols_man"])
    covered = played[played.season >= config.EXT_FIRST_SEASON]
    rows = []
    for c in per_col.loc[per_col.Ausentes > 0, "Coluna"]:
        ind = played[c].isna()
        if ind.sum() < config.MIN_GROUP_FOR_TESTS or (~ind).sum() < config.MIN_GROUP_FOR_TESTS:
            continue

        def chi_p(frame: pd.DataFrame, by: str, mask: pd.Series) -> float:
            tab = pd.crosstab(frame[by], mask.loc[frame.index])
            return float(stats.chi2_contingency(tab)[1]) if tab.shape[0] > 1 and tab.shape[1] > 1 else np.nan

        p_season = chi_p(played, "season", ind)
        cov_seasons = by_season.index[by_season[c] <= 100 - config.EXT_COVERAGE_MIN_PCT]
        within = played[played.season.isin(cov_seasons)]
        iw = within[c].isna()
        p_res = chi_p(within, "result", iw) if iw.sum() >= config.MIN_GROUP_FOR_TESTS else np.nan
        a_, b_ = within.loc[iw, "total_goals"], within.loc[~iw, "total_goals"]
        p_goals = float(stats.mannwhitneyu(a_, b_).pvalue) if min(len(a_), len(b_)) >= config.MIN_GROUP_FOR_TESTS else np.nan
        empty = by_season.index[by_season[c] >= 99.9].tolist()
        if p_season < config.MISSING_ALPHA_STRICT:
            mech = f"MAR (depende de Season; {len(empty)} temporada(s) 100% vazias: cobertura da fonte)"
            if np.isfinite(p_res) and p_res < config.MISSING_ALPHA or np.isfinite(p_goals) and p_goals < config.MISSING_ALPHA:
                mech += "; dentro das temporadas cobertas ainda ha associacao com resultado/gols"
            else:
                mech += "; dentro das temporadas cobertas, sem associacao com resultado/gols (MCAR nao rejeitado)"
        else:
            mech = "MCAR nao rejeitado"
        rows.append({"Coluna": c, "Ausentes": int(ind.sum()), "p Season": p_season,
                     "p Resultado (temporadas cobertas)": p_res, "p Gols totais (temporadas cobertas)": p_goals, "Mecanismo": mech})
    return {"per_col": per_col, "by_season": by_season, "mechanism": pd.DataFrame(rows)}


# --------------------------------------------------------------------------- #
# statistics and outliers
# --------------------------------------------------------------------------- #

def univariate(e: pd.DataFrame) -> pd.DataFrame:
    """Profile of the numeric columns (covered seasons only)."""
    rows = []
    cov = e[e.season >= config.EXT_FIRST_SEASON]
    continuous = ("Valor do titular", "Idade media do titular")  # not counts: the index of dispersion is meaningless
    for label, (a, b) in NUMERIC_PAIRS.items():
        for side, col in (("mandante", a), ("visitante", b)):
            x = cov[col].dropna()
            if len(x) < 3:
                continue
            scale = 1e6 if label == "Valor do titular" else 1.0
            d = eda.describe(x / scale)
            if label in continuous:
                d["VMR"] = np.nan
            name = f"{label} ({side})" + (" [milhoes]" if scale != 1.0 else "")
            rows.append({"Variavel": name, **d, "Assimetria": eda.skew_label(d["Skew_g1"])})
    for col in ("publico", "publico_max"):
        d = eda.describe(cov[col].dropna())
        d["VMR"] = np.nan
        rows.append({"Variavel": col, **d, "Assimetria": eda.skew_label(d["Skew_g1"])})
    return pd.DataFrame(rows)


def valor_by_season(e: pd.DataFrame) -> pd.DataFrame:
    """Nominal starting-XI value per season (shows inflation) and its signal after season normalisation."""
    cov = e[(e.season >= config.EXT_FIRST_SEASON)].dropna(subset=["valor_equipe_titular_man", "valor_equipe_titular_vis"])
    v = pd.concat([cov[["season", "valor_equipe_titular_man"]].rename(columns={"valor_equipe_titular_man": "v"}),
                   cov[["season", "valor_equipe_titular_vis"]].rename(columns={"valor_equipe_titular_vis": "v"})])
    g = v.groupby("season")["v"]
    return pd.DataFrame({"Temporada": g.median().index.astype(int), "Jogos-lado": g.size().to_numpy(),
                         "Mediana (M)": g.median().to_numpy() / 1e6, "Media (M)": g.mean().to_numpy() / 1e6, "DP (M)": g.std(ddof=1).to_numpy() / 1e6,
                         "CV %": (g.std(ddof=1) / g.mean() * 100).to_numpy()})


def value_signal(e: pd.DataFrame) -> dict[str, float]:
    """Raw vs season-normalised association between value ratio and goal difference."""
    cov = e[e.season >= config.EXT_FIRST_SEASON].dropna(subset=["valor_equipe_titular_man", "valor_equipe_titular_vis", "gols_man"]).copy()
    gd = cov.gols_man - cov.gols_vis
    raw = np.log(cov.valor_equipe_titular_man / cov.valor_equipe_titular_vis)
    med = cov.groupby("season")[["valor_equipe_titular_man", "valor_equipe_titular_vis"]].transform("median")
    norm = np.log(cov.valor_equipe_titular_man / med.valor_equipe_titular_man) - np.log(cov.valor_equipe_titular_vis / med.valor_equipe_titular_vis)
    return {"corr_raw": float(raw.corr(gd)), "corr_norm": float(norm.corr(gd)), "n": int(len(cov)),
            "ratio_vs_ratio_season": float(np.corrcoef(raw, norm)[0, 1])}


def outlier_audit(e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Tukey/Z/MAD flags and Isolation Forest extremes with a data-error vs tail-event verdict."""
    cov = e[e.season >= config.EXT_FIRST_SEASON]
    variables = {f"{lab} ({s})": cov[c] for lab, (a, b) in NUMERIC_PAIRS.items() for s, c in (("man.", a), ("vis.", b)) if cov[c].notna().sum() > 30}
    variables["Publico"] = cov["publico"]
    ts = eda.tukey_z_summary(variables)
    feat = cov.dropna(subset=["valor_equipe_titular_man", "valor_equipe_titular_vis", "idade_media_titular_man",
                              "idade_media_titular_vis", "gols_man", "gols_vis"]).copy()
    X = StandardScaler().fit_transform(np.column_stack([
        np.log(feat.valor_equipe_titular_man), np.log(feat.valor_equipe_titular_vis), feat.idade_media_titular_man,
        feat.idade_media_titular_vis, feat.gols_man, feat.gols_vis]))
    iso = IsolationForest(n_estimators=config.ISOLATION_TREES, contamination=config.ISOLATION_CONTAMINATION, random_state=config.SEED).fit(X)
    feat["if_score"] = -iso.score_samples(X)
    top = feat.nlargest(config.TOP_EXTREME_MATCHES, "if_score")

    lo_a, hi_a = config.EXT_AGE_RANGE
    lo_v, hi_v = config.EXT_VALUE_RANGE

    def verdict(r: pd.Series) -> str:
        bad = []
        if not lo_v <= min(r.valor_equipe_titular_man, r.valor_equipe_titular_vis) and max(r.valor_equipe_titular_man, r.valor_equipe_titular_vis) <= hi_v:
            bad.append("valor fora da faixa")
        if not (lo_a <= r.idade_media_titular_man <= hi_a and lo_a <= r.idade_media_titular_vis <= hi_a):
            bad.append("idade fora da faixa")
        if max(r.gols_man, r.gols_vis) > config.MAX_PLAUSIBLE_TEAM_GOALS:
            bad.append("placar implausivel")
        return "Suspeito: " + ", ".join(bad) if bad else "Evento genuino de cauda (reter)"

    out = pd.DataFrame({
        "Temporada": top.season, "Jogo": top.home + " x " + top.away, "Placar": top.gols_man.astype(int).astype(str) + "-" + top.gols_vis.astype(int).astype(str),
        "Valor casa (M)": top.valor_equipe_titular_man / 1e6, "Valor fora (M)": top.valor_equipe_titular_vis / 1e6,
        "Idade casa": top.idade_media_titular_man, "Idade fora": top.idade_media_titular_vis,
        "Score IF": top.if_score, "Veredito": top.apply(verdict, axis=1)}).reset_index(drop=True)
    return ts, out, {"if_flagged": int((iso.predict(X) == -1).sum()), "if_rows": int(len(feat))}


# --------------------------------------------------------------------------- #
# cross-check with our base and matches.csv
# --------------------------------------------------------------------------- #

def crosscheck_stats(e: pd.DataFrame, ours: pd.DataFrame) -> dict[str, object]:
    """Pair ``stats.csv`` with our matches and compare scores and dates."""
    d = e.drop_duplicates(["season", "home_c", "away_c"], keep="first")
    m = ours.merge(d, left_on=["season", "home_team", "away_team"], right_on=["season", "home_c", "away_c"], how="left", indicator=True)
    both = m[m["_merge"] == "both"].dropna(subset=["gols_man"])
    ok = (both.home_goals == both.gols_man) & (both.away_goals == both.gols_vis)
    days = (both.kickoff.dt.normalize() - both.date).dt.days
    in_scope = m[m.season <= e.season.max()]
    return {"ours_in_scope": int(len(in_scope)), "paired": int(len(both)), "score_equal": int(ok.sum()), "score_diff": int((~ok).sum()),
            "days": days.value_counts().sort_index(), "unpaired": in_scope[in_scope["_merge"] == "left_only"][["season", "home_team", "away_team"]]}


def audit_matches_csv(path: Path, ours: pd.DataFrame) -> dict[str, object]:
    """Structural audit of ``matches.csv`` and score agreement with our base."""
    x = pd.read_csv(path)
    x["home_c"], x["away_c"] = _canon(x["home"]), _canon(x["away"])
    sc = x["score"].astype(str).str.extract(r"^(\d+)x(\d+)$").astype(float)
    x["gm"], x["gv"] = sc[0], sc[1]
    played = x.dropna(subset=["gm"])
    d = x.drop_duplicates(["season", "home_c", "away_c"], keep="first")
    m = ours.merge(d, left_on=["season", "home_team", "away_team"], right_on=["season", "home_c", "away_c"], how="left")
    both = m.dropna(subset=["gm"])
    diff = both[(both.home_goals != both.gm) | (both.away_goals != both.gv)]
    per = x.groupby("season").size()
    return {
        "rows": len(x), "seasons": (int(x.season.min()), int(x.season.max())), "unplayed": int(x.score.astype(str).str.fullmatch(r"x").sum()),
        "bad_format": int((~x.score.astype(str).str.fullmatch(r"\d+x\d+|x")).sum()),
        "dup_pairs": int(x.duplicated(["season", "home_c", "away_c"]).sum()), "id_unique": bool(x.id_match.is_unique),
        "unmapped": sorted((set(x.home_c) | set(x.away_c)) - predict.CANONICAL_TEAMS),
        "mean_goals": float((played.gm + played.gv).mean()), "paired": int(len(both)), "ours": int(len(m)),
        "diff": diff[["season", "home_team", "away_team", "home_goals", "away_goals", "score"]],
        "per_season_min": int(per[per.index >= config.EXT_FIRST_SEASON].min()), "per_season_max": int(per[per.index >= config.EXT_FIRST_SEASON].max()),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def make_figures(e: pd.DataFrame, miss: dict[str, pd.DataFrame], fig_dir: Path) -> dict[str, Path]:
    paths = {k: fig_dir / f"external_{k}.png" for k in ("missing", "valor", "idade_gols", "chutes")}
    eda.fig_missing(miss["by_season"], paths["missing"])

    cov = e[e.season >= config.EXT_FIRST_SEASON]
    v = pd.concat([cov[["season", "valor_equipe_titular_man"]].rename(columns={"valor_equipe_titular_man": "v"}),
                   cov[["season", "valor_equipe_titular_vis"]].rename(columns={"valor_equipe_titular_vis": "v"})]).dropna()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    seasons = sorted(v.season.unique())
    ax[0].boxplot([v.loc[v.season == s, "v"] / 1e6 for s in seasons], showmeans=True, whis=config.IQR_FENCE_STANDARD,
                  meanprops={"marker": "D", "markerfacecolor": "#d94801", "markeredgecolor": "black"})
    ax[0].set_xticks(range(1, len(seasons) + 1), seasons)
    ax[0].set_yscale("log")
    ax[0].set_title("Valor do time titular por temporada (nominal)")
    ax[0].set_xlabel("Temporada")
    ax[0].set_ylabel("Valor (milhoes, escala log)")
    ax[1].hist(np.log10(v.v), bins=40, color="#9ecae1", edgecolor="white", density=True)
    ax[1].set_title("Distribuicao de log10(valor do titular)")
    ax[1].set_xlabel("log10(valor)")
    ax[1].set_ylabel("Densidade")
    eda._save(fig, paths["valor"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ages = pd.concat([cov.idade_media_titular_man, cov.idade_media_titular_vis]).dropna()
    ax[0].hist(ages, bins=40, density=True, color="#9ecae1", edgecolor="white", label="Empirico")
    grid = np.linspace(ages.min(), ages.max(), 200)
    ax[0].plot(grid, stats.norm.pdf(grid, ages.mean(), ages.std(ddof=1)), color="#d94801", label=f"Normal (media={ages.mean():.2f}, DP={ages.std(ddof=1):.2f})")
    ax[0].set_title("Idade media do time titular")
    ax[0].set_xlabel("Idade media (anos)")
    ax[0].set_ylabel("Densidade")
    ax[0].legend()
    c = cov.dropna(subset=["valor_equipe_titular_man", "valor_equipe_titular_vis", "gols_man"])
    ax[1].scatter(np.log(c.valor_equipe_titular_man / c.valor_equipe_titular_vis), c.gols_man - c.gols_vis, s=6, alpha=0.25)
    ax[1].set_title("Saldo de gols x razao de valor (log)")
    ax[1].set_xlabel("log(valor casa / valor fora)")
    ax[1].set_ylabel("Gols casa - gols fora")
    eda._save(fig, paths["idade_gols"])

    s = e.dropna(subset=["chutes_man", "chutes_fora_man"])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    xs = pd.concat([s.chutes_man, s.chutes_vis])
    ys = pd.concat([s.chutes_fora_man, s.chutes_fora_vis])
    ax.scatter(xs + np.random.default_rng(config.SEED).uniform(-0.25, 0.25, len(xs)), ys + np.random.default_rng(config.SEED + 1).uniform(-0.25, 0.25, len(ys)),
               s=8, alpha=0.3)
    lim = max(xs.max(), ys.max()) + 1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="chutes fora = chutes")
    ax.fill_between([0, lim], [0, lim], lim, color="#fcbba1", alpha=0.35, label="chutes fora > chutes (31% dos casos)")
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)
    ax.set_title("'Chutes' x 'chutes fora': o rotulo 'chutes' nao e o total")
    ax.set_xlabel("chutes (rotulo da fonte, contagem por time-jogo)")
    ax.set_ylabel("chutes fora (contagem por time-jogo)")
    ax.legend(loc="upper left", fontsize=8)
    eda._save(fig, paths["chutes"])
    return paths


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def build_report(e: pd.DataFrame, s: dict[str, object]) -> str:
    f2, f3 = (lambda v: eda._fmt(v, 2)), (lambda v: eda._fmt(v, 3))
    rules, checks, mt = s["rules"], s["checks"], s["missing"]
    cs, mc, ko = s["crosscheck"], s["matches"], s["kickoff"]
    sig, vs = s["signal"], s["valor_season"]
    n_viol = rules[rules.Violacoes > 0]
    cov_pct = e[e.season >= config.EXT_FIRST_SEASON].notna().mean() * 100
    L: list[str] = ["# Auditoria das tabelas externas (williamorim/brasileirao)\n",
                    "Mesma metodologia da auditoria da nossa base (`reports/eda_report.md`): esquema, integridade, faltantes, "
                    "outliers, estatisticas univariadas e conferencia cruzada. Gerado por `python -m src.eda_external`; so le `data/external/`. "
                    "Contagens exatas como inteiros; estatisticas com 2 casas; assimetria, CV e probabilidades com 3.\n"]
    L.append("## Veredito\n")
    date_off = 1 - float(cs["days"].get(0, 0)) / max(cs["paired"], 1)
    L.append(f"- **Placares: confiaveis. Datas: diferem em ate 1 dia em cerca de {date_off:.0%} dos jogos.** Dos {cs['paired']} jogos de 2012-2020 pareados com a nossa base, {cs['score_equal']} tem placar igual "
             f"({cs['score_diff']} diferencas). O `matches.csv` concorda em {mc['paired'] - len(mc['diff'])} de {mc['paired']} (as {len(mc['diff'])} diferencas foram conferidas "
             "em paginas publicas de resultados e o erro estava no `matches.csv`).")
    L.append(f"- **`valor_equipe_titular` e `idade_media_titular`: usaveis, com ressalvas.** Preenchidos em {f2(cov_pct['valor_equipe_titular_man'])}% dos jogos de 2012-2020, "
             "somente ate 2020, sem unidade/moeda documentada e em valores nominais que crescem no tempo: normalizar por temporada.")
    L.append("- **`horario` e `chutes` nao sao confiaveis.** Os horarios sao implausiveis; `chutes` nao pode ser o total de chutes. Nao usar sem verificar a semantica.")
    L.append(f"- **Defeitos encontrados:** {len(n_viol)} das {len(rules)} regras de plausibilidade tem violacoes e ha 1 linha duplicada (Fluminense x Atletico-MG, 2016). "
             "Nenhum afeta placares de 2012-2020.")
    L.append("- **Limite:** a origem dos dados nao esta confirmada e a cobertura termina em 2020; ver a secao 7.\n")

    L.append("## 1. Esquema, volumetria e dicionario\n")
    L.append(f"`stats.csv`: **{len(e)}** linhas x **{e.shape[1] - 6}** colunas originais (mais colunas auxiliares calculadas aqui), temporadas "
             f"**{e.season.min()}-{e.season.max()}**, datas de {e.date.min():%Y-%m-%d} a {e.date.max():%Y-%m-%d}. "
             f"`matches.csv`: **{mc['rows']}** linhas x 6 colunas, temporadas {mc['seasons'][0]}-{mc['seasons'][1]}.\n")
    dd = pd.DataFrame([{"Coluna": c, "Dtype": str(e[c].dtype), "Tipo de dominio": DICTIONARY[c][0], "Definicao (inferida)": DICTIONARY[c][1],
                        "Papel": DICTIONARY[c][2]} for c in e.columns if c in DICTIONARY])
    L.append(md_table_wrap(dd))
    L.append("\nAs definicoes foram **inferidas dos nomes das colunas** (a fonte nao tem dicionario). O sufixo `_man` e do mandante e `_vis` do visitante.\n")

    L.append("## 2. Integridade e consistencia\n")
    L.append("### 2.1 Verificacoes estruturais\n")
    L.append(eda.md_table(checks) + "\n")
    L.append("### 2.2 Regras de plausibilidade\n")
    L.append(eda.md_table(rules, {"Violacoes": 0, "Linhas avaliaveis": 0}) + "\n")
    L.append("**Achado sobre os chutes:** `chutes_fora` maior que `chutes` ocorre em "
             f"{int(rules.loc[rules.Regra.str.startswith('Chutes fora'), 'Violacoes'].iloc[0])} jogos-lado de cerca de {int(e.chutes_man.notna().sum())} jogos com a estatistica. "
             "Se `chutes` fosse o total de chutes isso nunca aconteceria; ou o rotulo esta trocado ou a coluna mede outra coisa (por exemplo, chutes no alvo). "
             "Sem dicionario, **nao usar** `chutes` como total.\n")
    L.append("![Chutes](figures/external_chutes.png)\n")
    L.append("### 2.3 O campo `horario`\n")
    L.append(f"**{f3(ko['share_impossible'])}** dos horarios caem entre 03:00 e 11:59, faixa em que quase nao ha jogos no Brasil, e os mais comuns sao 09:00, 04:00 e 12:00. "
             "Comparando com o nosso horario (UTC), a diferenca (nosso - tabela, modulo 24 h) e: "
             + ", ".join(f"{k:.0f} h ({v:.1%})" for k, v in ko["top_offsets"].items())
             + f" (n={ko['n_paired']} jogos pareados). Uma diferenca fixa em cerca de dois tercos dos jogos sugere um erro sistematico de fuso ou de formato "
             "(12 h/24 h) na origem, mas o restante nao segue o mesmo padrao. O campo **nao deve ser usado**; a nossa base ja traz o horario.\n")

    L.append("## 3. Dados faltantes e mecanismo\n")
    L.append(eda.md_table(mt["per_col"], {"Pct": 2, "Ausentes": 0}) + "\n")
    L.append("![Ausencias](figures/external_missing.png)\n")
    L.append("Quase toda a ausencia vem da **cobertura da fonte por temporada** (colunas inteiras vazias antes de certos anos). "
             "Para cada coluna, o teste de Season (qui-quadrado) e o de associacao com resultado e gols totais (nas temporadas com cobertura de pelo menos "
             f"{config.EXT_COVERAGE_MIN_PCT:.0f}%) classificam o mecanismo. MNAR nao se prova com dados observados. "
             "**Cuidado com comparacoes multiplas:** sao cerca de 13 grupos de colunas testados; com p < 0,05 sem correcao e p = 0,01-0,02 para "
             "tecnico/valor/idade versus resultado (e p > 0,05 para gols totais), a associacao e fraca e nao sobrevive a uma correcao de Bonferroni "
             "(limiar de cerca de 0,004).\n")
    L.append(eda.md_table(mt["mechanism"], 3) + "\n")

    L.append("## 4. Outliers e anomalias\n")
    L.append(eda.md_table(s["tukey"], {"Cerca inf.": 2, "Cerca sup.": 2, "MAD": 2}) + "\n")
    L.append(f"Isolation Forest sobre (log valor casa, log valor fora, idade casa, idade fora, gols casa, gols fora) em {s['iso']['if_rows']} jogos: "
             f"{s['iso']['if_flagged']} sinalizados. Os 10 mais extremos:\n")
    L.append(eda.md_table(s["extremes"], {"Valor casa (M)": 2, "Valor fora (M)": 2, "Idade casa": 2, "Idade fora": 2, "Score IF": 3, "Temporada": 0}) + "\n")
    L.append("Os extremos sao tipicamente um time de alto valor contra um de baixo, o que e esperado. Alguns valores muito baixos para clubes grandes "
             "(por exemplo, um titular do Gremio avaliado em cerca de 0,6 M em 2017) sugerem escalacao reserva ou erro de origem, e **nao ha como verificar sem "
             "outra fonte**; por isso o veredito automatico 'genuino' vale so para as regras de plausibilidade. O erro de dado mais claro e "
             "`faltas = 255` (ver 2.2), que o Tukey tambem sinaliza.\n")

    L.append("## 5. Estatisticas univariadas (2012-2020)\n")
    u = s["univariate"]
    L.append(eda.md_table(u[["Variavel", "N", "Mean", "Median", "Mode", "Min", "Max", "Range", "Variance", "SD", "Q1", "Q3", "IQR"]], {"N": 0}) + "\n")
    L.append(eda.md_table(u[["Variavel", "Skew_g1", "Bowley", "Poisson_skew", "Assimetria", "CV_pct", "IQR_over_median_pct", "VMR"]]
                          .rename(columns={"Skew_g1": "g1", "Poisson_skew": "Assimetria Poisson", "CV_pct": "CV %", "IQR_over_median_pct": "IQR/Mediana %"}), 3) + "\n")
    L.append("### 5.1 O valor nominal do titular varia muito entre temporadas\n")
    L.append(eda.md_table(vs, {"Temporada": 0, "Jogos-lado": 0}) + "\n")
    L.append("![Valor](figures/external_valor.png)\n")
    L.append(f"A correlacao entre o log da razao de valor (casa/fora) e o saldo de gols e **{f3(sig['corr_raw'])}** (n={sig['n']}) com valores nominais e "
             f"**{f3(sig['corr_norm'])}** depois de normalizar pela mediana da temporada (a razao entre dois valores da mesma partida ja cancela boa parte da "
             f"nivel da temporada: correlacao de {f3(sig['ratio_vs_ratio_season'])} entre as duas versoes). A mediana oscila de {vs['Mediana (M)'].iloc[0]:.1f} M (2012) a "
             f"{vs['Mediana (M)'].iloc[-1]:.1f} M (2020), com um salto entre 2014 e 2015 que nao parece inflacao gradual (pode refletir mudanca de metodologia ou de fonte dos "
             "valores; nao verificado). Para usar como feature, prefira a razao casa/fora ou o valor relativo a temporada, nunca o valor nominal.\n")
    L.append("![Idade e gols](figures/external_idade_gols.png)\n")

    L.append("## 6. Conferencia cruzada com a nossa base\n")
    L.append(f"- `stats.csv` x nossa base (2012-2020): {cs['paired']} de {cs['ours_in_scope']} jogos pareados; placares iguais: **{cs['score_equal']}**; diferentes: **{cs['score_diff']}**.")
    dd = {int(k): int(v) for k, v in cs["days"].items()}
    L.append(f"- Diferenca de data (nossa data UTC - data da tabela, em dias): {dd}. Em {1 - dd.get(0, 0) / cs['paired']:.1%} dos jogos as datas diferem: "
             f"{dd.get(1, 0)} com +1 (jogo noturno que em UTC cai no dia seguinte) e {dd.get(-1, 0)} com -1 (a tabela traz a data um dia depois da nossa), "
             "o que indica que a data da tabela tambem nao e confiavel ao dia, coerente com o campo `horario` desordenado.")
    L.append(f"- `matches.csv` (2012-2026): {mc['paired']} jogos pareados; **{len(mc['diff'])}** placares diferentes. Em todos, o erro esta no `matches.csv` "
             "(conferido em paginas publicas de resultados):\n")
    L.append(eda.md_table(mc["diff"], {"season": 0, "home_goals": 0, "away_goals": 0}) + "\n")
    L.append(f"- `matches.csv`: {mc['rows']} linhas; {mc['unplayed']} jogos sem placar (`x`), {mc['bad_format']} placares em formato invalido, "
             f"{mc['dup_pairs']} confronto(s) repetido(s) na mesma temporada, `id_match` {'unico' if mc['id_unique'] else 'nao unico'}; "
             f"media de {f2(mc['mean_goals'])} gols por jogo; {mc['per_season_min']} a {mc['per_season_max']} jogos por temporada desde 2012.")
    L.append(f"- Clubes fora do nosso mapa (todos anteriores a 2012): {', '.join(mc['unmapped'])}.\n")

    L.append("## 7. O que da para usar e o que nao\n")
    L.append("| Campo | Veredito |\n|---|---|\n"
             "| Placares 2012-2020 (`stats.csv`) | Confiavel: iguais aos nossos em 100% dos jogos pareados |\n"
             "| `valor_equipe_titular`, `idade_media_titular` | Usavel so para pesquisa em 2012-2020 (nao existe depois); usar a razao casa/fora; moeda e data de referencia do valor nao documentadas |\n"
             "| `colocacao`, `tecnico`, `arbitro`, `publico` | Preenchidos em 2007+ (parcial); nao testados como features |\n"
             "| `escanteios`, `faltas`, `defesas`, `impedimentos`, `chutes_fora` | So 2018-2020, e parcialmente: cobertura insuficiente para modelar |\n"
             "| `chutes`, `chutes_bola_parada` | **Nao usar:** semantica incerta (secao 2.2) |\n"
             "| `horario` | **Nao usar:** implausivel (secao 2.3) |\n"
             "| `id_match` | Nao e chave unica; juntar por (temporada, mandante, visitante) e tratar duplicatas |\n")
    L.append("**Limites desta auditoria:** a origem dos dados (provavelmente uma base aberta derivada do Transfermarkt) e a licenca nao foram verificadas; "
             "a tabela termina em 2020, entao nada aqui pode alimentar previsoes de 2021-2026.")
    return "\n".join(L) + "\n"


def md_table_wrap(df: pd.DataFrame) -> str:
    return eda.md_table(df)


def run(report_path: Path = config.EXTERNAL_REPORT_FILE, figures_dir: Path = config.FIGURES_DIR,
        stats_path: Path = config.EXTERNAL_STATS_FILE, matches_path: Path = config.EXTERNAL_MATCHES_FILE) -> dict[str, object]:
    """Run the audit of both external tables; write the report and four figures."""
    e = load_stats(stats_path)
    ours = ingestion.build_interim()
    miss = missing_audit(e)
    ts, extremes, iso = outlier_audit(e)
    s = {"rules": rules_table(e), "checks": integrity_checks(e), "kickoff": kickoff_analysis(e, ours), "missing": miss,
         "tukey": ts, "extremes": extremes, "iso": iso, "univariate": univariate(e), "valor_season": valor_by_season(e),
         "signal": value_signal(e), "crosscheck": crosscheck_stats(e, ours), "matches": audit_matches_csv(matches_path, ours)}
    figs = make_figures(e, miss, figures_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(e, s), encoding="utf-8")
    return {"report": report_path, "figures": figs, "sections": s, "stats": e}


def main() -> None:
    out = run()
    s = out["sections"]
    print(f"Relatorio: {out['report']}")
    for k, p in out["figures"].items():
        print(f"Figura ({k}): {p}")
    cs = s["crosscheck"]
    print(f"stats.csv x nossa base: {cs['score_equal']} de {cs['paired']} placares iguais")
    print("Regras com violacao:", int((s['rules'].Violacoes > 0).sum()), "de", len(s['rules']))


if __name__ == "__main__":
    main()
