"""Cross-source validation: compare our CSV with API-Football (independent source).

The free plan serves seasons 2022-2024 only, so the comparison covers those three
seasons (3 requests, cached per day). Writes ``reports/crosscheck.md``.

    python -m src.crosscheck
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config, fixtures, ingestion, predict

SEASONS: tuple[int, ...] = (2022, 2023, 2024)


def fetch_season(season: int) -> pd.DataFrame:
    """Finished fixtures of a season from API-Football with canonical team names and full-time goals."""
    body = fixtures.api_get("/fixtures", {"league": fixtures.LEAGUE_ID_SERIE_A, "season": season})
    rows = []
    for it in body.get("response", []):
        ft = it["score"]["fulltime"]
        if it["fixture"]["status"]["short"] not in ("FT", "AET", "PEN") or ft["home"] is None:
            continue
        rows.append({"season": season, "home": predict.resolve_team(it["teams"]["home"]["name"]),
                     "away": predict.resolve_team(it["teams"]["away"]["name"]),
                     "api_home_goals": int(ft["home"]), "api_away_goals": int(ft["away"]),
                     "api_date": pd.to_datetime(it["fixture"]["date"]).tz_convert("UTC").tz_localize(None)})
    return pd.DataFrame(rows)


def compare(ours: pd.DataFrame, api: pd.DataFrame) -> dict[str, object]:
    """Match fixtures by (season, home, away) and compare scores and dates."""
    o = ours.rename(columns={"home_team": "home", "away_team": "away"})[
        ["season", "home", "away", "kickoff", "home_goals", "away_goals"]]
    m = o.merge(api, on=["season", "home", "away"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"].copy()
    both["score_ok"] = (both.home_goals == both.api_home_goals) & (both.away_goals == both.api_away_goals)
    both["days_apart"] = (both.kickoff - both.api_date).abs().dt.total_seconds() / 86400
    return {
        "n_ours": int(m["_merge"].isin(["both", "left_only"]).sum()),
        "n_api": int(m["_merge"].isin(["both", "right_only"]).sum()),
        "matched": int(len(both)),
        "score_mismatch": both[~both.score_ok],
        "date_off": int((both.days_apart > 1.0).sum()),
        "only_ours": m[m["_merge"] == "left_only"][["season", "home", "away"]],
        "only_api": m[m["_merge"] == "right_only"][["season", "home", "away"]],
        "by_season": both.groupby("season").agg(jogos=("score_ok", "size"), placares_iguais=("score_ok", "sum")).reset_index(),
    }


def build_report(res: dict[str, object]) -> str:
    """Markdown report of the comparison."""
    mism = res["score_mismatch"]
    L = ["# Conferencia com fonte independente (API-Football)\n",
         "Comparacao jogo a jogo dos placares do `BRA_footballdata.csv` com a API-Football para as temporadas liberadas "
         f"no plano gratuito ({', '.join(map(str, SEASONS))}). Chave: (temporada, mandante, visitante); datas comparadas em UTC.\n",
         f"- Jogos no nosso CSV nessas temporadas: **{res['n_ours']}**; na API (finalizados): **{res['n_api']}**; pareados: **{res['matched']}**.",
         f"- **Placares divergentes: {len(mism)}** ({len(mism) / max(res['matched'], 1):.2%} dos jogos pareados).",
         f"- Datas com mais de 1 dia de diferenca: **{res['date_off']}**.",
         f"- So no nosso CSV: {len(res['only_ours'])}; so na API: {len(res['only_api'])}.\n",
         "| Temporada | Jogos pareados | Placares iguais |", "|---|---|---|"]
    for r in res["by_season"].itertuples():
        L.append(f"| {r.season} | {r.jogos} | {int(r.placares_iguais)} |")
    if len(mism):
        L.append("\n## Divergencias de placar\n")
        L.append("| Temporada | Jogo | Nosso CSV | API |\n|---|---|---|---|")
        for r in mism.itertuples():
            L.append(f"| {r.season} | {r.home} x {r.away} | {int(r.home_goals)}-{int(r.away_goals)} | {r.api_home_goals}-{r.api_away_goals} |")
    for key, title in (("only_ours", "So no nosso CSV"), ("only_api", "So na API")):
        df = res[key]
        if len(df):
            L.append(f"\n## {title}\n")
            L.append("| Temporada | Jogo |\n|---|---|")
            L.extend(f"| {r.season} | {r.home} x {r.away} |" for r in df.itertuples())
    return "\n".join(L) + "\n"


def run(path: Path = config.CROSSCHECK_REPORT_FILE) -> dict[str, object]:
    """Fetch, compare and write the report."""
    ours = ingestion.build_interim()
    ours = ours[ours.season.isin(SEASONS)]
    api = pd.concat([fetch_season(s) for s in SEASONS], ignore_index=True)
    res = compare(ours, api)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(res), encoding="utf-8")
    return res


def main() -> None:
    res = run()
    print(f"pareados {res['matched']} de {res['n_ours']} (API {res['n_api']}); placares divergentes: "
          f"{len(res['score_mismatch'])}; datas fora (>1 dia): {res['date_off']}; "
          f"so nosso: {len(res['only_ours'])}; so API: {len(res['only_api'])}")
    print(f"relatorio: {config.CROSSCHECK_REPORT_FILE}")


if __name__ == "__main__":
    main()
