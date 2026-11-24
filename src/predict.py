"""Inference for upcoming fixtures with the production Dixon-Coles model.

Usage
-----
    python -m src.predict --refit --match Flamengo Palmeiras
    python -m src.predict --fixtures fixtures.csv      # columns: home,away
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import baselines, config, ingestion, probability

MODEL_FILE: Path = config.MODELS_DIR / "dc_production.json"
CANONICAL_TEAMS: frozenset[str] = frozenset(config.TEAM_MAP.values())
RHO_SEASONS: int = 6  # seasons of out-of-sample predictions used to estimate rho


#: extra spellings seen in other sources (API-Football etc.), normalized form -> canonical
ALIASES: dict[str, str] = {
    "vasco da gama": "Vasco", "cr vasco da gama": "Vasco", "rb bragantino": "Bragantino",
    "red bull bragantino": "Bragantino", "atletico mineiro": "Atletico-MG",
    "atletico paranaense": "Athletico-PR", "athletico paranaense": "Athletico-PR",
    "atletico goianiense": "Atletico-GO", "america mineiro": "America-MG",
    "sport recife": "Sport", "clube do remo": "Remo", "chapecoense sc": "Chapecoense",
    "botafogo rj": "Botafogo", "flamengo rj": "Flamengo", "gremio fbpa": "Gremio",
    "fortaleza ec": "Fortaleza",
}


def _norm(name: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def resolve_team(name: str) -> str:
    """Map any known spelling (source, canonical, accents, API-Football) to the canonical name.

    Order: exact source name, normalized canonical/alias match, close fuzzy match.
    Unknown names are returned unchanged (and later treated as an average team).
    """
    name = name.strip()
    if name in config.TEAM_MAP:
        return config.TEAM_MAP[name]
    key = _norm(name)
    table = {_norm(c): c for c in CANONICAL_TEAMS}
    table.update({_norm(k): v for k, v in config.TEAM_MAP.items()})
    table.update(ALIASES)
    if key in table:
        return table[key]
    import difflib
    close = difflib.get_close_matches(key, list(table), n=1, cutoff=0.88)
    return table[close[0]] if close else name


def fit_production(matches: pd.DataFrame, half_life_days: float = 730.0) -> dict:
    """Fit Dixon-Coles on all played matches and estimate ``rho`` out-of-sample.

    ``rho`` is fitted on walk-forward (out-of-sample) lambdas of the last
    ``RHO_SEASONS`` seasons, so it is not biased by predictions of matches the
    model was trained on. The in-sample estimate is kept for reference.

    Returns
    -------
    dict
        JSON-serializable artifact: model parameters, ``rho``, ``rho_in_sample``
        and training metadata.
    """
    ref = matches["kickoff"].max() + pd.Timedelta(1, unit="D")
    dc = baselines.DixonColes(half_life_days).fit(matches, ref)
    hg, ag = matches["home_goals"].to_numpy(), matches["away_goals"].to_numpy()
    lh, la = dc.predict(matches)
    rho_in = probability.fit_rho(lh, la, hg, ag)

    seasons = sorted(matches["season"].unique())
    start = int(seasons[max(1, len(seasons) - RHO_SEASONS)]) if len(seasons) > 1 else int(seasons[0])
    oos = baselines.dc_oos_lambdas(matches, start_season=start, half_life_days=half_life_days)
    m = matches.merge(oos, on="match_id")
    rho = probability.fit_rho(m["dc_lh"].to_numpy(), m["dc_la"].to_numpy(),
                              m["home_goals"].to_numpy(), m["away_goals"].to_numpy()) if len(m) else rho_in
    return {"model": dc.to_dict(), "rho": rho, "rho_in_sample": rho_in,
            "trained_through": str(matches["kickoff"].max()), "n_matches": int(len(matches))}


def save_artifact(artifact: dict, path: Path = MODEL_FILE) -> None:
    """Write the artifact as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=1))


def load_artifact(path: Path = MODEL_FILE) -> dict:
    """Read an artifact written by :func:`save_artifact`."""
    return json.loads(path.read_text())


def predict_fixtures(artifact: dict, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Predict upcoming fixtures.

    Parameters
    ----------
    artifact : dict
        Output of :func:`fit_production` / :func:`load_artifact`.
    fixtures : pd.DataFrame
        Columns ``home`` and ``away`` (team names, any known spelling).

    Returns
    -------
    pd.DataFrame
        Expected goals, 1X2, Over 2.5, BTTS, top-3 scorelines and a
        ``warning`` column (unknown/new teams are treated as average).
    """
    dc = baselines.DixonColes.from_dict(artifact["model"])
    df = pd.DataFrame({"home_team": fixtures["home"].map(resolve_team),
                       "away_team": fixtures["away"].map(resolve_team)})
    lh, la = dc.predict(df)
    mat = probability.scoreline_matrix(lh, la, artifact["rho"])
    p = probability.outcome_probs(mat)
    mk = probability.market_probs(mat)

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        unknown = [t for t in (r.home_team, r.away_team) if t not in dc.att]
        top = probability.top_scorelines(mat[i], 3)
        rows.append({
            "home": r.home_team, "away": r.away_team,
            "lam_home": lh[i], "lam_away": la[i],
            "p_home": p[i, 0], "p_draw": p[i, 1], "p_away": p[i, 2],
            "p_over_2_5": mk["over_2_5"][i], "p_btts": mk["btts"][i],
            **{f"top{k + 1}": f"{h}-{a} ({pr:.1%})" for k, (h, a, pr) in enumerate(top)},
            "warning": ("unknown team(s), treated as average: " + ", ".join(unknown)) if unknown else "",
        })
    return pd.DataFrame(rows)


def _print(res: pd.DataFrame) -> None:
    for r in res.itertuples(index=False):
        when = f"[{r.kickoff}] " if hasattr(r, "kickoff") else ""
        print(f"\n{when}{r.home} x {r.away}   (xG {r.lam_home:.2f} - {r.lam_away:.2f})")
        print(f"  1X2: {r.p_home:.1%} / {r.p_draw:.1%} / {r.p_away:.1%}   "
              f"Over2.5: {r.p_over_2_5:.1%}   BTTS: {r.p_btts:.1%}")
        print(f"  Placares: {r.top1} | {r.top2} | {r.top3}")
        if r.warning:
            print(f"  AVISO: {r.warning}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--refit", action="store_true", help="refit on all played matches and save")
    ap.add_argument("--fixtures", type=Path, help="CSV with columns home,away")
    ap.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"))
    ap.add_argument("--upcoming", action="store_true",
                    help="fetch the next fixtures from API-Football (1 request/day, cached)")
    ap.add_argument("--remaining", action="store_true",
                    help="predict every matchup still unplayed this season (no API needed)")
    args = ap.parse_args()

    if args.refit or not MODEL_FILE.exists():
        matches = ingestion.build_interim()
        art = fit_production(matches)
        save_artifact(art)
        print(f"Modelo salvo ({art['n_matches']} jogos ate {art['trained_through']}, rho={art['rho']:.3f})")
    art = load_artifact()

    if args.remaining:
        from src import fixtures as api
        rem = api.attach_schedule(api.remaining_fixtures(ingestion.build_interim()), api.load_schedule())
        res = predict_fixtures(art, rem)
        res.insert(0, "kickoff", rem["kickoff"].dt.strftime("%d/%m %H:%M").fillna("sem data").to_numpy())
        print(f"{len(res)} jogos restantes da temporada, em ordem cronologica:")
        _print(res)
        return
    if args.upcoming:
        from src import fixtures as api
        try:
            up = api.upcoming_fixtures()
        except api.ApiError as e:
            print(f"API indisponivel: {e}\nUse --remaining (jogos que faltam na temporada) ou --fixtures.")
            return
        if up.empty:
            print("Nenhum jogo futuro retornado pela API.")
            return
        fx = pd.DataFrame({"home": up["home_api"], "away": up["away_api"]})
        res = predict_fixtures(art, fx)
        res.insert(0, "kickoff", up["kickoff"].dt.strftime("%d/%m %H:%M").to_numpy())
        _print(res)
        return
    if args.fixtures:
        fx = pd.read_csv(args.fixtures)
    elif args.match:
        fx = pd.DataFrame({"home": [args.match[0]], "away": [args.match[1]]})
    else:
        return
    _print(predict_fixtures(art, fx))


if __name__ == "__main__":
    main()
