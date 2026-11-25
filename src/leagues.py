"""League registry: everything that differs between competitions lives here.

A :class:`League` bundles the data location, season labelling, number of teams,
model/prediction paths, split seasons and team-name resolution. The rest of the
pipeline (ingestion, backtest, prediction, forward test, refresh) receives a
league and stays free of country-specific constants.

Season labels
-------------
* ``BRA`` (Brasileirão): calendar year of the season (2020 ran Aug/2020-Feb/2021).
* football-data "main" leagues (``E0``, ``SP1``, ...): the year the season *starts*
  (``2025`` = 2025/26).
"""
from __future__ import annotations

import difflib
import functools
import hashlib
import json
import re
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Mapping

from src import config

FD_BASE_URL = "https://football-data.co.uk"
FD_USER_AGENT = "brasileirao-predictor/0.1 (research)"
FD_REQUIRED_COLUMNS: tuple[str, ...] = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")

#: extra spellings of Brazilian clubs seen in other sources (normalized form -> canonical name)
BRA_ALIASES: dict[str, str] = {
    "vasco da gama": "Vasco", "cr vasco da gama": "Vasco", "rb bragantino": "Bragantino",
    "red bull bragantino": "Bragantino", "atletico mineiro": "Atletico-MG",
    "atletico paranaense": "Athletico-PR", "athletico paranaense": "Athletico-PR",
    "atletico goianiense": "Atletico-GO", "america mineiro": "America-MG",
    "sport recife": "Sport", "clube do remo": "Remo", "chapecoense sc": "Chapecoense",
    "botafogo rj": "Botafogo", "flamengo rj": "Flamengo", "gremio fbpa": "Gremio",
    "fortaleza ec": "Fortaleza",
}

#: Premier League: API-Football / press spellings -> football-data.co.uk names
E0_ALIASES: dict[str, str] = {
    "manchester united": "Man United", "manchester city": "Man City", "nottingham forest": "Nott'm Forest",
    "nottm forest": "Nott'm Forest", "wolverhampton wanderers": "Wolves", "newcastle united": "Newcastle",
    "tottenham hotspur": "Tottenham", "west ham united": "West Ham", "brighton hove albion": "Brighton",
    "brighton and hove albion": "Brighton", "leicester city": "Leicester", "leeds united": "Leeds",
    "norwich city": "Norwich", "sheffield utd": "Sheffield United", "west bromwich albion": "West Brom",
    "cardiff city": "Cardiff", "huddersfield town": "Huddersfield", "swansea city": "Swansea",
    "stoke city": "Stoke", "hull city": "Hull", "queens park rangers": "QPR", "ipswich town": "Ipswich",
    "luton town": "Luton", "afc bournemouth": "Bournemouth", "birmingham city": "Birmingham",
}


@dataclass(frozen=True)
class League:
    """Static description of one competition (see the module docstring)."""

    code: str
    name: str
    n_teams: int
    source: str                      # "football-data-new" (one CSV) | "football-data-main" (one CSV per season)
    raw_dir: Path
    first_season: int
    model_file: Path
    pred_dir: Path
    warmup_seasons: tuple[int, ...]
    first_validation_season: int
    div: str = ""                    # league code inside football-data's fixtures.csv
    aliases: Mapping[str, str] = field(default_factory=dict)   # normalized alias -> canonical name
    team_map: Mapping[str, str] | None = None                  # source spelling -> canonical (Brazil only)
    schedule_file: Path | None = None
    fixtures_file: Path | None = None

    @property
    def games_per_team(self) -> int:
        """Games each team plays in a full double round-robin season."""
        return 2 * (self.n_teams - 1)

    @property
    def matches_per_season(self) -> int:
        """Matches in a complete double round-robin season."""
        return self.n_teams * (self.n_teams - 1)

    def season_tag(self, season: int) -> str:
        """football-data folder tag of a start-year season (``2012 -> '1213'``)."""
        return f"{season % 100:02d}{(season + 1) % 100:02d}"

    def data_files(self) -> list[Path]:
        """Raw CSV files of this league (one for ``football-data-new``, one per season otherwise)."""
        if self.source == "football-data-new":
            return [config.RAW_MATCHES_FILE]
        return sorted(self.raw_dir.glob(f"{self.code}_*.csv"))

    def current_season(self, today: date | None = None) -> int:
        """Season label of the season in progress on ``today``."""
        today = today or date.today()
        return today.year if self.source == "football-data-new" else (today.year if today.month >= 7 else today.year - 1)

    def test_season(self, matches: "object") -> int:
        """Latest season with a complete set of matches (the natural hold-out)."""
        sizes = matches.groupby("season").size()  # type: ignore[attr-defined]
        full = sizes[sizes >= self.matches_per_season - 1]
        return int(full.index.max())


def _fd_main(code: str, name: str, n_teams: int, slug: str, aliases: Mapping[str, str] | None = None) -> League:
    first = 2012
    return League(
        code=code, name=name, n_teams=n_teams, source="football-data-main",
        raw_dir=config.ROOT / "data" / "external" / f"football_data_{slug}", first_season=first,
        model_file=config.MODELS_DIR / f"dc_{code}.json", pred_dir=config.ROOT / "predictions" / code,
        warmup_seasons=(first, first + 1), first_validation_season=first + 4, div=code,
        aliases=aliases or {}, fixtures_file=config.ROOT / "data" / "external" / "football_data_england" / "fixtures.csv",
    )


BRA = League(
    code="BRA", name="Brasileirão Série A", n_teams=config.TEAMS_PER_SEASON, source="football-data-new",
    raw_dir=config.DATA_RAW, first_season=2012, model_file=config.MODELS_DIR / "dc_production.json",
    pred_dir=config.ROOT / "predictions", warmup_seasons=config.WARMUP_SEASONS,
    first_validation_season=config.FIRST_VALIDATION_SEASON, aliases=BRA_ALIASES, team_map=config.TEAM_MAP,
    schedule_file=config.DATA_RAW / "schedule_2026_manual.csv",
)

LEAGUES: dict[str, League] = {
    "BRA": BRA,
    "E0": _fd_main("E0", "Premier League", 20, "england", E0_ALIASES),
    "SP1": _fd_main("SP1", "La Liga", 20, "spain"),
    "I1": _fd_main("I1", "Serie A (Itália)", 20, "italy"),
    "D1": _fd_main("D1", "Bundesliga", 18, "germany"),
    "F1": _fd_main("F1", "Ligue 1", 18, "france"),
}


def get(code: str | League) -> League:
    """Look up a league by code (case-insensitive); a :class:`League` is returned unchanged."""
    if isinstance(code, League):
        return code
    key = code.strip().upper()
    if key not in LEAGUES:
        raise KeyError(f"Unknown league {code!r}; available: {', '.join(sorted(LEAGUES))}")
    return LEAGUES[key]


# --------------------------------------------------------------------------- #
# team names
# --------------------------------------------------------------------------- #

def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def resolve_name(name: str, canonical: frozenset[str] | set[str], aliases: Mapping[str, str],
                 source_map: Mapping[str, str] | None = None, cutoff: float = 0.88) -> str:
    """Map a spelling to the canonical team name.

    Order: exact source spelling, normalized canonical/alias match, close fuzzy match.
    Unknown names are returned unchanged (and later treated as an average team).
    """
    name = name.strip()
    if source_map and name in source_map:
        return source_map[name]
    if name in canonical:
        return name
    key = _norm(name)
    table = {_norm(c): c for c in canonical}
    if source_map:
        table.update({_norm(k): v for k, v in source_map.items()})
    table.update(aliases)
    if key in table:
        return table[key]
    close = difflib.get_close_matches(key, list(table), n=1, cutoff=cutoff)
    return table[close[0]] if close else name


@functools.lru_cache(maxsize=None)
def _canonical_cached(code: str, signature: tuple) -> frozenset[str]:
    from src import ingestion

    lg = LEAGUES[code]
    m = ingestion.build_league(lg)
    return frozenset(set(m["home_team"]) | set(m["away_team"]))


def canonical_teams(league: League | str) -> frozenset[str]:
    """Canonical team names of a league (Brazil: from the team map; others: from the data files)."""
    lg = get(league)
    if lg.team_map is not None:
        return frozenset(lg.team_map.values())
    sig = tuple((p.name, p.stat().st_mtime_ns) for p in lg.data_files())
    return _canonical_cached(lg.code, sig) if sig else frozenset()


def resolver(league: League | str) -> Callable[[str], str]:
    """A function mapping any known spelling of a team of ``league`` to its canonical name."""
    lg = get(league)
    canon = canonical_teams(lg)
    return lambda name: resolve_name(name, canon, lg.aliases, lg.team_map)


# --------------------------------------------------------------------------- #
# downloading football-data "main" league files
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest`` through a temp file; return True if the bytes changed.

    The file is validated (required football-data columns in the header) before replacing ``dest``.
    """
    tmp = dest.with_suffix(".download")
    req = urllib.request.Request(url, headers={"User-Agent": FD_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"download failed for {url}: {e}") from None
    header = data.decode("utf-8-sig", errors="replace").splitlines()[0] if data else ""
    cols = {c.strip() for c in header.split(",")}
    if dest.name != "fixtures.csv" and not set(FD_REQUIRED_COLUMNS) <= cols:
        raise RuntimeError(f"unexpected header in {url}: {header[:80]!r}")
    tmp.write_bytes(data)
    changed = not dest.exists() or dest.read_bytes() != data
    tmp.replace(dest)
    return changed


def download_football_data(league: League | str, base_url: str = FD_BASE_URL, today: date | None = None) -> dict[str, list[str]]:
    """Fetch missing seasons and refresh the current one (plus ``fixtures.csv``); rewrite ``MANIFEST.json``.

    Historical seasons already on disk are not downloaded again.

    Returns
    -------
    dict
        ``changed`` (new or updated files), ``unchanged`` and ``missing`` (seasons the server does not have).
    """
    lg = get(league)
    if lg.source != "football-data-main":
        raise ValueError(f"{lg.code} is not a football-data main league")
    lg.raw_dir.mkdir(parents=True, exist_ok=True)
    cur = lg.current_season(today)
    out: dict[str, list[str]] = {"changed": [], "unchanged": [], "missing": []}
    for season in range(lg.first_season, cur + 1):
        tag = lg.season_tag(season)
        dest = lg.raw_dir / f"{lg.code}_{tag}.csv"
        if dest.exists() and season < cur:
            out["unchanged"].append(dest.name)
            continue
        try:
            changed = _fetch(f"{base_url}/mmz4281/{tag}/{lg.code}.csv", dest)
        except RuntimeError:
            if season == cur:
                raise
            out["missing"].append(tag)
            continue
        out["changed" if changed else "unchanged"].append(dest.name)
    if lg.fixtures_file is not None:
        lg.fixtures_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            _fetch(f"{base_url}/fixtures.csv", lg.fixtures_file)
            out["changed"].append("fixtures.csv")
        except RuntimeError:
            out["missing"].append("fixtures.csv")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in sorted(lg.raw_dir.glob("*.csv"))}
    (lg.raw_dir / "MANIFEST.json").write_text(json.dumps(
        {"source": f"{FD_BASE_URL}/mmz4281/YYZZ/{lg.code}.csv and /fixtures.csv", "downloaded": date.today().isoformat(),
         "files": files}, indent=1))
    return out
