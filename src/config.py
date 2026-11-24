"""Project-wide configuration: paths, seeds, windows, team map, split seasons."""
from __future__ import annotations

from pathlib import Path

SEED: int = 42

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_INTERIM: Path = ROOT / "data" / "interim"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
MODELS_DIR: Path = ROOT / "models"

RAW_MATCHES_FILE: Path = DATA_RAW / "BRA_footballdata.csv"
RAW_URL: str = "https://www.football-data.co.uk/new/BRA.csv"
INTERIM_MATCHES_FILE: Path = DATA_INTERIM / "matches.parquet"

ROLLING_WINDOWS: tuple[int, ...] = (3, 5, 10)
MAX_GOALS: int = 10  # scoreline grid is 0..MAX_GOALS per side

WARMUP_SEASONS: tuple[int, ...] = (2012, 2013)
FIRST_VALIDATION_SEASON: int = 2016
TEST_SEASON: int = 2025

# Source name -> canonical team name (handles source spelling quirks).
TEAM_MAP: dict[str, str] = {
    "America MG": "America-MG",
    "Athletico-PR": "Athletico-PR",
    "Atletico GO": "Atletico-GO",
    "Atletico-MG": "Atletico-MG",
    "Avai": "Avai",
    "Bahia": "Bahia",
    "Botafogo RJ": "Botafogo",
    "Bragantino": "Bragantino",
    "CSA": "CSA",
    "Ceara": "Ceara",
    "Chapecoense-SC": "Chapecoense",
    "Corinthians": "Corinthians",
    "Coritiba": "Coritiba",
    "Criciuma": "Criciuma",
    "Cruzeiro": "Cruzeiro",
    "Cuiaba": "Cuiaba",
    "Figueirense": "Figueirense",
    "Flamengo RJ": "Flamengo",
    "Fluminense": "Fluminense",
    "Fortaleza": "Fortaleza",
    "Goias": "Goias",
    "Gremio": "Gremio",
    "Internacional": "Internacional",
    "Joinville": "Joinville",
    "Juventude": "Juventude",
    "Mirassol": "Mirassol",
    "Nautico": "Nautico",
    "Palmeiras": "Palmeiras",
    "Parana": "Parana",
    "Ponte Preta": "Ponte-Preta",
    "Portuguesa": "Portuguesa",
    "Remo": "Remo",
    "Santa Cruz": "Santa-Cruz",
    "Santos": "Santos",
    "Sao Paulo": "Sao-Paulo",
    "Sport Recife": "Sport",
    "Vasco": "Vasco",
    "Vitoria": "Vitoria",
}


# ----------------------------------------------------------------------------- #
# EDA / data-quality audit (src/eda.py)
# ----------------------------------------------------------------------------- #
REPORTS_DIR: Path = ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
EDA_REPORT_FILE: Path = REPORTS_DIR / "eda_report.md"
CROSSCHECK_REPORT_FILE: Path = REPORTS_DIR / "crosscheck.md"
FIG_DPI: int = 300

IQR_FENCE_STANDARD: float = 1.5
IQR_FENCE_EXTREME: float = 3.0
Z_THRESHOLD: float = 3.0
MODIFIED_Z_THRESHOLD: float = 3.5
MAD_NORMAL_CONSTANT: float = 0.6745      # makes MAD comparable to a standard deviation
MEAN_AD_NORMAL_CONSTANT: float = 1.253314  # fallback scale when MAD == 0
ISOLATION_CONTAMINATION: float = 0.01
ISOLATION_TREES: int = 300
LOF_NEIGHBORS: int = 35
TOP_EXTREME_MATCHES: int = 10

MIN_SEASONS_PER_TEAM: int = 3
BOXPLOT_TEAMS: int = 8
MODE_DECIMALS: int = 2
SKEW_SYMMETRIC: float = 0.5
SKEW_STRONG: float = 1.0
POISSON_TAIL_POOL: int = 5               # goal counts >= this are pooled in the chi-square test
MISSING_ALPHA: float = 0.05
MISSING_ALPHA_STRICT: float = 0.01
MIN_GROUP_FOR_TESTS: int = 5

MAX_PLAUSIBLE_TEAM_GOALS: int = 10
MAX_PLAUSIBLE_TOTAL_GOALS: int = 12
OVERROUND_RANGE: tuple[float, float] = (1.0, 1.20)
GAMES_PER_TEAM_SEASON: int = 38
TEAMS_PER_SEASON: int = 20
TOP_TEAMS_FOR_CHAMPION_CHECK: int = 1

#: Public record (season label -> champion canonical name, final points; None = points not asserted).
KNOWN_CHAMPIONS: dict[int, tuple[str, int | None]] = {
    2012: ("Fluminense", 77), 2013: ("Cruzeiro", 76), 2014: ("Cruzeiro", 80),
    2015: ("Corinthians", 81), 2016: ("Palmeiras", 80), 2017: ("Corinthians", 72),
    2018: ("Palmeiras", 80), 2019: ("Flamengo", 90), 2020: ("Flamengo", 71),
    2021: ("Atletico-MG", 84), 2022: ("Palmeiras", 81), 2023: ("Palmeiras", 70),
    2024: ("Botafogo", 79), 2025: ("Flamengo", None),
}


# ----------------------------------------------------------------------------- #
# External tables audit (src/eda_external.py)
# ----------------------------------------------------------------------------- #
EXTERNAL_DIR: Path = ROOT / "data" / "external" / "williamorim_brasileirao"
EXTERNAL_STATS_FILE: Path = EXTERNAL_DIR / "stats.csv"
EXTERNAL_MATCHES_FILE: Path = EXTERNAL_DIR / "matches.csv"
EXTERNAL_REPORT_FILE: Path = REPORTS_DIR / "eda_external.md"
EXT_AGE_RANGE: tuple[float, float] = (18.0, 40.0)     # plausible mean age of a starting XI (years)
EXT_FOULS_MAX: int = 60                               # fouls by one team in one match above this are implausible
EXT_VALUE_RANGE: tuple[float, float] = (1e4, 5e8)     # plausible starting-XI market value (currency units)
EXT_COVERAGE_MIN_PCT: float = 75.0                    # seasons at least this filled count as "covered"
EXT_FIRST_SEASON: int = 2012                          # overlap with our base starts here


# ----------------------------------------------------------------------------- #
# API-Football match statistics store (src/xg_api.py); git-ignored (data/external/)
# ----------------------------------------------------------------------------- #
EXTERNAL_API_DIR: Path = ROOT / "data" / "external" / "api_football"
API_FREE_SEASONS: tuple[int, ...] = (2022, 2023, 2024)  # seasons the free plan serves
API_DAILY_LIMIT: int = 100
API_MIN_REMAINING: int = 10        # never spend the last requests of the day
API_SLEEP_SECONDS: float = 6.5     # free plan allows 10 requests/minute
