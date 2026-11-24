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
