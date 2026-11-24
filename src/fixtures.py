"""Upcoming Série A fixtures from API-Football (free plan: 100 requests/day).

The key is read from the ``API_FOOTBALL_KEY`` environment variable or the
git-ignored ``.env`` file and is never printed or logged. Every response is
cached on disk per day so repeated runs do not spend quota.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from src import config

BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID_SERIE_A = 71
CACHE_DIR: Path = config.DATA_RAW / "api_football"
DAILY_BUDGET = 100


class ApiError(RuntimeError):
    """Raised for API failures (message never contains the key)."""


def load_key() -> str:
    """Return the API key from the environment or ``.env``."""
    key = os.environ.get("API_FOOTBALL_KEY")
    env_file = config.ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("API_FOOTBALL_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise ApiError("API_FOOTBALL_KEY not set (environment or .env)")
    return key


def _cache_path(path: str, params: dict) -> Path:
    tag = hashlib.sha1(json.dumps([path, sorted(params.items())]).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{date.today().isoformat()}_{path.strip('/').replace('/', '-')}_{tag}.json"


def api_get(path: str, params: dict | None = None, use_cache: bool = True) -> dict:
    """GET ``path`` with query ``params``; cached per day. Returns the parsed JSON body.

    Each network call costs 1 request of the daily budget (``/status`` is free).
    """
    params = params or {}
    cache = _cache_path(path, params)
    if use_cache and cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("errors"):  # errors are cached too, so retries do not burn quota
            raise ApiError(f"API errors for {path}: {cached['errors']} (cached)")
        return cached
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-apisports-key": load_key(),
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            remaining = resp.headers.get("x-ratelimit-requests-remaining")
    except urllib.error.HTTPError as e:
        raise ApiError(f"HTTP {e.code} for {path}") from None
    except urllib.error.URLError as e:
        raise ApiError(f"network error for {path}: {e.reason}") from None
    body["_remaining_requests"] = remaining
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(body))
    if body.get("errors"):
        raise ApiError(f"API errors for {path}: {body['errors']}")
    return body


def parse_fixtures(body: dict) -> pd.DataFrame:
    """Flatten an API-Football ``/fixtures`` response."""
    rows = []
    for item in body.get("response", []):
        rows.append({
            "fixture_id": item["fixture"]["id"],
            "kickoff": pd.to_datetime(item["fixture"]["date"]),
            "status": item["fixture"]["status"]["short"],
            "round": item["league"].get("round"),
            "home_api": item["teams"]["home"]["name"],
            "away_api": item["teams"]["away"]["name"],
        })
    return pd.DataFrame(rows)


def upcoming_fixtures(season: int | None = None, limit: int = 10) -> pd.DataFrame:
    """Next ``limit`` not-started Série A fixtures (1 request, cached for the day)."""
    season = season or pd.Timestamp.today().year
    body = api_get("/fixtures", {"league": LEAGUE_ID_SERIE_A, "season": season, "next": limit})
    return parse_fixtures(body)


def remaining_fixtures(matches: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Matchups still to be played in a double round-robin season (no API needed).

    Série A is a 20-team double round-robin, so every ordered pair
    ``(home, away)`` is played exactly once. The remaining fixtures are the
    ordered pairs of the season's teams that have not appeared yet. Dates are
    unknown. Only valid for the current season, when the table is complete.

    Parameters
    ----------
    matches : pd.DataFrame
        Played matches from ``ingestion.build_interim``.
    season : int, optional
        Season to complete (default: the latest in ``matches``).

    Returns
    -------
    pd.DataFrame
        Columns ``home`` and ``away`` (canonical names), sorted alphabetically.
    """
    season = int(season if season is not None else matches["season"].max())
    s = matches[matches["season"] == season]
    teams = sorted(set(s["home_team"]) | set(s["away_team"]))
    played = set(zip(s["home_team"], s["away_team"]))
    rows = [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]
    return pd.DataFrame(rows, columns=["home", "away"])
