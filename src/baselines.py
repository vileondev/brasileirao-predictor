"""Baseline models: bookmaker odds, Elo -> Poisson, time-decayed Dixon-Coles.

Every model exposes expected goals (or probabilities) using only matches that
kicked off strictly before the matches being predicted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --------------------------------------------------------------------------- #
# 1. Bookmaker closing odds
# --------------------------------------------------------------------------- #

def odds_probs(df: pd.DataFrame, prefix: str = "AvgC") -> np.ndarray:
    """De-vigged 1X2 probabilities from closing odds (proportional normalization).

    Parameters
    ----------
    df : pd.DataFrame
        Matches with ``{prefix}H``, ``{prefix}D``, ``{prefix}A`` odds columns.
    prefix : str
        ``"AvgC"`` (market average, complete) or ``"PSC"`` (Pinnacle, ~7% missing).

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)`` ordered home/draw/away; rows sum to 1.
    """
    inv = 1.0 / df[[f"{prefix}H", f"{prefix}D", f"{prefix}A"]].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# 2. Elo ratings
# --------------------------------------------------------------------------- #

def compute_elo(
    matches: pd.DataFrame,
    k: float = 20.0,
    home_adv: float = 60.0,
    season_regress: float = 0.25,
    init: float = 1500.0,
    promoted_penalty: float = 75.0,
) -> pd.DataFrame:
    """Pre-match Elo ratings for every match (strictly leak-free).

    Ratings are stored *before* the match result is applied. A team's rating only
    changes through its own matches, so same-day matches of other teams cannot
    leak. At the start of each season ratings regress ``season_regress`` towards
    the mean; teams seen for the first time after the first season start
    ``promoted_penalty`` below the mean.

    Parameters
    ----------
    matches : pd.DataFrame
        Sorted by kickoff with ``season, home_team, away_team, home_goals, away_goals``.

    Returns
    -------
    pd.DataFrame
        Index aligned with ``matches``; columns ``elo_home, elo_away, elo_diff``.
    """
    ratings: dict[str, float] = {}
    first_season = int(matches["season"].iloc[0])
    current_season = None
    elo_h = np.empty(len(matches))
    elo_a = np.empty(len(matches))

    for n, row in enumerate(matches.itertuples(index=False)):
        if row.season != current_season:
            if current_season is not None:
                for t in ratings:
                    ratings[t] = init + (1 - season_regress) * (ratings[t] - init)
            current_season = row.season
        for t in (row.home_team, row.away_team):
            if t not in ratings:
                ratings[t] = init if row.season == first_season else init - promoted_penalty
        rh, ra = ratings[row.home_team], ratings[row.away_team]
        elo_h[n], elo_a[n] = rh, ra

        expected_h = 1.0 / (1.0 + 10 ** (-(rh - ra + home_adv) / 400.0))
        gd = row.home_goals - row.away_goals
        score_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        mult = np.log(abs(gd) + 1.0) + 1.0
        delta = k * mult * (score_h - expected_h)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta

    out = pd.DataFrame({"elo_home": elo_h, "elo_away": elo_a}, index=matches.index)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    return out


class EloPoisson:
    """Poisson goal model driven by ``elo_diff``.

    ``log(lam_home) = a_h + b * d`` and ``log(lam_away) = a_a - b * d`` with
    ``d = elo_diff / 400``.
    """

    def __init__(self) -> None:
        self.params: np.ndarray | None = None

    def fit(self, elo_diff: np.ndarray, home_goals: np.ndarray, away_goals: np.ndarray) -> "EloPoisson":
        d = np.asarray(elo_diff, dtype=float) / 400.0
        hg = np.asarray(home_goals, dtype=float)
        ag = np.asarray(away_goals, dtype=float)

        def nll(p: np.ndarray) -> float:
            lh = np.exp(p[0] + p[2] * d)
            la = np.exp(p[1] - p[2] * d)
            return float((lh - hg * np.log(lh)).sum() + (la - ag * np.log(la)).sum())

        res = minimize(nll, x0=np.array([0.3, 0.0, 0.3]), method="L-BFGS-B")
        self.params = res.x
        return self

    def predict(self, elo_diff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.params is not None, "call fit() first"
        d = np.asarray(elo_diff, dtype=float) / 400.0
        return (np.exp(self.params[0] + self.params[2] * d),
                np.exp(self.params[1] - self.params[2] * d))


# --------------------------------------------------------------------------- #
# 3. Time-decayed Dixon-Coles (attack / defence per team)
# --------------------------------------------------------------------------- #

class DixonColes:
    """Independent-Poisson attack/defence model with exponential time decay.

    ``log(lam_home) = mu + gamma + att[home] - dfn[away]``
    ``log(lam_away) = mu + att[away] - dfn[home]``

    Fitted by weighted Poisson maximum likelihood with a small ridge penalty for
    identifiability. The low-score dependence ``rho`` is estimated separately
    (see :func:`src.probability.fit_rho`). Teams unseen in training get average
    attack/defence.

    Parameters
    ----------
    half_life_days : float
        Weight halves every ``half_life_days`` before the reference date.
    ridge : float
        L2 penalty on attack/defence parameters.
    """

    def __init__(self, half_life_days: float = 730.0, ridge: float = 0.5) -> None:
        self.half_life_days = half_life_days
        self.ridge = ridge
        self.teams: list[str] = []
        self.mu = 0.0
        self.gamma = 0.0
        self.att: dict[str, float] = {}
        self.dfn: dict[str, float] = {}

    def fit(self, train: pd.DataFrame, ref_time: pd.Timestamp) -> "DixonColes":
        """Fit on ``train`` (all matches strictly before ``ref_time``)."""
        teams = sorted(set(train["home_team"]) | set(train["away_team"]))
        tidx = {t: i for i, t in enumerate(teams)}
        nt = len(teams)
        h = train["home_team"].map(tidx).to_numpy()
        a = train["away_team"].map(tidx).to_numpy()
        hg = train["home_goals"].to_numpy(dtype=float)
        ag = train["away_goals"].to_numpy(dtype=float)
        age = (ref_time - train["kickoff"]).dt.total_seconds().to_numpy() / 86400.0
        w = 0.5 ** (np.clip(age, 0, None) / self.half_life_days)

        def unpack(x: np.ndarray):
            return x[0], x[1], x[2:2 + nt], x[2 + nt:]

        def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
            mu, gamma, att, dfn = unpack(x)
            lh = np.exp(mu + gamma + att[h] - dfn[a])
            la = np.exp(mu + att[a] - dfn[h])
            nll = (w * (lh - hg * np.log(lh))).sum() + (w * (la - ag * np.log(la))).sum()
            nll += self.ridge * ((att ** 2).sum() + (dfn ** 2).sum())
            rh, ra = w * (hg - lh), w * (ag - la)  # d(-nll)/d(log lam)
            g = np.zeros_like(x)
            g[0] = -(rh.sum() + ra.sum())
            g[1] = -rh.sum()
            g[2:2 + nt] = -(np.bincount(h, rh, nt) + np.bincount(a, ra, nt)) + 2 * self.ridge * att
            g[2 + nt:] = (np.bincount(a, rh, nt) + np.bincount(h, ra, nt)) + 2 * self.ridge * dfn
            return float(nll), g

        x0 = np.zeros(2 + 2 * nt)
        x0[0] = np.log(max(np.average(np.r_[hg, ag], weights=np.r_[w, w]), 0.1))
        res = minimize(fun, x0, jac=True, method="L-BFGS-B")
        mu, gamma, att, dfn = unpack(res.x)
        self.teams, self.mu, self.gamma = teams, float(mu), float(gamma)
        self.att = dict(zip(teams, att))
        self.dfn = dict(zip(teams, dfn))
        return self

    def to_dict(self) -> dict:
        """JSON-serializable parameters."""
        return {"half_life_days": self.half_life_days, "ridge": self.ridge, "mu": self.mu,
                "gamma": self.gamma, "att": self.att, "dfn": self.dfn}

    @classmethod
    def from_dict(cls, d: dict) -> "DixonColes":
        """Rebuild a fitted model from :meth:`to_dict` output."""
        m = cls(d["half_life_days"], d["ridge"])
        m.mu, m.gamma, m.att, m.dfn = d["mu"], d["gamma"], dict(d["att"]), dict(d["dfn"])
        m.teams = sorted(m.att)
        return m

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals ``(lam_home, lam_away)`` for the matches in ``df``."""
        ah = df["home_team"].map(lambda t: self.att.get(t, 0.0)).to_numpy()
        dh = df["home_team"].map(lambda t: self.dfn.get(t, 0.0)).to_numpy()
        aa = df["away_team"].map(lambda t: self.att.get(t, 0.0)).to_numpy()
        da = df["away_team"].map(lambda t: self.dfn.get(t, 0.0)).to_numpy()
        return (np.exp(self.mu + self.gamma + ah - da), np.exp(self.mu + aa - dh))
