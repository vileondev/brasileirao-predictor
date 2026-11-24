"""Scoreline probability matrix (Poisson + Dixon-Coles) and derived markets."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from src import config


def scoreline_matrix(
    lam_home: np.ndarray | float,
    lam_away: np.ndarray | float,
    rho: float = 0.0,
    max_goals: int = config.MAX_GOALS,
) -> np.ndarray:
    """Dixon-Coles scoreline probability matrices.

    Parameters
    ----------
    lam_home, lam_away : array_like
        Expected goals, shape ``(N,)`` or scalars.
    rho : float
        Dixon-Coles dependence parameter (``0`` gives independent Poisson).
    max_goals : int
        Grid covers ``0..max_goals`` goals per side.

    Returns
    -------
    np.ndarray
        Shape ``(N, K, K)`` with ``K = max_goals + 1``; entry ``[n, i, j]`` is
        ``P(home=i, away=j)``. Each matrix is normalized to sum to 1.
    """
    lh = np.atleast_1d(np.asarray(lam_home, dtype=float))
    la = np.atleast_1d(np.asarray(lam_away, dtype=float))
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[None, :], lh[:, None])
    pa = poisson.pmf(goals[None, :], la[:, None])
    mat = ph[:, :, None] * pa[:, None, :]
    mat[:, 0, 0] *= np.clip(1.0 - lh * la * rho, 1e-10, None)
    mat[:, 1, 0] *= np.clip(1.0 + la * rho, 1e-10, None)
    mat[:, 0, 1] *= np.clip(1.0 + lh * rho, 1e-10, None)
    mat[:, 1, 1] *= np.clip(1.0 - rho, 1e-10, None)
    mat /= mat.sum(axis=(1, 2), keepdims=True)
    return mat


def outcome_probs(mat: np.ndarray) -> np.ndarray:
    """1X2 probabilities ``[home, draw, away]`` from scoreline matrices ``(N, K, K)``."""
    k = mat.shape[-1]
    i, j = np.indices((k, k))
    home = mat[:, i > j].sum(axis=1)
    draw = mat[:, i == j].sum(axis=1)
    away = mat[:, i < j].sum(axis=1)
    return np.stack([home, draw, away], axis=1)


def market_probs(mat: np.ndarray) -> dict[str, np.ndarray]:
    """Over 2.5 goals and both-teams-to-score probabilities, shape ``(N,)`` each."""
    k = mat.shape[-1]
    i, j = np.indices((k, k))
    return {
        "over_2_5": mat[:, (i + j) >= 3].sum(axis=1),
        "btts": mat[:, (i >= 1) & (j >= 1)].sum(axis=1),
    }


def top_scorelines(mat_single: np.ndarray, n: int = 3) -> list[tuple[int, int, float]]:
    """Top-``n`` most likely scorelines of one ``(K, K)`` matrix as ``(home, away, p)``."""
    flat = np.argsort(mat_single, axis=None)[::-1][:n]
    k = mat_single.shape[1]
    return [(int(f // k), int(f % k), float(mat_single[f // k, f % k])) for f in flat]


def fit_rho(
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    bounds: tuple[float, float] = (-0.3, 0.1),
) -> float:
    """Maximum-likelihood estimate of the Dixon-Coles ``rho``.

    Should be fitted on out-of-fold / training predictions, never on the test set.
    """
    hg = np.minimum(home_goals.astype(int), config.MAX_GOALS)
    ag = np.minimum(away_goals.astype(int), config.MAX_GOALS)
    idx = np.arange(len(hg))

    def nll(rho: float) -> float:
        mat = scoreline_matrix(lam_home, lam_away, rho)
        return float(-np.log(mat[idx, hg, ag] + 1e-300).sum())

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)
