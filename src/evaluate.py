"""Evaluation metrics for 1X2 probabilities and goal-rate predictions.

Probabilities are arrays of shape ``(N, 3)`` ordered ``[home, draw, away]``;
outcomes are integer indices ``0=home, 1=draw, 2=away``.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def outcome_index(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    """Map goals to outcome index (0 home win, 1 draw, 2 away win)."""
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))


def rps_per_match(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Ranked Probability Score per match (lower is better)."""
    onehot = np.eye(3)[outcome]
    diff = np.cumsum(probs, axis=1) - np.cumsum(onehot, axis=1)
    return 0.5 * (diff[:, :2] ** 2).sum(axis=1)


def log_loss_per_match(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Negative log-probability of the realized outcome."""
    p = probs[np.arange(len(outcome)), outcome]
    return -np.log(np.clip(p, EPS, 1.0))


def brier_per_match(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Multiclass Brier score per match."""
    return ((probs - np.eye(3)[outcome]) ** 2).sum(axis=1)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance of goals ``y`` against expected goals ``mu``."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), EPS, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(np.mean(2.0 * (term - (y - mu))))


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Mean of ``a - b`` with a 95% paired bootstrap CI (by match).

    Returns
    -------
    tuple
        ``(mean_diff, ci_low, ci_high)``. Negative values mean ``a`` is better
        for loss-type metrics.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def calibration_table(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> np.ndarray:
    """Reliability table rows ``(mean_pred, empirical_freq, count)`` per probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        if m.any():
            rows.append((p[m].mean(), hit[m].mean(), m.sum()))
    return np.array(rows)
