from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import StratifiedKFold

from .objective import evaluate_params
from .search_space import ParamSpec, params_to_vector, sample_random_params, vector_to_params


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_value: float,
    xi: float = 0.01,
) -> np.ndarray:
    sigma_safe = np.maximum(sigma, 1e-12)
    imp = mu - best_value - xi
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def run_random_search(
    n_trials: int,
    space: list[ParamSpec],
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step in range(1, n_trials + 1):
        params = sample_random_params(space, rng)
        score = evaluate_params(params, X, y, cv)
        rows.append({"method": "manual_random", "step": step, "score": score, **params})
    return pd.DataFrame(rows)


def run_manual_bayesian_optimization(
    n_trials: int,
    n_init: int,
    n_candidates: int,
    space: list[ParamSpec],
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    rng: np.random.Generator,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    observed_x: list[np.ndarray] = []
    observed_y: list[float] = []

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=np.ones(len(space)), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=seed,
    )

    for step in range(1, n_trials + 1):
        if len(observed_x) < n_init:
            params = sample_random_params(space, rng)
        else:
            x_train = np.vstack(observed_x)
            y_train = np.asarray(observed_y, dtype=float)
            try:
                gp.fit(x_train, y_train)
                candidates = rng.random((n_candidates, len(space)))
                mu, sigma = gp.predict(candidates, return_std=True)
                ei = expected_improvement(mu, sigma, best_value=float(np.max(y_train)))
                best_idx = int(np.argmax(ei))
                params = vector_to_params(candidates[best_idx], space)
            except Exception:
                params = sample_random_params(space, rng)

        score = evaluate_params(params, X, y, cv)
        observed_x.append(params_to_vector(params, space))
        observed_y.append(score)
        rows.append({"method": "manual_bo", "step": step, "score": score, **params})

    return pd.DataFrame(rows)

