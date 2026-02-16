from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import RandomSampler, TPESampler
from sklearn.model_selection import StratifiedKFold

from .objective import evaluate_params


optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_search(
    sampler_name: str,
    n_trials: int,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    seed: int,
) -> pd.DataFrame:
    if sampler_name == "random":
        sampler = RandomSampler(seed=seed)
        method_name = "optuna_random"
    elif sampler_name == "tpe":
        sampler = TPESampler(seed=seed, multivariate=True)
        method_name = "optuna_tpe"
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 350),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }
        return evaluate_params(params, X, y, cv)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.value is None:
            continue
        params = trial.params
        rows.append(
            {
                "method": method_name,
                "step": len(rows) + 1,
                "score": float(trial.value),
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "min_samples_split": int(params["min_samples_split"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "max_features": float(params["max_features"]),
                "bootstrap": bool(params["bootstrap"]),
                "criterion": str(params["criterion"]),
            }
        )
    return pd.DataFrame(rows)

