from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


def load_data(
    openml_data_id: int,
    data_home: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data_home.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(data_id=openml_data_id, as_frame=False, data_home=str(data_home))
    X = np.asarray(dataset.data, dtype=np.float32)
    y_raw = np.asarray(dataset.target)
    y = LabelEncoder().fit_transform(y_raw).astype(np.int64)

    name = str(getattr(dataset, "DESCR", "")).split("\n", 1)[0] or f"openml_{openml_data_id}"
    if hasattr(dataset, "details") and isinstance(dataset.details, dict):
        name = str(dataset.details.get("name", name))

    meta = {
        "source": "openml",
        "data_id": int(openml_data_id),
        "name": name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return X, y, meta


def make_cv(seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)


def evaluate_params(
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> float:
    model = RandomForestClassifier(random_state=42, n_jobs=1, **params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(np.mean(scores))
