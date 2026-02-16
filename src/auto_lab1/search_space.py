from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] | None = None


def get_search_space() -> list[ParamSpec]:
    return [
        ParamSpec("n_estimators", "int", 50, 350),
        ParamSpec("max_depth", "int", 2, 30),
        ParamSpec("min_samples_split", "int", 2, 20),
        ParamSpec("min_samples_leaf", "int", 1, 15),
        ParamSpec("max_features", "float", 0.2, 1.0),
        ParamSpec("bootstrap", "cat", choices=(False, True)),
        ParamSpec("criterion", "cat", choices=("gini", "entropy", "log_loss")),
    ]


def sample_random_params(space: list[ParamSpec], rng: np.random.Generator) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for spec in space:
        if spec.kind == "int":
            assert spec.low is not None and spec.high is not None
            params[spec.name] = int(rng.integers(int(spec.low), int(spec.high) + 1))
        elif spec.kind == "float":
            assert spec.low is not None and spec.high is not None
            params[spec.name] = float(rng.uniform(float(spec.low), float(spec.high)))
        else:
            assert spec.choices is not None
            idx = int(rng.integers(0, len(spec.choices)))
            params[spec.name] = spec.choices[idx]
    return params


def params_to_vector(params: dict[str, Any], space: list[ParamSpec]) -> np.ndarray:
    encoded: list[float] = []
    for spec in space:
        value = params[spec.name]
        if spec.kind in {"int", "float"}:
            assert spec.low is not None and spec.high is not None
            lo = float(spec.low)
            hi = float(spec.high)
            encoded.append((float(value) - lo) / (hi - lo))
        else:
            assert spec.choices is not None
            idx = spec.choices.index(value)
            denom = max(1, len(spec.choices) - 1)
            encoded.append(idx / denom)
    return np.asarray(encoded, dtype=float)


def vector_to_params(vector: np.ndarray, space: list[ParamSpec]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for value, spec in zip(vector, space):
        x = float(np.clip(value, 0.0, 1.0))
        if spec.kind == "int":
            assert spec.low is not None and spec.high is not None
            lo = int(spec.low)
            hi = int(spec.high)
            params[spec.name] = int(np.clip(round(lo + x * (hi - lo)), lo, hi))
        elif spec.kind == "float":
            assert spec.low is not None and spec.high is not None
            lo = float(spec.low)
            hi = float(spec.high)
            params[spec.name] = float(lo + x * (hi - lo))
        else:
            assert spec.choices is not None
            last = max(0, len(spec.choices) - 1)
            idx = int(np.clip(round(x * last), 0, last))
            params[spec.name] = spec.choices[idx]
    return params


def row_to_params(row: pd.Series, space: list[ParamSpec]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for spec in space:
        value = row[spec.name]
        if spec.kind == "int":
            params[spec.name] = int(round(float(value)))
        elif spec.kind == "float":
            params[spec.name] = float(value)
        else:
            assert spec.choices is not None
            if isinstance(spec.choices[0], bool):
                if isinstance(value, str):
                    params[spec.name] = value.strip().lower() in {"1", "true", "yes"}
                else:
                    params[spec.name] = bool(value)
            else:
                params[spec.name] = str(value)
    return params

