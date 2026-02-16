from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from sklearn.ensemble import RandomForestRegressor

from .search_space import ParamSpec, params_to_vector, row_to_params


def add_best_so_far(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["method", "step"]).copy()
    out["best_so_far"] = out.groupby("method")["score"].cummax()
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method, group in df.groupby("method"):
        g = group.sort_values("step")
        best_idx = int(g["score"].idxmax())
        rows.append(
            {
                "method": method,
                "best_score": float(g.loc[best_idx, "score"]),
                "best_step": int(g.loc[best_idx, "step"]),
                "mean_score": float(g["score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("best_score", ascending=False).reset_index(drop=True)


def encode_trials(df: pd.DataFrame, space: list[ParamSpec]) -> np.ndarray:
    encoded = [
        params_to_vector(row_to_params(row, space), space) for _, row in df.iterrows()
    ]
    return np.vstack(encoded)


def compute_importance(df: pd.DataFrame, space: list[ParamSpec]) -> pd.DataFrame:
    X_meta = encode_trials(df, space)
    y_meta = df["score"].to_numpy()
    model = RandomForestRegressor(n_estimators=600, random_state=42)
    model.fit(X_meta, y_meta)
    imp_df = pd.DataFrame(
        {
            "param": [spec.name for spec in space],
            "importance": model.feature_importances_,
        }
    )
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_markdown_table(df: pd.DataFrame, path: str, float_fmt: str = ".5f") -> None:
    out = df.copy()
    for col in out.columns:
        if is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: format(float(x), float_fmt))

    cols = list(out.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, separator]
    for _, row in out.iterrows():
        vals = [str(row[col]) for col in cols]
        lines.append("| " + " | ".join(vals) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
