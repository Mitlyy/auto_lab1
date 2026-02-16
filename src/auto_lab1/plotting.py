from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .reporting import compute_importance, encode_trials
from .search_space import ParamSpec


LABELS = {
    "manual_random": "Manual Random Search",
    "manual_bo": "Manual Bayesian (GP + EI)",
    "optuna_random": "Optuna RandomSampler",
    "optuna_tpe": "Optuna TPESampler",
}


def plot_best_vs_step(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted(df["method"].unique()):
        part = df[df["method"] == method].sort_values("step")
        ax.plot(
            part["step"],
            part["best_so_far"],
            label=LABELS.get(method, method),
            linewidth=2,
            marker="o",
            markersize=3,
        )
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Best CV accuracy so far")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_space_projection(
    df: pd.DataFrame,
    space: list[ParamSpec],
    path: Path,
    title: str,
) -> None:
    methods = sorted(df["method"].unique())
    encoded = encode_trials(df, space)
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(encoded)

    plot_df = df.copy()
    plot_df["pc1"] = proj[:, 0]
    plot_df["pc2"] = proj[:, 1]
    vmin = float(plot_df["score"].min())
    vmax = float(plot_df["score"].max())

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(7 * len(methods), 5),
        sharex=True,
        sharey=True,
    )
    if len(methods) == 1:
        axes = np.asarray([axes])

    scatter = None
    for ax, method in zip(axes, methods):
        part = plot_df[plot_df["method"] == method]
        scatter = ax.scatter(
            part["pc1"],
            part["pc2"],
            c=part["score"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=55,
            edgecolor="black",
            linewidths=0.3,
        )
        ax.set_title(LABELS.get(method, method))
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.grid(alpha=0.2)

    assert scatter is not None
    fig.colorbar(scatter, ax=axes.tolist(), label="CV accuracy", shrink=0.9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_importance(
    df: pd.DataFrame,
    space: list[ParamSpec],
    path: Path,
    title: str,
) -> pd.DataFrame:
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5))
    if len(methods) == 1:
        axes = np.asarray([axes])

    all_importances: list[pd.DataFrame] = []
    for ax, method in zip(axes, methods):
        part = df[df["method"] == method]
        imp_df = compute_importance(part, space)
        imp_df["method"] = method
        all_importances.append(imp_df)
        ax.barh(imp_df["param"], imp_df["importance"], color="#33658A")
        ax.set_title(LABELS.get(method, method))
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return pd.concat(all_importances, ignore_index=True)

