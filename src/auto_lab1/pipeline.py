from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .manual_search import run_manual_bayesian_optimization, run_random_search
from .objective import load_data, make_cv
from .optuna_search import run_optuna_search
from .plotting import plot_best_vs_step, plot_importance, plot_space_projection
from .reporting import add_best_so_far, save_markdown_table, summarize
from .search_space import get_search_space


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_experiment(config: ExperimentConfig) -> dict[str, Path]:
    out_dir = config.out_dir
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    X, y, dataset_meta = load_data(
        openml_data_id=config.openml_data_id,
        data_home=config.data_dir,
    )
    cv = make_cv(config.seed)
    space = get_search_space()

    rng_random = np.random.default_rng(config.seed)
    rng_bo = np.random.default_rng(config.seed + 1)

    manual_random = run_random_search(
        n_trials=config.n_trials,
        space=space,
        X=X,
        y=y,
        cv=cv,
        rng=rng_random,
    )
    manual_bo = run_manual_bayesian_optimization(
        n_trials=config.n_trials,
        n_init=config.n_init,
        n_candidates=config.n_candidates,
        space=space,
        X=X,
        y=y,
        cv=cv,
        rng=rng_bo,
        seed=config.seed,
    )
    manual_all = add_best_so_far(pd.concat([manual_random, manual_bo], ignore_index=True))
    manual_all.to_csv(table_dir / "manual_trials.csv", index=False)

    manual_summary = summarize(manual_all)
    manual_summary.to_csv(table_dir / "manual_summary.csv", index=False)
    save_markdown_table(manual_summary, table_dir / "manual_summary.md")

    plot_best_vs_step(
        manual_all,
        fig_dir / "manual_best_vs_step.png",
        "Manual: Bayesian vs Random",
    )
    plot_space_projection(
        manual_all,
        space,
        fig_dir / "manual_space_projection.png",
        "Manual: Tested hyperparameter space (PCA projection)",
    )
    manual_importance = plot_importance(
        manual_all,
        space,
        fig_dir / "manual_importance.png",
        "Manual: Hyperparameter importance",
    )
    manual_importance.to_csv(table_dir / "manual_importance.csv", index=False)

    optuna_random = run_optuna_search(
        sampler_name="random",
        n_trials=config.n_trials,
        X=X,
        y=y,
        cv=cv,
        seed=config.seed,
    )
    optuna_tpe = run_optuna_search(
        sampler_name="tpe",
        n_trials=config.n_trials,
        X=X,
        y=y,
        cv=cv,
        seed=config.seed,
    )
    optuna_all = add_best_so_far(pd.concat([optuna_random, optuna_tpe], ignore_index=True))
    optuna_all.to_csv(table_dir / "optuna_trials.csv", index=False)

    optuna_summary = summarize(optuna_all)
    optuna_summary.to_csv(table_dir / "optuna_summary.csv", index=False)
    save_markdown_table(optuna_summary, table_dir / "optuna_summary.md")

    plot_best_vs_step(
        optuna_all,
        fig_dir / "optuna_best_vs_step.png",
        "Optuna: TPE vs Random",
    )
    plot_space_projection(
        optuna_all,
        space,
        fig_dir / "optuna_space_projection.png",
        "Optuna: Tested hyperparameter space (PCA projection)",
    )
    optuna_importance = plot_importance(
        optuna_all,
        space,
        fig_dir / "optuna_importance.png",
        "Optuna: Hyperparameter importance",
    )
    optuna_importance.to_csv(table_dir / "optuna_importance.csv", index=False)

    report = {
        "dataset": f"openml:{dataset_meta['name']} (id={dataset_meta['data_id']})",
        "dataset_meta": dataset_meta,
        "metric": "4-fold CV accuracy",
        "model": "RandomForestClassifier",
        "n_trials_per_method": config.n_trials,
        "manual_summary": manual_summary.to_dict(orient="records"),
        "optuna_summary": optuna_summary.to_dict(orient="records"),
        "artifacts_dir": str(out_dir),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "out_dir": out_dir,
        "table_dir": table_dir,
        "fig_dir": fig_dir,
        "manual_summary": table_dir / "manual_summary.csv",
        "optuna_summary": table_dir / "optuna_summary.csv",
        "report_json": report_path,
    }
