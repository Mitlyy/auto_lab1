from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_lab1.config import ExperimentConfig
from auto_lab1.pipeline import run_experiment


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Manual and Optuna HPO comparison")
    parser.add_argument("--n-trials", type=int, default=32, help="Trials per method")
    parser.add_argument(
        "--n-init",
        type=int,
        default=8,
        help="Initial random trials for manual Bayesian optimization",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=2000,
        help="Candidate pool size for EI maximization",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openml-data-id", type=int, default=37)
    parser.add_argument("--data-dir", type=str, default="auto_lab1/data/openml_cache")
    parser.add_argument("--out-dir", type=str, default="auto_lab1/outputs")
    args = parser.parse_args()
    return ExperimentConfig(
        n_trials=args.n_trials,
        n_init=args.n_init,
        n_candidates=args.n_candidates,
        seed=args.seed,
        openml_data_id=args.openml_data_id,
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
    )


def main() -> None:
    config = parse_args()
    artifacts = run_experiment(config)
    print(f"Artifacts saved to: {artifacts['out_dir']}")
    print(f"Manual summary: {artifacts['manual_summary']}")
    print(f"Optuna summary: {artifacts['optuna_summary']}")


if __name__ == "__main__":
    main()
