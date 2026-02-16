from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    n_trials: int = 32
    n_init: int = 8
    n_candidates: int = 2000
    seed: int = 42
    openml_data_id: int = 37
    data_dir: Path = Path("auto_lab1/data/openml_cache")
    out_dir: Path = Path("auto_lab1/outputs")
