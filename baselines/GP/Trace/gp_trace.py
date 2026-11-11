from __future__ import annotations
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
_ROOT = str(Path(__file__).resolve().parent.parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import attack_trace as engine

from gp_shared import gp_report, main as _main, paired_gp_experiment


def greedy_report(data: engine.Dataset, cfg: engine.Config, run_seed: int,
                  *, knowledge_size: int = 0, seed_candidates: int = 32,
                  evaluate_test: bool = True) -> dict:
    return gp_report(engine, data, cfg, run_seed, symbols=False,
                     knowledge_size=knowledge_size, seed_candidates=seed_candidates,
                     evaluate_test=evaluate_test)


def paired_experiment(data: engine.Dataset, cfg: engine.Config, runs: int,
                      seed: int, trial_start: int = 0,
                      checkpoint_dir: Path | None = None, resume: bool = False,
                      knowledge_size: int = 0, seed_candidates: int = 32) -> dict:
    return paired_gp_experiment(engine, data, cfg, symbols=False, runs=runs,
                                seed=seed, trial_start=trial_start,
                                checkpoint_dir=checkpoint_dir, resume=resume,
                                knowledge_size=knowledge_size,
                                seed_candidates=seed_candidates)


if __name__ == "__main__":
    _main(engine, symbols=False)
