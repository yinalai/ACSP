"""Random Report Attack (RRA) with extra users on the original Symbols protocol.

Example:
    python rra_symbols.py --data-dir path/to/symbols --epsilon 1 --beta 0.10 \
        --runs 1 --output rra_symbols_result.json

Symbols retains the source GRR length mechanism, corrected per-level C
sampling, weighted SAX/DTW matching, and match_clustering-style Accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import attack_symbols as engine

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rra_shared import main as _main
from rra_shared import paired_rra_experiment, rra_report


def random_report(data: engine.Dataset, cfg: engine.Config,
                  run_seed: int, evaluate_test: bool = True) -> dict:
    return rra_report(engine, data, cfg, run_seed, symbols=True,
                      evaluate_test=evaluate_test)


def paired_experiment(data: engine.Dataset, cfg: engine.Config, runs: int,
                      seed: int, trial_start: int = 0,
                      checkpoint_dir: Path | None = None,
                      resume: bool = False) -> dict:
    return paired_rra_experiment(
        engine, data, cfg, symbols=True, runs=runs, seed=seed,
        trial_start=trial_start, checkpoint_dir=checkpoint_dir, resume=resume)


if __name__ == "__main__":
    _main(engine, symbols=True)
