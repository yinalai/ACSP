"""Random Report Attack (RRA) with extra users on the original Trace protocol.

Example:
    python rra_trace.py --data-dir path/to/trace --epsilon 1 --beta 0.10 \
        --runs 1 --output rra_trace_result.json

No genuine report is replaced. All injected identities are preassigned to
A/B/C-layer/D in the proportions used by the normal protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import attack_trace as engine

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rra_shared import main as _main
from rra_shared import paired_rra_experiment, rra_report


def random_report(data: engine.Dataset, cfg: engine.Config,
                  run_seed: int, evaluate_test: bool = True) -> dict:
    return rra_report(engine, data, cfg, run_seed, symbols=False,
                      evaluate_test=evaluate_test)


def paired_experiment(data: engine.Dataset, cfg: engine.Config, runs: int,
                      seed: int, trial_start: int = 0,
                      checkpoint_dir: Path | None = None,
                      resume: bool = False) -> dict:
    return paired_rra_experiment(
        engine, data, cfg, symbols=False, runs=runs, seed=seed,
        trial_start=trial_start, checkpoint_dir=checkpoint_dir, resume=resume)


if __name__ == "__main__":
    _main(engine, symbols=False)
