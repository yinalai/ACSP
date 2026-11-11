from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, exp
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from scipy.stats import binom


def dataset_classes(dataset: str) -> tuple[int, ...]:
    """Never infer the expected classes from surviving attack targets."""
    key = dataset.lower()
    if key == "trace":
        return (1, 2, 3)
    if key == "symbols":
        return (1, 2, 3, 4, 5, 6)
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def strict_round_success(dataset: str, active_targets: Mapping[int, Any],
                         representatives: Sequence[Any]) -> tuple[bool, dict[str, bool]]:
    """A complete round succeeds only if *every* expected class is hit."""
    classes = dataset_classes(dataset)
    hits = {str(c): (active_targets.get(c) is not None and
                     len(representatives) >= c and
                     representatives[c - 1] == active_targets[c])
            for c in classes}
    return bool(set(active_targets) == set(classes) and all(hits.values())), hits


def normalized_estimates(values: Mapping[Hashable, float]
                         ) -> tuple[dict[Hashable, float], dict[str, Any]]:
    """Cao et al.'s min-shift/sum normalization, preserving key order.

    In a degenerate all-equal/empty domain, keep original scores so a zero
    cannot be turned into a fabricated positive value.
    """
    if not values:
        return {}, {"applied": False, "reason": "empty_domain", "dimension": 0}
    keys = list(values)
    raw = np.asarray([float(values[k]) for k in keys], dtype=float)
    if not np.all(np.isfinite(raw)):
        return dict(values), {"applied": False, "reason": "nonfinite", "dimension": len(keys)}
    shifted = raw - float(raw.min())
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= 0:
        return dict(values), {"applied": False, "reason": "all_equal", "dimension": len(keys)}
    return ({k: float(v) for k, v in zip(keys, shifted / total)},
            {"applied": True, "reason": None, "dimension": len(keys),
             "min_raw": float(raw.min()), "max_raw": float(raw.max())})


def normalized_oue_estimates(counts: Mapping[Hashable, int], honest_or_total_n: int,
                             epsilon: float
                             ) -> tuple[dict[Hashable, float], dict[str, Any]]:
    """Debias the complete joint OUE domain, then normalize exactly once."""
    if honest_or_total_n <= 0 or epsilon <= 0:
        return dict(counts), {"applied": False, "reason": "no_reports", "dimension": len(counts)}
    p, q = 0.5, 1.0 / (exp(epsilon) + 1.0)
    estimates = {cell: (float(n) / honest_or_total_n - q) / (p - q)
                 for cell, n in counts.items()}
    result, diag = normalized_estimates(estimates)
    diag["domain"] = "complete_joint_oue"
    return result, diag


def randk_keep_count(*, dataset: str, candidate_count: int, class_count: int,
                     extra_slots: int, run_seed: int, stage: str, position: int
                     ) -> tuple[int, dict[str, Any]]:
    """Randomize candidate retention; do not mutate the class-count top_k.

    The original server retains 2*K (Trace) or 3*K (Symbols) candidates.
    The defense uniformly draws the actual retention size from
    [K, min(available, (factor+extra_slots)*K)]. Thus it may shrink as far as
    K but never below the number of classes when at least K candidates exist;
    expansion is capped to (c+1)*K by default (extra_slots=1), or a larger
    multiple if explicitly requested; c is 2 (Trace) or 3 (Symbols).
    Separate stable stage/level seeds couple clean and attacked server rules.
    """
    factor = {"trace": 2, "symbols": 3}[dataset.lower()]
    if stage not in ("B", "C"):
        raise ValueError("Rand(k) only applies to B/C candidate selection")
    if class_count < 1 or candidate_count < 0 or extra_slots < 1 or position < 0:
        raise ValueError("Invalid Rand(k) configuration")
    base = factor * class_count
    maximum_cap = (factor + extra_slots) * class_count
    maximum = min(candidate_count, maximum_cap)
    choices = list(range(class_count, maximum + 1))
    if len(choices) < 2:
        actual = min(candidate_count, base)
        return actual, {
            "stage": stage, "position": position, "base_keep": base,
            "class_count": class_count,
            "shrink_floor": class_count, "expansion_cap": maximum_cap,
            "actual_keep": actual, "sampled_candidate_k": actual,
            "feasible_k_values": len(choices), "randomized": False,
            "reason": "fewer_than_two_feasible_retention_sizes"}
    
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(run_seed) & 0xFFFFFFFF, 0xD3F3E5, 1 if stage == "B" else 2, position]))
    selected_keep = int(rng.choice(choices))
    return selected_keep, {
        "stage": stage, "position": position, "base_keep": base,
        "class_count": class_count,
        "shrink_floor": class_count, "expansion_cap": maximum_cap,
        "actual_keep": selected_keep, "sampled_candidate_k": selected_keep,
        "feasible_k_values": len(choices), "randomized": True,
        "reason": None}


@dataclass(frozen=True)
class DiffstatsResult:
    removed_indices: tuple[int, ...]
    original_chi_square: float
    remaining_chi_square: float
    checked_candidates: int
    dimension: int
    top_l: int
    reason: str


def _chi_squared(hist: np.ndarray, n: int, probabilities: np.ndarray) -> float:
    """Pearson statistic for the theoretical OUE Hamming-weight distribution."""
    expected = n * probabilities
    
    
    return float(np.sum((hist - expected)**2 / np.maximum(expected, 1e-12)))


def diffstats_oue(reports: Sequence[set[Hashable]],
                  domain: Sequence[Hashable], epsilon: float, top_l: int = 6
                  ) -> DiffstatsResult:
    """Diffstats Algorithm-1-style two-statistic search on complete OUE reports.

    Implements the paper's E_sq (Hamming-weight discrepancy), iteratively
    constructed suspicious population, top-L common supports and nonempty
    support-subset enumeration, then E_freq (chi-squared) refinement.
    Paper's binomial approximation is used for X, not per-class one-hot PMFs.
    The baseline no-removal score must be improved before filtering anything.
    """
    d, n = len(domain), len(reports)
    if d == 0 or n == 0:
        return DiffstatsResult((), 0.0, 0.0, 0, d, top_l, "empty")
    if epsilon <= 0 or top_l < 1:
        raise ValueError("Diffstats requires epsilon>0 and top_l>=1")
    if len(set(domain)) != d:
        raise ValueError("Joint OUE domain must contain unique cells")
    index = {cell: i for i, cell in enumerate(domain)}
    bits = np.zeros((n, d), dtype=np.bool_)
    for row, report in enumerate(reports):
        for cell in report:
            if cell not in index:
                raise ValueError(f"Report cell {cell!r} is outside the joint OUE domain")
            bits[row, index[cell]] = True
    weights = bits.sum(axis=1).astype(np.int64)
    observed = np.bincount(weights, minlength=d+1).astype(float)
    p, q = 0.5, 1.0 / (exp(epsilon) + 1.0)
    mean_p = (p + (d - 1) * q) / d
    probabilities = binom.pmf(np.arange(d+1), d, mean_p)
    baseline = _chi_squared(observed, n, probabilities)
    errors = (observed - n * probabilities)**2
    active = set(range(d + 1))
    best_score = baseline
    best_removed: tuple[int, ...] = ()
    checked = 0
    for delta in sorted(range(d+1), key=lambda k: (float(errors[k]), k)):
        active.discard(delta)
        suspicious = np.flatnonzero(np.isin(weights, list(active)))
        if not len(suspicious):
            continue
        
        supports = bits[suspicious].sum(axis=0)
        L = min(top_l, d)
        top_cells = sorted(range(d), key=lambda i: (-int(supports[i]), i))[:L]
        
        local_bits = bits[suspicious][:, top_cells]
        for size in range(1, L + 1):
            for subset in combinations(range(L), size):
                flagged = suspicious[np.all(local_bits[:, subset], axis=1)]
                checked += 1
                if not len(flagged) or len(flagged) == n:
                    continue
                retained_n = n - len(flagged)
                retained_hist = (observed -
                                 np.bincount(weights[flagged], minlength=d+1))
                statistic = _chi_squared(retained_hist, retained_n, probabilities)
                if statistic < best_score - 1e-10:
                    best_score = statistic
                    best_removed = tuple(int(i) for i in flagged)
    return DiffstatsResult(best_removed, baseline, best_score, checked, d, top_l,
                           "filtered" if best_removed else "no_detection")


def four_way_accuracy_pair(clean_none: Mapping[str, Any],
                           attack_none: Mapping[str, Any],
                           clean_defense: Mapping[str, Any],
                           attack_defense: Mapping[str, Any]
                           ) -> dict[str, Any]:
    """Per-seed mitigation requires FOUR valid accuracy observations."""
    arms = (clean_none, attack_none, clean_defense, attack_defense)
    valid = all(x.get("protocol_valid", True) and
                x.get("accuracy") is not None and
                np.isfinite(float(x["accuracy"])) for x in arms)
    if not valid:
        return {"paired_accuracy_valid": False, "mitigation": None,
                "loss_none": None, "loss_defense": None}
    lc = float(clean_none["accuracy"]) - float(attack_none["accuracy"])
    ld = float(clean_defense["accuracy"]) - float(attack_defense["accuracy"])
    return {"paired_accuracy_valid": True, "mitigation": lc - ld,
            "loss_none": lc, "loss_defense": ld,
            "clean_utility_change": float(clean_defense["accuracy"]) -
                                    float(clean_none["accuracy"])}
