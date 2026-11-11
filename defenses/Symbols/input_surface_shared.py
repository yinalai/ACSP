
from __future__ import annotations

from itertools import permutations
from math import ceil, exp, floor, sqrt
import time
from typing import Any, Callable, Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm


def pairwise_variance(p_t: float, p_r: float, categorical: bool) -> float:
    """Variance of one report's target-minus-rival vote difference."""
    if categorical:
        return max(0.0, p_t + p_r - (p_t - p_r) ** 2)
    return max(0.0, p_t * (1 - p_t) + p_r * (1 - p_r))


def _deadline_check(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Input-poisoning planning deadline exhausted")


def minimum_input_allocation(
    items: tuple[Any, ...], inputs: dict[Any, np.ndarray],
    honest_means: dict[Any, float], honest_users: int,
    constraints: Iterable[tuple[Any, Any]], z: float,
    max_users: int, *, categorical: bool = True,
    deadline: float | None = None,
) -> dict[Any, int] | None:
    """Model-conditional minimum-input LP, reweighted for covariance.

    Each constraint requires target's expected count to beat rival's by
    z standard deviations plus one count.  GRR/EM report categories are
    mutually exclusive.  OUE bits are conditionally independent instead.
    Honest sampling is projected as an independent, identical mixture from
    the shared planning knowledge (the same approximation used for means);
    we do NOT assume that candidate counts within one report are independent.
    The LP is solved in real numbers, then conservatively rounded and tested.
    """
    _deadline_check(deadline)
    if max_users < 0:
        return None
    names = tuple(inputs)
    if not names:
        return None
    constraints = tuple(dict.fromkeys(constraints))
    if not constraints:
        return {}
    index = {name: i for i, name in enumerate(items)}
    if any(t not in index or r not in index or t == r for t, r in constraints):
        return None
    
    
    probs = np.asarray([inputs[name] for name in names], dtype=float)
    if probs.shape != (len(names), len(items)) or not np.isfinite(probs).all():
        raise ValueError("Malformed input randomization probability vectors")
    if (probs < -1e-12).any() or (probs > 1 + 1e-12).any():
        raise ValueError("Input randomization probabilities outside [0,1]")
    if categorical and not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("GRR/EM must emit exactly one candidate")
    honest_n = max(0, int(honest_users))
    means = np.array([float(honest_means.get(v, 0.0)) for v in items])
    indices_t = np.fromiter((index[t] for t, _ in constraints), dtype=int)
    indices_r = np.fromiter((index[r] for _, r in constraints), dtype=int)
    margin = means[indices_t] - means[indices_r]
    if honest_n:
        ht = np.clip(means[indices_t] / honest_n, 0.0, 1.0)
        hr = np.clip(means[indices_r] / honest_n, 0.0, 1.0)
    else:
        ht = hr = np.zeros(len(constraints))
    honest_var = honest_n * np.maximum(
        0.0, ht + hr - (ht - hr) ** 2 if categorical else
        ht * (1 - ht) + hr * (1 - hr))
    
    
    if np.all(margin >= 1 + z * np.sqrt(honest_var) - 1e-8):
        return {}
    if max_users == 0:
        return None
    pt, pr = probs[:, indices_t].T, probs[:, indices_r].T
    gain = pt - pr
    malicious_var = np.maximum(
        0.0, pt + pr - (pt - pr) ** 2 if categorical else
        pt * (1 - pt) + pr * (1 - pr))
    chosen = np.zeros(len(names), dtype=int)

    def valid(counts: np.ndarray) -> bool:
        means_delta = margin + gain @ counts
        variances = honest_var + malicious_var @ counts
        return bool(np.all(means_delta >= 1 + z * np.sqrt(np.maximum(0.0, variances)) - 1e-8))

    if valid(chosen):
        return {}
    for _ in range(16):
        _deadline_check(deadline)
        rhs = 1 + z * np.sqrt(np.maximum(0.0, honest_var + malicious_var @ chosen)) - margin
        remaining = max_users - int(chosen.sum())
        if remaining < 0:
            return None
        
        need = rhs - gain @ chosen
        if np.all(need <= 1e-9):
            if valid(chosen):
                return {name: int(count) for name, count in zip(names, chosen) if count}
        options = {"time_limit": max(0.01, deadline - time.monotonic())} if deadline else None
        model = linprog(np.ones(len(names)), A_ub=np.vstack((-gain, np.ones(len(names)))),
                        b_ub=np.r_[-need, float(remaining)], bounds=(0, None),
                        method="highs", options=options)
        _deadline_check(deadline)
        if not model.success or model.x is None:
            return None
        addition = np.maximum(0, np.ceil(model.x - 1e-8)).astype(int)
        if not addition.any() and not valid(chosen):
            
            weakest = int(np.argmax(need))
            best = int(np.argmax(gain[weakest]))
            if gain[weakest, best] <= 0:
                return None
            addition[best] += 1
        chosen += addition
        if int(chosen.sum()) > max_users:
            return None
        if valid(chosen):
            return {name: int(count) for name, count in zip(names, chosen) if count}
    return None


def categorical_topk_inputs(
    items: tuple[str, ...], protected: Iterable[str], honest_means: dict[str, float],
    honest_users: int, topk: int, z: float, max_users: int,
    inputs: dict[str, np.ndarray], *, deadline: float | None = None,
) -> dict[str, int] | None:
    """Choose legal input counts sufficient for original Top-k membership.

    The original Top-k rule is unchanged.  The highest `topk-len(protected)`
    outside candidates may outrank protected ones; all remaining outsiders
    are used in the sufficient LP constraints (a conservative approximation).
    """
    kept = tuple(dict.fromkeys(protected))
    if not kept or not set(kept) <= set(items) or len(kept) > topk:
        return {} if not kept else None
    outside = [v for v in items if v not in kept]
    if len(outside) <= topk - len(kept):
        return {}
    
    
    outside.sort(key=lambda v: -honest_means.get(v, 0.0))
    outsiders_to_beat = outside[topk - len(kept):]
    constraints = tuple((t, r) for t in kept for r in outsiders_to_beat)
    return minimum_input_allocation(items, inputs, honest_means, honest_users,
                                    constraints, z, max_users,
                                    categorical=True, deadline=deadline)


def categorical_topk_no_input_needed(
    items: tuple[str, ...], protected: Iterable[str],
    honest_means: dict[str, float], honest_users: int,
    topk: int, z: float,
) -> bool | None:
    """Exact zero-added-user test for categorical_topk_inputs.

    True: original sufficient inequalities already hold; False: compute
    legal-input probabilities and solve; None: invalid protected targets.
    It deliberately uses the SAME allowed-outsider ordering and joint
    target-minus-rival variance as minimum_input_allocation.
    """
    kept = tuple(dict.fromkeys(protected))
    if not kept:
        return True
    if not set(kept) <= set(items) or len(kept) > topk:
        return None
    outside = [v for v in items if v not in kept]
    slots = topk - len(kept)
    if len(outside) <= slots:
        return True
    outside.sort(key=lambda v: -honest_means.get(v, 0.0))
    honest_n = max(0, int(honest_users))
    for target in kept:
        mu_t = float(honest_means.get(target, 0.0))
        ht = min(1.0, max(0.0, mu_t / honest_n)) if honest_n else 0.0
        for rival in outside[slots:]:
            mu_r = float(honest_means.get(rival, 0.0))
            hr = min(1.0, max(0.0, mu_r / honest_n)) if honest_n else 0.0
            variance = honest_n * pairwise_variance(ht, hr, True)
            if mu_t - mu_r < 1 + z * sqrt(variance) - 1e-8:
                return False
    return True


def expected_input_counts(items: tuple[str, ...], inputs: dict[str, int],
                          probability: Callable[[str], np.ndarray]) -> dict[str, float]:
    """Full count-vector expectation, including random votes for rivals."""
    counts = np.zeros(len(items), dtype=float)
    for value, amount in inputs.items():
        if amount:
            counts += int(amount) * probability(value)
    return dict(zip(items, counts.tolist()))


def legal_b_graph_prefixes(
    graph: dict[int, tuple[str, ...]], depth: int, alphabet: str,
    *, deadline: float | None = None,
) -> Iterable[str]:
    """Enumerate EVERY reachable, legal compressed-SAX prefix up to depth+1.

These are pre-EM *inputs*, not the C server's survivor/report candidates.
Position-ordered B edges may be combined even when the resulting sequence
never appeared verbatim in the attacker knowledge.  Symbols' padded B edges
with adjacent equal characters remain valid SERVER edges, but are rejected
only for raw Compressive-SAX inputs here.  This iterator has no beam/Top-N
cutoff and checks the established planning deadline during enumeration.
    """
    if depth < 0:
        raise ValueError("C depth cannot be negative")
    alphabet_set = set(alphabet)
    initial = tuple(dict.fromkeys(
        edge[0] for edge in graph.get(0, ())
        if len(edge) == 2 and edge[0] in alphabet_set))
    stack = list(reversed(initial))
    while stack:
        _deadline_check(deadline)
        prefix = stack.pop()
        yield prefix  
        if len(prefix) >= depth + 1:
            continue
        next_edges = graph.get(len(prefix) - 1, ())
        extensions = [prefix + edge[1] for edge in next_edges
                      if len(edge) == 2 and edge[0] == prefix[-1]
                      and edge[1] in alphabet_set and edge[1] != prefix[-1]]
        stack.extend(reversed(extensions))


def input_oue_allocations(
    candidates: tuple[str, ...], desired: dict[int, str],
    honest_means: dict[tuple[str, int], float], K: int,
    epsilon: float, z: float, honest_users: int, max_users: int,
    *, deadline: float | None = None,
) -> tuple[dict[tuple[str, int], int], list[int]] | None:
    """Global one-to-one D binding with *one* pre-OUE bit per fake user."""
    if not desired or len(set(desired.values())) != len(desired):
        return None
    cells = tuple((s, c) for s in candidates for c in range(1, K + 1))
    selected = tuple((desired[c], c) for c in sorted(desired))
    if not set(selected) <= set(cells):
        return None
    q, p = 1.0 / (exp(epsilon) + 1.0), 0.5
    matrix = {}
    for cell in selected:
        v = np.full(len(cells), q, dtype=float)
        v[cells.index(cell)] = p
        matrix[cell] = v
    best = None
    for order in permutations(sorted(desired)):
        _deadline_check(deadline)
        comparisons = []
        for i, cls in enumerate(order):
            shape = desired[cls]
            reserved_cls = set(order[:i])
            reserved_shapes = {desired[c] for c in order[:i]}
            for cell in cells:
                if (cell != (shape, cls) and cell[0] not in reserved_shapes
                        and cell[1] not in reserved_cls):
                    comparisons.append(((shape, cls), cell))
        counts = minimum_input_allocation(cells, matrix, honest_means, honest_users,
                                          comparisons, z, max_users,
                                          categorical=False, deadline=deadline)
        if counts is not None:
            key = (sum(counts.values()), tuple(counts.get(cell, 0) for cell in selected), order)
            if best is None or key < best[0]:
                best = (key, counts, list(order))
    return (best[1], best[2]) if best is not None else None


def randomize_onehot_oue(
    true_cell: Any, cells: Iterable[Any], epsilon: float,
    rng: np.random.Generator,
) -> set[Any]:
    """Randomize one genuine categorical one-hot vector to ALL OUE bits.

    Exactly one raw input bit is 1.  Every output bit is independently sampled;
    the number of 1s in the *randomized report* is not capped or repaired.
    """
    domain = tuple(cells)
    if len(set(domain)) != len(domain) or true_cell not in domain:
        raise ValueError("The pre-OUE input must identify exactly one valid cell")
    q = 1. / (exp(epsilon) + 1.)
    return {cell for cell in domain
            if rng.random() < (0.5 if cell == true_cell else q)}


def make_paired_trial_record(
    trial_index: int, trial_seed: int,
    clean: dict[str, Any], attack: dict[str, Any],
    same_length_clean: dict[str, Any],
    a_selection: dict[str, Any],
) -> dict[str, Any]:
    """Separate clean/attack primary validity from optional length diagnostics.

    This intentionally does not require the same-length diagnostic to exist
    for an otherwise valid primary comparison. A valid primary difference
    is never replaced by None merely because the diagnostic had no output.
    """
    def valid(report: dict[str, Any]) -> bool:
        return (bool(report.get("protocol_valid", report.get("accuracy") is not None))
                and report.get("accuracy") is not None)

    failures = {name: report.get("protocol_failure") or "missing_accuracy"
                for name, report in (("clean", clean), ("attack", attack))
                if not valid(report)}
    diagnostic_valid = valid(same_length_clean)
    diagnostic_failures = ({} if diagnostic_valid else
                           {"same_length_clean":
                            same_length_clean.get("protocol_failure") or "missing_accuracy"})
    primary_valid = not failures
    return {
        "trial_index": trial_index, "trial_seed": trial_seed,
        "status": "completed" if primary_valid else "failed_protocol",
        "failure_reasons": failures,
        "diagnostic_failure_reasons": diagnostic_failures,
        "same_length_diagnostic_valid": diagnostic_valid,
        "clean": clean, "same_length_clean": same_length_clean,
        "attack": attack, "a_analytical_selection": a_selection,
        "accuracy_change": (
            attack["accuracy"] - clean["accuracy"] if primary_valid else None),
        "post_length_accuracy_change": (
            attack["accuracy"] - same_length_clean["accuracy"]
            if primary_valid and diagnostic_valid else None),
    }


def input_length_inputs(
    cfg: Any, domain: tuple[int, ...], expected_support: dict[int, float],
    H: dict[int, int], fake_total: int, honest_a: int, population_n: int,
    pi: dict[int, float],
    honest_moments: Callable[..., tuple[float, float]],
    seed_pools: tuple[tuple[int, ...], ...],
    bucket: Callable[[int, int, int, int], int],
) -> dict[str, Any]:
    """Find the original A length objective using INPUT vote probabilities.

    Trace OLH: seed is chosen from its per-identity 100-seed pool BEFORE
    randomization, and the input length is selected without examining outputs.
    Symbols GRR: input length is randomized once for each malicious identity.
    """
    bmin = ceil(cfg.length_min_multiplier * cfg.alpha * fake_total)
    bmax = min(floor(cfg.length_max_multiplier * cfg.alpha * fake_total), fake_total)
    if bmax < bmin or bmax <= 0:
        return {"attempts": [{"reason": "infeasible A input budget"}]}
    clean_mode = sorted(domain, key=lambda v: (-expected_support[v], v))[0]
    pref = sorted((v for v in domain if min(H.values()) <= v < cfg.length_max
                   and v != clean_mode and v >= 2),
                  key=lambda v: (-sum(abs(v - H[c]) for c in H), -expected_support[v], v))
    g = floor(exp(cfg.epsilon)) + 1
    d = g if cfg.length_mechanism == "olh" else len(domain)
    p = exp(cfg.epsilon) / (exp(cfg.epsilon) + d - 1)
    q = 1 / (exp(cfg.epsilon) + d - 1)
    delta = p - q
    delta2 = p + q - delta ** 2
    factor = (norm.ppf(1 - cfg.length_success_delta / (len(domain) - 1))
              if cfg.probability_bound == "normal" else
              sqrt((1 - cfg.length_success_delta / (len(domain) - 1)) /
                   (cfg.length_success_delta / (len(domain) - 1))))
    attempts = []
    for t in pref:
        rivals = [r for r in domain if r != t]
        honest = {r: honest_moments(pi, honest_a, cfg.epsilon, domain,
                                     cfg.length_mechanism, t, r, population_n)
                  for r in rivals}
        expected_gain = {r: 0.0 for r in rivals}
        malicious_var = {r: 0.0 for r in rivals}
        reports: list[Any] = []
        selected_seeds: set[int] = set()

        def deficits() -> dict[int, float]:
            return {r: (mu + factor * sqrt(sigma ** 2 + malicious_var[r]) + 1
                        - expected_gain[r]) for r, (mu, sigma) in honest.items()}

        if cfg.length_mechanism == "grr":
            for _ in range(bmax):
                if len(reports) >= bmin and max(deficits().values(), default=0) <= 0:
                    break
                reports.append(t)
                for r in rivals:
                    expected_gain[r] += delta
                    malicious_var[r] += delta2
        else:
            for seeds in seed_pools[:bmax]:
                if len(reports) >= bmin and max(deficits().values(), default=0) <= 0:
                    break
                old = deficits()
                best = None
                for seed in seeds:
                    tb = bucket(seed, t, domain[0], g)
                    candidate = {}
                    for r in rivals:
                        hit = bucket(seed, r, domain[0], g) != tb
                        incr = delta if hit else 0.0
                        var = delta2 if hit else 0.0
                        candidate[r] = (incr, var)
                    new = {r: (honest[r][0] + factor * sqrt(honest[r][1] ** 2
                             + malicious_var[r] + candidate[r][1]) + 1
                             - expected_gain[r] - candidate[r][0]) for r in rivals}
                    gain = sum(max(0.0, old[r]) - max(0.0, new[r]) for r in rivals)
                    rank = (gain, int(seed not in selected_seeds), -seed)
                    if best is None or rank > best[0]:
                        best = (rank, seed, candidate)
                if best is None:
                    break
                _, seed, candidate = best
                reports.append((seed, t))
                selected_seeds.add(seed)
                for r, (incr, var) in candidate.items():
                    expected_gain[r] += incr
                    malicious_var[r] += var
        final_deficits = deficits()
        feasible = (bmin <= len(reports) <= bmax and
                    max(final_deficits.values(), default=0) <= 1e-8)
        attempts.append({"length": t, "reports": len(reports),
                         "expected_net_gains": dict(expected_gain),
                         "worst_confidence_deficit": max(final_deficits.values(), default=0.0),
                         "feasible": feasible,
                         "distinct_seeds": len(selected_seeds) if selected_seeds else None})
        if feasible:
            return {"target": t, "reports": reports, "feasible": True,
                    "min_required": len(reports), "reference_min": min(H.values()),
                    "bounds": final_deficits, "net_gains": expected_gain,
                    "strategy": "input_confidence_planned", "attempts": attempts}
    return {"attempts": attempts, "reference_min": min(H.values()),
            "strategy": "input_no_feasible_length"}
