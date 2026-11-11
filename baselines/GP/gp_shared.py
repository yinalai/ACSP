from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from math import exp, floor, inf
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from rra_shared import malicious_quotas


def _knowledge(data: Any, size: int, seed: int) -> Any:
    n = len(data.train)
    if size < 0 or size > n:
        raise ValueError("greedy_auxiliary_sample must be 0 or in [1,N]")
    selected = (range(n) if size == 0 else
                sorted(np.random.default_rng(seed).choice(n, size, replace=False)))
    return SimpleNamespace(train=tuple(data.train[i] for i in selected),
                           labels=tuple(data.labels[i] for i in selected))


def _length_expectation(engine: Any, known: Any, cfg: Any,
                        n_a: int) -> dict[int, float]:
    """Genuine expected report supports; OLH uses actual indexed hash seeds."""
    domain = tuple(range(cfg.length_min, cfg.length_max + 1))
    population = Counter(max(cfg.length_min, min(cfg.length_max, len(s)))
                         for s in known.train)
    mass = {v: population[v] / len(known.train) for v in domain}
    if cfg.length_mechanism == "grr":
        p, q = engine.grr_prob(cfg.epsilon, len(domain))
        return {v: n_a * (q + (p - q) * mass[v]) for v in domain}
    if cfg.length_mechanism != "olh":
        raise ValueError("Unknown length mechanism")
    bins = floor(exp(cfg.epsilon)) + 1
    p, q = engine.grr_prob(cfg.epsilon, bins)
    expectation = dict.fromkeys(domain, 0.0)
    for seed in range(n_a):
        bucket_of = {v: engine.bucket(seed, v, domain[0], bins) for v in domain}
        bucket_mass = {b: sum(mass[v] for v in domain if bucket_of[v] == b)
                       for b in range(bins)}
        for v in domain:
            expectation[v] += q + (p - q) * bucket_mass[bucket_of[v]]
    return expectation


def _expected_c(engine: Any, known: Any, candidates: tuple[str, ...],
                n_c: int, epsilon: float) -> dict[str, float]:
    votes = dict.fromkeys(candidates, 0.0)
    if not candidates or not n_c:
        return votes
    for sequence, multiplicity in Counter(known.train).items():
        probabilities = engine.em_probs(sequence, candidates, epsilon)
        scale = n_c * multiplicity / len(known.train)
        for j, candidate in enumerate(candidates):
            votes[candidate] += scale * float(probabilities[j])
    return votes


def _d_vote_distance(engine: Any, sequence: str,
                     shape: str, symbols: bool) -> float:
    
    return engine.trace_distance(sequence if symbols else sequence[:len(shape)], shape)


def _d_shape_distance(engine: Any, original: str, substitute: str) -> float:
    
    return engine.trace_distance(original, substitute)


def _expected_d(engine: Any, known: Any, candidates: tuple[str, ...],
                n_d: int, cfg: Any, symbols: bool) -> dict[tuple[str, int], float]:
    cells = {(s, c): 0.0 for s in candidates for c in range(1, cfg.top_k + 1)}
    if not candidates:
        return cells
    q = 1.0 / (exp(cfg.epsilon) + 1.0)
    for cell in cells:
        cells[cell] = n_d * q
    for (sequence, cls), times in Counter(zip(known.train, known.labels)).items():
        nearest = min(candidates, key=lambda s: _d_vote_distance(engine, sequence, s, symbols))
        cells[(nearest, cls)] += n_d * (.5 - q) * times / len(known.train)
    return cells


def _normal_reference(engine: Any, known: Any, cfg: Any, run_seed: int,
                      genuine_n: int, symbols: bool) -> dict[str, Any]:
    before = engine.distribute_groups(genuine_n, cfg, run_seed + 1, length=1)
    length_scores = _length_expectation(engine, known, cfg, len(before["A"]))
    domain = tuple(range(cfg.length_min, cfg.length_max + 1))
    length = engine.select_length_output(Counter(length_scores), domain)
    if symbols and length < 2:
        return {"length": length, "representatives": [None] * cfg.top_k,
                "leaves": [], "reason": "B_no_pairs"}
    after = engine.distribute_groups(genuine_n, cfg, run_seed + 1, length=length)
    groups = engine.split_users(after["C"], length)
    b_means = engine.position_expected(known, length, len(after["B"]), cfg.epsilon)
    keep = cfg.branch_factor * cfg.top_k if symbols else 2 * cfg.top_k
    
    
    b_graph = {pos: tuple(engine.top_names({pair: b_means[pos][pair]
                                           for pair in engine.PAIRS}, keep))
               for pos in range(max(0, length - 1))}
    parents = [""]
    for depth in range(length):
        candidates = engine.expand_paths(parents, b_graph, depth)
        if not candidates:
            return {"length": length, "representatives": [None] * cfg.top_k,
                    "leaves": [], "reason": f"C_empty_{depth}"}
        score = _expected_c(engine, known, tuple(candidates),
                            len(groups[depth]), cfg.epsilon)
        parents = engine.top_nodes_in_creation_order(candidates, score, keep)
    leaves = tuple(parents)
    counts = _expected_d(engine, known, leaves, len(after["D"]), cfg, symbols)
    return {"length": length,
            "representatives": engine.trace_label_greedy(counts, leaves, cfg.top_k),
            "leaves": list(leaves), "reason": None}


def _a_score(engine: Any, counts: dict[int, float], original: int,
             domain: tuple[int, ...], symbols: bool) -> tuple[int, float]:
    actual = engine.select_length_output(Counter(counts), domain)
    available = domain[:-1] if symbols else domain
    if actual != original:
        margin = counts[actual] - counts.get(original, 0.0)
    else:
        others = [counts.get(v, 0.0) - counts.get(original, 0.0)
                  for v in available if v != original]
        margin = max(others) if others else -inf
    return (int(actual != original), float(margin))


def _choose_a(engine: Any, expectation: dict[int, float], original: int,
              cfg: Any, rng: np.random.Generator, probes: int,
              symbols: bool) -> Any:
    domain = tuple(expectation)
    if cfg.length_mechanism == "grr":
        options = [(v, (v,)) for v in domain]
    else:
        bins = floor(exp(cfg.epsilon)) + 1
        options = []
        for seed in rng.integers(cfg.seed_space, size=probes):
            for bucket in range(bins):
                msg = (int(seed), bucket)
                options.append((msg, engine.supported_lengths(msg, domain, cfg.epsilon)))
    best_score: tuple[int, float] | None = None
    winners = []
    for report, support in options:
        for value in support:
            expectation[value] += 1
        score = _a_score(engine, expectation, original, domain, symbols)
        for value in support:
            expectation[value] -= 1
        if best_score is None or score > best_score:
            best_score, winners = score, [report]
        elif score == best_score:
            winners.append(report)
    chosen = winners[int(rng.integers(len(winners)))]
    support = ((chosen,) if cfg.length_mechanism == "grr" else
               engine.supported_lengths(chosen, domain, cfg.epsilon))
    for value in support:
        expectation[value] += 1
    return chosen


def _top(engine: Any, values: dict[str, float], keep: int) -> tuple[str, ...]:
    return tuple(engine.top_names(values, keep))


def _local_rank(engine: Any, values: dict[str, float],
                original: set[str], keep: int) -> tuple[set[str], set[str], set[str]]:
    current = set(_top(engine, values, keep))
    new = current - original
    return current & original, set(values) - original - new, new


def _choose_rank(engine: Any, score: dict[str, float],
                 baseline: set[str], keep: int,
                 rng: np.random.Generator) -> str:
    """Fixed original top-k, unfinished outsiders only, creation-order ties."""
    incumbent, unpromoted, already = _local_rank(engine, score, baseline, keep)
    if not incumbent or not unpromoted:
        pool = already if already else set(score)
        least = min(score[s] for s in pool)
        matches = [s for s in score if s in pool and score[s] == least]
        chosen = matches[int(rng.integers(len(matches)))]
        score[chosen] += 1
        return chosen
    optimal: tuple[int, float] | None = None
    ties: list[str] = []
    for candidate in score:
        score[candidate] += 1
        current = set(_top(engine, score, keep))
        replacements = len(current - baseline)
        boundary = max(score[s] for s in unpromoted) - min(score[s] for s in incumbent)
        utility = replacements, boundary
        score[candidate] -= 1
        if optimal is None or utility > optimal:
            optimal, ties = utility, [candidate]
        elif utility == optimal:
            ties.append(candidate)
    chosen = ties[int(rng.integers(len(ties)))]
    score[chosen] += 1
    return chosen


def _choose_b(engine: Any, model: dict[int, dict[str, float]],
              original: dict[int, set[str]], keep: int,
              rng: np.random.Generator) -> tuple[int, str]:
    """Across positions maximize added replacements, then local boundary."""
    best: tuple[int, float] | None = None
    ties: list[tuple[int, str]] = []
    current_total = sum(len(_local_rank(engine, values, original[pos], keep)[2])
                        for pos, values in model.items())
    for pos, values in model.items():
        incumbent, unpromoted, already = _local_rank(engine, values, original[pos], keep)
        pool = ((already if already else set(values)) if not incumbent or not unpromoted
                else set(values))
        for pair in values:
            if pair not in pool:
                continue
            values[pair] += 1
            if incumbent and unpromoted:
                displaced = len(set(_top(engine, values, keep)) - original[pos])
                boundary = (max(values[v] for v in unpromoted) -
                            min(values[v] for v in incumbent))
                utility = (current_total - len(already) + displaced, boundary)
            else:
                utility = (current_total, -values[pair] + 1)
            values[pair] -= 1
            if best is None or utility > best:
                best, ties = utility, [(pos, pair)]
            elif utility == best:
                ties.append((pos, pair))
    pos, pair = ties[int(rng.integers(len(ties)))]
    model[pos][pair] += 1
    return pos, pair


def _binding_trace(counts: dict[tuple[str, int], float],
                   leaves: tuple[str, ...], K: int) -> list[tuple[str, int]]:
    """Reproduce the original global greedy sequence, including strict > ties."""
    selected: list[tuple[str, int]] = []
    used_shapes: set[str] = set()
    used_labels: set[int] = set()
    for _ in range(K):
        best = (0.0, None)
        for shape in leaves:
            if shape in used_shapes:
                continue
            for cls in range(1, K + 1):
                if cls not in used_labels and counts[(shape, cls)] > best[0]:
                    best = (counts[(shape, cls)], (shape, cls))
        if best[1] is None:
            break
        shape, cls = best[1]
        selected.append((shape, cls))
        used_shapes.add(shape)
        used_labels.add(cls)
    return selected


def _d_targets(engine: Any, known: Any, leaves: tuple[str, ...],
               expected: dict[tuple[str, int], float],
               original_clean: list[str | None], current_clean: list[str | None],
               cfg: Any, symbols: bool) -> tuple[dict[int, str], dict[int, float]]:
    references, _ = engine.length_reference(known, cfg.reference_coverage)
    targets: dict[int, str] = {}
    distortion: dict[int, float] = {}
    for cls in range(1, cfg.top_k + 1):
        base = original_clean[cls - 1]
        conditional = current_clean[cls - 1]
        if not base or not conditional or cls not in references:
            continue
        alternatives = []
        for shape in leaves:
            if shape == base or shape == conditional:
                continue
            cls_dist = {j: sum(weight * _d_vote_distance(engine, ref, shape, symbols)
                               for ref, weight in refs)
                        for j, refs in references.items()}
            if not all(cls_dist[cls] < d for j, d in cls_dist.items() if j != cls):
                continue
            difference = _d_shape_distance(engine, base, shape)
            if difference > 0:
                alternatives.append((expected[(shape, cls)], difference, shape))
        if alternatives:
            max_support = max(t[0] for t in alternatives)
            max_diff = max(t[1] for t in alternatives if t[0] == max_support)
            targets[cls] = next(t[2] for t in alternatives
                                if t[0] == max_support and t[1] == max_diff)
            distortion[cls] = max_diff
    return targets, distortion


def _d_score(engine: Any, counts: dict[tuple[str, int], float],
             leaves: tuple[str, ...], K: int, targets: dict[int, str],
             distortions: dict[int, float]) -> tuple[int, float, float]:
    actual = engine.trace_label_greedy(counts, leaves, K)
    completed = [cls for cls, s in targets.items() if actual[cls - 1] == s]
    number = len(completed)
    shape_shift = sum(distortions[cls] for cls in completed)
    if number == len(targets):
        return (number, shape_shift, -inf)
    chosen = _binding_trace(counts, leaves, K)
    margins = []
    for cls, shape in targets.items():
        if cls in completed:
            continue
        competitor = next((other for other in chosen
                           if other[0] == shape or other[1] == cls), None)
        margins.append((counts[(shape, cls)] - counts[competitor]
                        if competitor is not None else -inf))
    return number, shape_shift, max(margins) if margins else -inf


def _weakest_achieved(engine: Any, counts: dict[tuple[str, int], float],
                      leaves: tuple[str, ...], K: int,
                      targets: dict[int, str]) -> tuple[str, int]:
    chosen = _binding_trace(counts, leaves, K)
    used_s: set[str] = set()
    used_c: set[int] = set()
    margins = []
    for shape, cls in chosen:
        if targets.get(cls) == shape:
            others = [counts[(s, c)] for s in leaves if s not in used_s
                      for c in range(1, K + 1) if c not in used_c
                      and (s, c) != (shape, cls)]
            margins.append((counts[(shape, cls)] - max(others, default=0), (shape, cls)))
        used_s.add(shape)
        used_c.add(cls)
    if margins:
        return min(margins, key=lambda item: item[0])[1]
    return next(iter((shape, cls) for cls, shape in targets.items()))


def _random_bits(cells: tuple[tuple[str, int], ...], cap: int,
                 rng: np.random.Generator) -> list[tuple[str, int]]:
    how_many = int(rng.integers(1, cap + 1))
    indices = rng.choice(len(cells), size=how_many, replace=False)
    return [cells[int(i)] for i in indices]


def _choose_d(engine: Any, counts: dict[tuple[str, int], float],
              leaves: tuple[str, ...], cfg: Any,
              targets: dict[int, str], distortions: dict[int, float],
              rng: np.random.Generator) -> list[tuple[str, int]]:
    cells = tuple(counts)
    cap = min(len(cells), engine.oue_ones_cap(cfg.top_k, len(leaves), cfg.epsilon))
    if not targets:
        bits = _random_bits(cells, cap, rng)
        for cell in bits:
            counts[cell] += 1
        return bits
    baseline = _d_score(engine, counts, leaves, cfg.top_k, targets, distortions)
    if baseline[0] == len(targets):
        cell = _weakest_achieved(engine, counts, leaves, cfg.top_k, targets)
        counts[cell] += 1
        return [cell]
    chosen: list[tuple[str, int]] = []
    current = baseline
    for slot in range(cap):
        best = None
        ties: list[tuple[str, int]] = []
        for cell in cells:
            if cell in chosen:
                continue
            counts[cell] += 1
            candidate_score = _d_score(engine, counts, leaves, cfg.top_k,
                                       targets, distortions)
            counts[cell] -= 1
            if best is None or candidate_score > best:
                best, ties = candidate_score, [cell]
            elif candidate_score == best:
                ties.append(cell)
        if slot and best <= current:
            break
        target_ties = [cell for cell in ties if targets.get(cell[1]) == cell[0]]
        if target_ties:
            
            
            
            largest = max(counts[cell] for cell in target_ties)
            options = [cell for cell in target_ties if counts[cell] == largest]
        else:
            options = ties
        cell = options[int(rng.integers(len(options)))]
        chosen.append(cell)
        counts[cell] += 1
        current = _d_score(engine, counts, leaves, cfg.top_k, targets, distortions)
    return chosen


def gp_report(engine: Any, data: Any, cfg: Any, run_seed: int,
              *, symbols: bool, knowledge_size: int = 0,
              seed_candidates: int = 32,
              evaluate_test: bool = True) -> dict[str, Any]:
    """One genuine protocol run with injected stage-local greedily chosen reports."""
    if cfg.attack_mode != "gp" or cfg.epsilon <= 0 or cfg.beta < 0 or run_seed < 0:
        raise ValueError("GP requires attack_mode='gp', epsilon>0, beta>=0, seed>=0")
    if seed_candidates < 1:
        raise ValueError("seed_candidates must be positive")
    if symbols:
        engine.configure_symbol_size(cfg.symbol_size)
    data.validate()
    N, M = len(data.train), floor(cfg.beta * len(data.train))
    known = _knowledge(data, knowledge_size, run_seed + 1707)
    domain = tuple(range(cfg.length_min, cfg.length_max + 1))
    before = engine.distribute_groups(N, cfg, run_seed + 1, length=1)
    ma = min(M, floor(M * len(before["A"]) / N + .5))
    mb = min(M - ma, floor(M * len(before["B"]) / N + .5))
    fake_used = {s: 0 for s in "ABCD"}
    unreportable = {"B_no_pairs": 0, "C_no_frontier": 0, "D_no_leaves": 0}
    events: list[dict[str, Any]] = []

    
    
    honest_a_rng = np.random.default_rng(run_seed + 101)
    fake_a_rng = np.random.default_rng(run_seed + 1101)
    a_counts: Counter = Counter()
    estimates = _length_expectation(engine, known, cfg, len(before["A"]))
    normal_output = engine.select_length_output(Counter(estimates), domain)
    a_malicious: list[Any] = []
    if cfg.length_mechanism == "olh":
        for slot, user in enumerate(before["A"]):
            value = max(cfg.length_min, min(cfg.length_max, len(data.train[user])))
            report = engine.olh_report(value, slot, cfg.epsilon, domain, honest_a_rng)
            a_counts.update(engine.supported_lengths(report, domain, cfg.epsilon))
        for _ in range(ma):
            report = _choose_a(engine, estimates, normal_output, cfg,
                               fake_a_rng, seed_candidates, symbols)
            a_malicious.append(report)
            a_counts.update(engine.supported_lengths(report, domain, cfg.epsilon))
    elif cfg.length_mechanism == "grr":
        for user in before["A"]:
            length = max(cfg.length_min, min(cfg.length_max, len(data.train[user])))
            a_counts[engine.randomized_response(length, domain,
                                                cfg.epsilon, honest_a_rng)] += 1
        for _ in range(ma):
            report = _choose_a(engine, estimates, normal_output, cfg,
                               fake_a_rng, seed_candidates, symbols)
            a_malicious.append(report)
            a_counts[report] += 1
    else:
        raise ValueError("Unsupported length mechanism")
    fake_used["A"] = ma
    L = engine.select_length_output(a_counts, domain)
    expected_post_a = engine.select_length_output(Counter(estimates), domain)
    after = engine.distribute_groups(N, cfg, run_seed + 1, length=L)
    c_groups = engine.split_users(after["C"], L)
    quota = malicious_quotas(M, N, before, after, c_groups)
    assert (quota["A"], quota["B"]) == (ma, mb)
    events.append({"event": "gp_length", "length": L,
                   "estimated_clean_length": normal_output,
                   "estimated_attacked_length": expected_post_a,
                   "estimated_length_displaced": expected_post_a != normal_output,
                   "fake_A": ma})
    events.append({"event": "gp_quotas", "quotas": quota})
    failure = "symbols_B_requires_L_at_least_2" if symbols and L < 2 else None

    
    
    honest_b_rng = np.random.default_rng(run_seed + 202)
    fake_b_rng = np.random.default_rng(run_seed + 1202)
    b_counts = {pos: Counter() for pos in range(max(0, L - 1))}
    keep = cfg.branch_factor * cfg.top_k if symbols else 2 * cfg.top_k
    if L >= 2 and failure is None:
        for user in after["B"]:
            original = data.train[user]
            if symbols:
                sequence = original[:L].ljust(L, original[-1])
                pos = int(honest_b_rng.integers(L - 1))
            else:
                sequence = original[:L]
                if len(sequence) < 2:
                    continue
                pos = int(honest_b_rng.integers(len(sequence) - 1))
            report = engine.randomized_response(sequence[pos:pos + 2],
                                                engine.PAIRS, cfg.epsilon, honest_b_rng)
            b_counts[pos][report] += 1
        b_model = engine.position_expected(known, L, len(after["B"]), cfg.epsilon)
        original_b = {pos: set(_top(engine, values, keep))
                      for pos, values in b_model.items()}
        for _ in range(mb):
            pos, report = _choose_b(engine, b_model, original_b, keep, fake_b_rng)
            b_counts[pos][report] += 1
        fake_used["B"] = mb
        events.append({"event": "gp_b_local_gain", "displaced_per_position": {
            str(pos): len(set(_top(engine, values, keep)) - original_b[pos])
            for pos, values in b_model.items()}})
    else:
        unreportable["B_no_pairs"] = mb
    b_graph = {pos: tuple(engine.top_names(dict(b_counts[pos]), keep))
               for pos in range(max(0, L - 1))}

    
    
    honest_c_rng = np.random.default_rng(run_seed + 303)
    fake_c_rng = np.random.default_rng(run_seed + 1303)
    parents = [""]
    c_states: list[dict[str, Any]] = []
    for depth in range(L):
        nodes = engine.expand_paths(parents, b_graph, depth)
        if not nodes:
            unreportable["C_no_frontier"] = sum(quota["C"][depth:])
            parents = []
            events.append({"event": "gp_c_no_frontier", "depth": depth})
            break
        candidates = tuple(nodes)
        votes = Counter()
        for user in c_groups[depth]:
            probabilities = engine.em_probs(data.train[user], candidates, cfg.epsilon)
            report = candidates[int(honest_c_rng.choice(len(candidates), p=probabilities))]
            votes[report] += 1
        model = _expected_c(engine, known, candidates,
                            len(c_groups[depth]), cfg.epsilon)
        original = set(_top(engine, model, keep))
        for _ in range(quota["C"][depth]):
            report = _choose_rank(engine, model, original, keep, fake_c_rng)
            votes[report] += 1
        fake_used["C"] += quota["C"][depth]
        if sum(votes.values()) != len(c_groups[depth]) + quota["C"][depth]:
            raise AssertionError("GP must append, never replace, C reports")
        survivors = engine.top_nodes_in_creation_order(
            nodes, {name: votes[name] for name in nodes}, keep)
        c_states.append({"depth": depth, "candidates": list(candidates),
                         "survivors": survivors, "honest_reports": len(c_groups[depth]),
                         "malicious_reports": quota["C"][depth],
                         "received_reports": sum(votes.values()),
                         "estimated_displaced_candidates":
                             len(set(_top(engine, model, keep)) - original)})
        parents = survivors

    leaves = tuple(parents) if c_states and c_states[-1]["depth"] == L - 1 else ()
    honest_d_rng = np.random.default_rng(run_seed + 404)
    fake_d_rng = np.random.default_rng(run_seed + 1404)
    
    d_counts = engine.honest_oue_counts(data, after["D"], leaves, cfg, honest_d_rng)
    d_vectors: list[list[tuple[str, int]]] = []
    targets: dict[int, str] = {}
    expected_d_target_hits: int | None = None
    gp_original: dict[str, Any] | None = None
    if leaves and quota["D"]:
        expected_d = _expected_d(engine, known, leaves, len(after["D"]), cfg, symbols)
        conditional = engine.trace_label_greedy(expected_d, leaves, cfg.top_k)
        gp_original = _normal_reference(engine, known, cfg, run_seed, N, symbols)
        targets, distortions = _d_targets(
            engine, known, leaves, expected_d, gp_original["representatives"],
            conditional, cfg, symbols)
        
        shadow_d = dict(expected_d)
        for _ in range(quota["D"]):
            bits = _choose_d(engine, shadow_d, leaves, cfg,
                             targets, distortions, fake_d_rng)
            d_vectors.append(bits)
            for cell in bits:
                d_counts[cell] += 1
        fake_used["D"] = quota["D"]
        expected_d_target_hits = sum(
            engine.trace_label_greedy(shadow_d, leaves, cfg.top_k)[c - 1] == s
            for c, s in targets.items())
        events.append({"event": "gp_d_targets", "targets": {str(k): v for k, v in targets.items()},
                       "expected_clean_representatives": gp_original["representatives"],
                       "conditional_D_representatives": conditional,
                       "expected_attacked_target_hits": expected_d_target_hits,
                       "no_valid_D_target": not bool(targets)})
    elif not leaves:
        unreportable["D_no_leaves"] = quota["D"]

    representatives = (engine.trace_label_greedy(d_counts, leaves, cfg.top_k)
                       if leaves else [None] * cfg.top_k)
    complete = len(representatives) == cfg.top_k and all(
        isinstance(s, str) and bool(s) for s in representatives)
    if symbols:
        if failure is None and not complete:
            failure = "symbols_classification_requires_nonempty_representatives"
    elif not any(isinstance(s, str) and bool(s) for s in representatives):
        failure = "trace_classification_requires_nonempty_representatives"
    if failure is not None:
        events.append({"event": "protocol_unavailable", "reason": failure})

    
    
    
    references, _ = engine.length_reference(data, cfg.reference_coverage)
    shape_dist = {}
    norm_dist = {}
    for cls in range(1, cfg.top_k + 1):
        shape = representatives[cls - 1]
        if not shape or cls not in references:
            shape_dist[str(cls)] = norm_dist[str(cls)] = None
            continue
        refs = references[cls]
        shape_dist[str(cls)] = engine.ref_dist(shape, refs)
        norm_dist[str(cls)] = engine.ref_dist(shape, refs, True)
    unused = M - sum(fake_used.values())
    if unused != sum(unreportable.values()):
        raise AssertionError("All reportable injected identities must be spent")
    if fake_used["A"] != quota["A"] or fake_used["B"] + unreportable["B_no_pairs"] != quota["B"]:
        raise AssertionError("A/B injected identity accounting mismatch")
    if fake_used["C"] + unreportable["C_no_frontier"] != sum(quota["C"]):
        raise AssertionError("C injected identity accounting mismatch")
    if fake_used["D"] + unreportable["D_no_leaves"] != quota["D"]:
        raise AssertionError("D injected identity accounting mismatch")
    server_count = {"A": len(after["A"]) + fake_used["A"],
                    "B": sum(sum(x.values()) for x in b_counts.values()),
                    "C_per_layer": [x["received_reports"] for x in c_states],
                    "D": len(after["D"]) + fake_used["D"] if leaves else 0}
    report = {
        "seed": run_seed, "attacked": True, "attack_model": "gp_injected_users",
        "epsilon": cfg.epsilon, "beta": cfg.beta, "length": L,
        "protocol_valid": failure is None, "protocol_failure": failure,
        "genuine_population": N, "fixed_server_population": N + M,
        "fake_total": M, "honest_groups": {name: len(users) for name, users in after.items()},
        "malicious_group_quotas": quota, "fake_used": fake_used,
        "fake_remaining": unused, "gp_unreportable_slots": unreportable,
        "gp_length_reports": [list(x) if isinstance(x, tuple) else x for x in a_malicious],
        "server_report_counts": server_count,
        "b_kept": {str(pos): list(edges) for pos, edges in b_graph.items()},
        "c_levels": c_states,
        "d_ones_cap": engine.oue_ones_cap(cfg.top_k, len(leaves), cfg.epsilon) if leaves else 0,
        "d_actual_ones_per_vector": [len(v) for v in d_vectors],
        "d_malicious_vectors": [[f"{s}:{c}" for s, c in bits] for bits in d_vectors],
        "gp_targets": {str(cls): s for cls, s in targets.items()},
        "gp_realized_target_hits": {str(c): representatives[c - 1] == s
                                    for c, s in targets.items()},
        "gp_expected_attacked_target_hits": expected_d_target_hits,
        "gp_knowledge_size": len(known.train), "gp_seed_candidates": seed_candidates,
        "gp_reference_estimator": "deterministic_expected_A_B_C_D_support",
        "gp_expected_clean_representatives": (gp_original["representatives"]
                                               if gp_original is not None else None),
        "representatives": representatives, "complete_representatives": complete,
        "accuracy": engine.accuracy(data, representatives)
                    if evaluate_test and failure is None else None,
        "recall_by_class": engine.recall_by_class(data, representatives)
                           if evaluate_test and failure is None else None,
        "shape_distance": shape_dist, "normalized_shape_distance": norm_dist,
        "test_evaluation_performed": evaluate_test and failure is None,
        "events": events,
    }
    if symbols:
        report["symbols_protocol_version"] = engine.SYMBOLS_PROTOCOL_VERSION
        report["honest_c_per_level"] = len(c_groups[0]) if c_groups else 0
    return report


def paired_gp_experiment(engine: Any, data: Any, cfg: Any, *, symbols: bool,
                         runs: int, seed: int, trial_start: int = 0,
                         checkpoint_dir: Path | None = None,
                         resume: bool = False, knowledge_size: int = 0,
                         seed_candidates: int = 32) -> dict[str, Any]:
    """Paired original clean vs injected GP; no PK data or PK planner is used."""
    if runs < 1 or trial_start < 0 or seed < 0:
        raise ValueError("runs must be >=1 and trial_start/seed nonnegative")
    if resume and checkpoint_dir is None:
        raise ValueError("resume requires checkpoint_dir")
    if cfg.attack_mode != "gp":
        raise ValueError("Use a Config(attack_mode='gp')")
    base_signature = (engine._prepare_trial_checkpoints(
        checkpoint_dir, data, cfg, seed, resume) if checkpoint_dir else None)
    extra = json.dumps({"baseline": "gp_v1", "knowledge_size": knowledge_size,
                        "seed_candidates": seed_candidates}, sort_keys=True)
    signature = (hashlib.sha256((str(base_signature) + extra).encode()).hexdigest()
                 if base_signature is not None else None)
    records: list[dict[str, Any]] = []

    def valid(report: dict[str, Any]) -> bool:
        value = report.get("accuracy")
        return (report.get("protocol_valid") is True and
                isinstance(value, (int, float, np.integer, np.floating)) and
                bool(np.isfinite(value)))

    def reconcile(record: dict[str, Any]) -> dict[str, Any]:
        flags = {key: valid(record[key]) for key in ("clean", "attack")}
        paired = flags["clean"] and flags["attack"]
        record.update({
            "valid_by_report": flags, "main_pair_valid": paired,
            "status": ("completed" if paired else
                       "partial_result" if any(flags.values()) else "failed_protocol"),
            "failure_reasons": {
                key: record[key].get("protocol_failure") or "accuracy_unavailable"
                for key in flags if not flags[key]},
            "accuracy_change": (record["attack"]["accuracy"] -
                                record["clean"]["accuracy"] if paired else None)})
        return record

    for offset in range(runs):
        index = trial_start + offset
        trial_seed = seed + index * 10007
        saved = (checkpoint_dir / f"trial_{index:08d}.json"
                 if checkpoint_dir is not None else None)
        if saved is not None and saved.exists():
            if not resume:
                raise ValueError(f"Existing GP checkpoint: {saved}; use resume")
            payload = json.loads(saved.read_text(encoding="utf8"))
            if (payload.get("signature") != signature or
                    payload.get("trial_index") != index or
                    payload.get("trial_seed") != trial_seed):
                raise ValueError(f"Incompatible GP checkpoint: {saved}")
            records.append(reconcile(payload["record"]))
            print(f"GP trial={offset + 1}/{runs} global={index} resumed", flush=True)
            continue
        clean = engine.trace_report(data, cfg, trial_seed, False)
        attack = gp_report(engine, data, cfg, trial_seed, symbols=symbols,
                           knowledge_size=knowledge_size,
                           seed_candidates=seed_candidates)
        record = reconcile({"trial_index": index, "trial_seed": trial_seed,
                            "clean": clean, "attack": attack})
        if saved is not None:
            engine._write_json_atomic(saved, {
                "signature": signature, "trial_index": index,
                "trial_seed": trial_seed, "record": record})
        records.append(record)
        print(f"GP trial={offset + 1}/{runs} global={index} "
              f"L={attack['length']} injected={sum(attack['fake_used'].values())}/"
              f"{attack['fake_total']} status={record['status']}", flush=True)

    def average(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    paired = [r for r in records if r["main_pair_valid"]]
    summary = {
        "runs": runs, "completed_trials": len(paired),
        "failed_protocol_trials": runs - len(paired),
        "valid_trials_by_report": {
            key: sum(r["valid_by_report"][key] for r in records)
            for key in ("clean", "attack")},
        "mean_clean_accuracy": average([r["clean"]["accuracy"] for r in records
                                         if r["valid_by_report"]["clean"]]),
        "mean_attack_accuracy": average([r["attack"]["accuracy"] for r in records
                                          if r["valid_by_report"]["attack"]]),
        "mean_accuracy_change": average([r["accuracy_change"] for r in paired]),
        "mean_injected_reports": average([
            sum(r["attack"]["fake_used"].values()) for r in records]),
        "mean_unreportable_injected_users": average([
            r["attack"]["fake_remaining"] for r in records]),
        "mean_gp_target_count": average([
            len(r["attack"]["gp_targets"]) for r in records]),
    }
    return {"attack_model": "gp_injected_users", "config": asdict(cfg),
            "knowledge_size": knowledge_size, "seed_candidates": seed_candidates,
            "trial_start": trial_start, "summary": summary, "trials": records}


def main(engine: Any, *, symbols: bool) -> None:
    parser = argparse.ArgumentParser(
        description="PrivShape independent stage-local Greedy Poisoning baseline")
    parser.add_argument("--data-dir", type=Path,
                        help="Directory with data.txt, label.txt, test_data.txt, test_label.txt")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-users", type=int, default=1200)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=.10,
                        help="Inject floor(beta * genuine training population) users")
    parser.add_argument("--knowledge-size", type=int, default=0,
                        help="0: all training records; otherwise deterministic subset")
    parser.add_argument("--seed-candidates", type=int, default=32,
                        help="Per-fake-user legal OLH seed probes in Trace A")
    if symbols:
        parser.add_argument("--symbol-size", type=int, default=6)
    else:
        parser.add_argument("--length-mechanism", choices=("olh", "grr"),
                            default="olh")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20231025)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(
        "gp_symbols_result.json" if symbols else "gp_trace_result.json"))
    args = parser.parse_args()
    if args.demo == (args.data_dir is not None):
        parser.error("Specify exactly one of --demo or --data-dir")
    if (args.demo_users < 1 or args.epsilon <= 0 or args.beta < 0 or
            args.seed < 0 or args.runs < 1 or args.trial_start < 0 or
            args.knowledge_size < 0 or args.seed_candidates < 1):
        parser.error("Invalid dataset size, parameter or seed")
    if symbols and not 3 <= args.symbol_size <= 10:
        parser.error("--symbol-size must be between 3 and 10")
    if args.output.exists() and not args.resume:
        parser.error("Output exists; choose another path or --resume")
    directory = Path(str(args.output) + ".trials")
    if args.resume and args.output.exists() and not (directory / "manifest.json").exists():
        parser.error("Cannot resume a legacy file without a checkpoint manifest")
    kwargs = {"epsilon": args.epsilon, "beta": args.beta, "attack_mode": "gp"}
    if symbols:
        kwargs["symbol_size"] = args.symbol_size
    else:
        kwargs["length_mechanism"] = args.length_mechanism
    cfg = engine.Config(**kwargs)
    data = (engine.demo_dataset(args.demo_users, seed=args.seed) if args.demo
            else engine.load_dataset(args.data_dir))
    result = paired_gp_experiment(
        engine, data, cfg, symbols=symbols, runs=args.runs, seed=args.seed,
        trial_start=args.trial_start, checkpoint_dir=directory,
        resume=args.resume, knowledge_size=args.knowledge_size,
        seed_candidates=args.seed_candidates)
    engine._write_json_atomic(args.output, result)
    print(f"Wrote {args.output}; paired clean/GP valid="
          f"{result['summary']['completed_trials']}/{args.runs}", flush=True)
