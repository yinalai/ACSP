from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, permutations, product
from math import ceil, exp, floor, isfinite, sqrt
from pathlib import Path
import json
import os
import random
import tempfile
import time
from typing import Any, Iterable

import numpy as np
from scipy.stats import norm


ABC = "abcd"
PAIRS = tuple(a + b for a in ABC for b in ABC if a != b)
MASK32 = 0xFFFFFFFF
P1, P2, P3, P4, P5 = 0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1


def rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & MASK32


def xxh32(payload: bytes, seed: int = 0) -> int:
    """Pure-Python xxHash32, matching code_trace's xxhash.xxh32(...)."""
    seed &= MASK32
    size = len(payload)
    cursor = 0

    def round32(acc: int, lane: int) -> int:
        return (rotl32((acc + lane * P2) & MASK32, 13) * P1) & MASK32

    if size >= 16:
        v1 = (seed + P1 + P2) & MASK32
        v2 = (seed + P2) & MASK32
        v3 = seed
        v4 = (seed - P1) & MASK32
        while cursor <= size - 16:
            v1 = round32(v1, int.from_bytes(payload[cursor:cursor + 4], "little"))
            v2 = round32(v2, int.from_bytes(payload[cursor + 4:cursor + 8], "little"))
            v3 = round32(v3, int.from_bytes(payload[cursor + 8:cursor + 12], "little"))
            v4 = round32(v4, int.from_bytes(payload[cursor + 12:cursor + 16], "little"))
            cursor += 16
        h = (rotl32(v1, 1) + rotl32(v2, 7) + rotl32(v3, 12) + rotl32(v4, 18)) & MASK32
    else:
        h = (seed + P5) & MASK32
    h = (h + size) & MASK32
    while cursor <= size - 4:
        lane = int.from_bytes(payload[cursor:cursor + 4], "little")
        h = (rotl32((h + lane * P3) & MASK32, 17) * P4) & MASK32
        cursor += 4
    while cursor < size:
        h = (rotl32((h + payload[cursor] * P5) & MASK32, 11) * P1) & MASK32
        cursor += 1
    h ^= h >> 15
    h = (h * P2) & MASK32
    h ^= h >> 13
    h = (h * P3) & MASK32
    h ^= h >> 16
    return h & MASK32


@lru_cache(maxsize=250000)
def bucket(seed: int, length: int, domain_min: int, bins: int) -> int:
    
    return xxh32(str(length - domain_min).encode("utf8"), seed) % bins


def grr_prob(epsilon: float, size: int) -> tuple[float, float]:
    e = exp(epsilon)
    return e / (e + size - 1), 1 / (e + size - 1)


def randomized_response(value: Any, domain: tuple[Any, ...], epsilon: float,
                        rng: np.random.Generator) -> Any:
    p, _ = grr_prob(epsilon, len(domain))
    if rng.random() < p:
        return value
    alternatives = [x for x in domain if x != value]
    return alternatives[int(rng.integers(len(alternatives)))]


def olh_report(value: int, seed: int, epsilon: float, domain: tuple[int, ...],
               rng: np.random.Generator) -> tuple[int, int]:
    bins = floor(exp(epsilon)) + 1
    p, q = grr_prob(epsilon, bins)
    actual = bucket(seed, value, domain[0], bins)
    
    if rng.random() > p - q:
        actual = int(rng.integers(bins))
    return seed, actual


def supported_lengths(report: tuple[int, int], domain: tuple[int, ...],
                      epsilon: float) -> tuple[int, ...]:
    seed, value = report
    bins = floor(exp(epsilon)) + 1
    return tuple(v for v in domain if bucket(seed, v, domain[0], bins) == value)


@dataclass
class Config:
    epsilon: float = 2.0
    beta: float = 0.20  
    alpha: float = 0.02  
    
    
    tree_level_fraction: float = 0.10
    length_mechanism: str = "olh"
    length_min: int = 1
    length_max: int = 10
    length_success_delta: float = 0.05
    probability_bound: str = "normal"  
    seed_search: int = 100  
    seed_space: int = 2**31
    seed_strategy_seed: int = 20231025
    reference_coverage: float = 0.70
    
    length_min_multiplier: float = 1.0
    length_max_multiplier: float = 2.0
    top_k: int = 3
    
    
    search_timeout_seconds: float = 0.0  
    
    
    planning_sample: int = 0
    score_z: float = 2.0  
    safety_share: float = 0.02
    allow_dynamic_reselect: bool = True


@dataclass
class Dataset:
    train: list[str]
    labels: list[int]
    test: list[str]
    test_labels: list[int]

    def validate(self) -> None:
        if not self.train or not self.test or len(self.train) != len(self.labels):
            raise ValueError("Need nonempty train/test sequences and matched labels")
        if len(self.test) != len(self.test_labels):
            raise ValueError("Test labels and sequences differ in length")
        if set(self.labels) != {1, 2, 3} or not set(self.test_labels) <= {1, 2, 3}:
            raise ValueError("Trace experiments expect the three labels 1, 2 and 3")
        for s in self.train + self.test:
            if not s or any(c not in ABC for c in s):
                raise ValueError(f"Invalid compressed SAX sequence: {s!r}")
            if any(a == b for a, b in zip(s, s[1:])):
                raise ValueError(f"Adjacent repeats remain in compressed SAX: {s!r}")


@dataclass(frozen=True)
class PlanningKnowledge:
    """One round's attacker-visible honest training records, never test records.

    Only attack planning may consume ``train`` and ``labels`` here. The real
    protocol always uses the original, unsampled Dataset for honest reports.
    """
    train: tuple[str, ...]
    labels: tuple[int, ...]
    indices: tuple[int, ...]
    population_size: int
    seed: int | None

    @property
    def fingerprint(self) -> str:
        return sha256(",".join(map(str, self.indices)).encode("ascii")).hexdigest()[:16]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": ("full_honest_training_population" if len(self.indices) == self.population_size
                       else "uniform_without_replacement"),
            "sample_size": len(self.indices),
            "honest_population_size": self.population_size,
            "seed": self.seed,
            "indices_fingerprint": self.fingerprint,
            "per_class_counts": {str(cls): self.labels.count(cls) for cls in (1, 2, 3)},
            "shared_by": ["A_length", "70pct_class_references", "B_position_pairs",
                          "C_prefix_expected_votes", "D_label_expected_votes"],
        }


def sample_planning_knowledge(data: Dataset, cfg: Config, seed: int
                              ) -> PlanningKnowledge:
    """One independent permutation; size-k samples are nested prefixes.

    The RNG is deliberately separate from the honest protocol's A/B/C/D
    partition and perturbation RNGs. Every attack-planning stage in a round
    receives this SAME knowledge object.
    """
    total = len(data.train)
    if cfg.planning_sample <= 0 or cfg.planning_sample >= total:
        indices = tuple(range(total))
        sample_seed: int | None = None
    else:
        indices = tuple(int(i) for i in
                        np.random.default_rng(seed).permutation(total)[:cfg.planning_sample])
        sample_seed = seed
    known = PlanningKnowledge(tuple(data.train[i] for i in indices),
                              tuple(data.labels[i] for i in indices), indices,
                              total, sample_seed)
    if set(known.labels) != {1, 2, 3}:
        raise ValueError("The randomly selected planning knowledge sample does not "
                         "contain all three Trace classes; increase --planning-sample "
                         "or change --seed. No unobserved users are added to the sample.")
    return known


def load_dataset(directory: Path) -> Dataset:
    def read_text(name: str) -> list[str]:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; provide four preprocessed Trace .txt files")
        return [x.strip() for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
    data = Dataset(read_text("data.txt"), [int(x) for x in read_text("label.txt")],
                   read_text("test_data.txt"), [int(x) for x in read_text("test_label.txt")])
    data.validate()
    return data


def demo_dataset(n: int = 1200, n_test: int = 180, seed: int = 29) -> Dataset:
    """Tiny synthetic compressed-symbol *smoke test*, not an attack result."""
    rng = np.random.default_rng(seed)
    centers = ("cdabc", "dabcd", "abcd")

    def draw(label: int) -> str:
        chars = list(centers[label - 1])
        if rng.random() < 0.50:
            i = int(rng.integers(len(chars)))
            allowed = [c for c in ABC if c != chars[i]
                       and (i == 0 or c != chars[i - 1])
                       and (i + 1 == len(chars) or c != chars[i + 1])]
            if allowed:
                chars[i] = allowed[int(rng.integers(len(allowed)))]
        if rng.random() < .20 and len(chars) > 3:
            chars.pop(int(rng.integers(1, len(chars) - 1)))
        
        while len(chars) < 10 and rng.random() < .22:
            chars.append(ABC[int(rng.choice([i for i, a in enumerate(ABC)
                                              if a != chars[-1]]))])
        return "".join(a for a, b in zip(chars, [None] + chars[:-1]) if a != b)

    labels = [1 + (i % 3) for i in range(n)]
    test_labels = [1 + (i % 3) for i in range(n_test)]
    ds = Dataset([draw(c) for c in labels], labels,
                 [draw(c) for c in test_labels], test_labels)
    ds.validate()
    return ds


@lru_cache(maxsize=500000)
def full_edit(s: str, t: str) -> float:
    """Trace's directional edit costs, without the trailing-deletion rollback.

    Use the original Trace DP's context-dependent cost for *all* three edits,
    including its unit-cost first row/column.  Unlike ``trace_distance``,
    measure the complete shape by returning dp[len(s)][len(t)] directly.

    This intentionally remains directional: ``full_edit(s, t)`` need not equal
    ``full_edit(t, s)``.  Here s is the proposed shape and t is a reference.
    """
    m, n = len(s), len(t)
    last = [float(j) for j in range(n + 1)]
    for i, a in enumerate(s, start=1):
        row = [float(i)]
        for j, b in enumerate(t, start=1):
            left = t[j - 2] if j != 1 else ""
            right = t[j] if j != n else ""

            def symdist(x: str, y: str) -> float:
                return max(1, abs(ABC.index(x) - ABC.index(y))) * .67 if x and y else .67

            weight = min(symdist(left, a), symdist(a, right))
            deletion = last[j] + weight
            insertion = row[-1] + weight
            substitution = last[j - 1] + (0 if a == b else weight)
            row.append(min(deletion, insertion, substitution))
        last = row
    return last[n]


@lru_cache(maxsize=100000)
def trace_distance(s: str, t: str) -> float:
    """Code_trace's directional weighted edit distance, including trailing-D rollback."""
    m, n = len(s), len(t)
    d = [[0.0] * (n + 1) for _ in range(m + 1)]
    op = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = float(i)
        op[i][0] = "D"
    for j in range(n + 1):
        d[0][j] = float(j)
        op[0][j] = "I"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            left = t[j - 2] if j != 1 else ""
            right = t[j] if j != n else ""
            def symdist(a: str, b: str) -> float:
                return max(1, abs(ABC.index(a) - ABC.index(b))) * .67 if a and b else .67
            weight = min(symdist(left, s[i - 1]), symdist(s[i - 1], right))
            deletion, insertion = d[i - 1][j] + weight, d[i][j - 1] + weight
            substitution = d[i - 1][j - 1] + (0 if s[i - 1] == t[j - 1] else weight)
            v = min(deletion, insertion, substitution)
            d[i][j] = v
            op[i][j] = "D" if v == deletion else ("I" if v == insertion else "S")
    i = m
    while i > 0 and op[i][n] == "D":
        i -= 1
    return d[i][n]


def em_probs(sequence: str, candidates: tuple[str, ...], epsilon: float) -> np.ndarray:
    if not candidates:
        raise ValueError("Cannot match into an empty trie level")
    scores = np.array([1 / (trace_distance(sequence[:len(c)], c) + 0.1)
                       for c in candidates], dtype=float)
    spread = float(scores.max() - scores.min())
    if spread == 0:
        return np.full(len(candidates), 1 / len(candidates))
    weights = np.exp(epsilon * (scores - scores.min()) / spread / 2)
    return weights / weights.sum()


def predict_label(representatives: list[str | None], sequence: str) -> int:
    if all(s is None for s in representatives):
        return 0
    distances = [trace_distance(sequence, s) if s is not None else float("inf")
                 for s in representatives]
    smallest = min(distances)
    best_ratio, selected = -1.0, 0
    for i, s in enumerate(representatives):
        if s is not None and distances[i] == smallest:
            ratio = SequenceMatcher(None, sequence, s).ratio()
            if ratio > best_ratio:
                best_ratio, selected = ratio, i + 1
    return selected


def accuracy(data: Dataset, representatives: list[str | None]) -> float:
    return sum(predict_label(representatives, s) == y
               for s, y in zip(data.test, data.test_labels)) / len(data.test)


def recall_by_class(data: Dataset, representatives: list[str | None]) -> dict[str, float]:
    correct, totals = Counter(), Counter(data.test_labels)
    for sequence, label in zip(data.test, data.test_labels):
        if predict_label(representatives, sequence) == label:
            correct[label] += 1
    return {str(cls): correct[cls] / totals[cls] if totals[cls] else float("nan")
            for cls in range(1, 4)}


def top_names(values: dict[str, float], limit: int) -> list[str]:
    
    
    return [k for k, _ in sorted(values.items(), key=lambda kv: -kv[1])[:limit]]


def top_nodes_in_creation_order(nodes: Iterable[str], scores: dict[str, float],
                                limit: int) -> list[str]:
    """Prune by count, then retain Trace's original C-node creation order.

    Original Trace sorts `levels_node[level]` only to decide what to prune;
    it does not reorder that list before expanding the next layer or D's
    final candidate enumeration.  Both membership and original order matter.
    """
    original = list(dict.fromkeys(nodes))
    kept = set(top_names({node: scores[node] for node in original}, limit))
    return [node for node in original if node in kept]


def honest_length_moments(pi: dict[int, float], n_a: int, epsilon: float,
                          domain: tuple[int, ...], mechanism: str,
                          target: int, rival: int, population_n: int) -> tuple[float, float]:
    """Exact mean/variance under a fixed finite-population length histogram.

    User slots are assigned seeds 0..n_a-1, with normal users sampled without
    replacement.  Matches the moment derivation in the supplied A-stage plan.
    """
    d = len(domain)
    a = np.zeros((n_a, d))
    second = np.zeros((n_a, d))
    if mechanism == "grr":
        p, q = grr_prob(epsilon, d)
        for i in range(n_a):
            for k, v in enumerate(domain):
                a[i, k] = (p - q) * (int(v == rival) - int(v == target))
                second[i, k] = 2 * q + (p - q) * int(v in (rival, target))
    else:
        g = floor(exp(epsilon)) + 1
        p, q = grr_prob(epsilon, g)
        for i in range(n_a):
            tb = bucket(i, target, domain[0], g)
            rb = bucket(i, rival, domain[0], g)
            if tb == rb:
                continue
            for k, v in enumerate(domain):
                vb = bucket(i, v, domain[0], g)
                a[i, k] = (p - q) * (int(vb == rb) - int(vb == tb))
                second[i, k] = (2 * q + (p - q) * int(vb in (rb, tb)))
    weights = np.array([pi.get(v, 0.0) for v in domain])
    avg = a @ weights
    mean = float(avg.sum())
    variance = float((second @ weights - avg ** 2).sum())
    
    if population_n > 1 and n_a > 1:
        sums = a.sum(axis=0)
        pair_a = float(np.dot(weights, sums ** 2 - (a ** 2).sum(axis=0)))
        pair_avg = float(mean ** 2 - (avg ** 2).sum())
        variance -= (pair_a - pair_avg) / (population_n - 1)
    return mean, sqrt(max(0.0, variance))


def honest_length_support_mean(pi: dict[int, float], n_a: int, epsilon: float,
                               domain: tuple[int, ...], mechanism: str,
                               queried: int) -> float:
    if mechanism == "grr":
        p, q = grr_prob(epsilon, len(domain))
        return n_a * (q + (p - q) * pi.get(queried, 0.0))
    g = floor(exp(epsilon)) + 1
    p, q = grr_prob(epsilon, g)
    return sum(q + (p - q) * sum(
        pi.get(v, 0.0) for v in domain
        if bucket(slot, v, domain[0], g) == bucket(slot, queried, domain[0], g))
        for slot in range(n_a))


def length_reference(data: Dataset | PlanningKnowledge, coverage: float = .70
                     ) -> tuple[dict[int, tuple[tuple[str, float], ...]], dict[int, int]]:
    """Smallest frequency-ordered full-shape set covering >=70% of EACH class."""
    if not 0 < coverage <= 1:
        raise ValueError("Reference coverage must be in (0, 1]")
    refs: dict[int, tuple[tuple[str, float], ...]] = {}
    highest: dict[int, int] = {}
    for cls in (1, 2, 3):
        counter = Counter(s for s, c in zip(data.train, data.labels) if c == cls)
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        chosen: list[tuple[str, int]] = []
        total = 0
        for shape, frequency in ordered:
            chosen.append((shape, frequency))
            total += frequency
            if total / sum(counter.values()) >= coverage - 1e-12:
                break
        if not chosen:
            raise ValueError(f"Class {cls} has no reference shapes")
        refs[cls] = tuple((s, n / total) for s, n in chosen)
        highest[cls] = ceil(sum(n * len(s) for s, n in chosen) / total)
    return refs, highest


def select_length_output(votes: Counter, domain: tuple[int, ...]) -> int:
    
    
    
    
    ranked = sorted(votes, key=lambda v: -votes[v])
    
    
    return ranked[0] if ranked else domain[0]


@dataclass
class LengthPlan:
    target: int | None = None
    reports: list[Any] = field(default_factory=list)
    min_required: int = 0
    bounds: dict[int, float] = field(default_factory=dict)
    net_gains: dict[int, int] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    reference_min: int | None = None
    knowledge_fingerprint: str | None = None


def length_net_gains(reports: Iterable[Any], target: int,
                     domain: tuple[int, ...], mechanism: str,
                     epsilon: float) -> dict[int, int]:
    """Actual malicious support *difference* against every other length."""
    gained = Counter()
    for report in reports:
        if mechanism == "grr":
            support = (report,)
        else:
            support = supported_lengths(report, domain, epsilon)
        gained.update(support)
    return {rival: gained[target] - gained[rival]
            for rival in domain if rival != target}


def olh_remaining_deficit_score(deficits: dict[int, int],
                                supported: Iterable[int]) -> int:
    """Weighted support improvement: a large outstanding rival gap matters more."""
    collisions = set(supported)
    return sum(max(0, due) for rival, due in deficits.items()
               if rival not in collisions)


def sample_identity_seed_pools(cfg: Config, count: int) -> tuple[tuple[int, ...], ...]:
    """Sample once per hypothetical identity, reusing its pool for every A target.

    A comparison between different target lengths must not grant an identity
    another ``seed_search`` attempts. Each pool contains distinct seeds, but
    different identities may independently draw overlapping pools.
    """
    rng = np.random.default_rng(cfg.seed_strategy_seed)
    limit = min(cfg.seed_search, cfg.seed_space)
    pools: list[tuple[int, ...]] = []
    for _ in range(count):
        seeds: set[int] = set()
        while len(seeds) < limit:
            seeds.add(int(rng.integers(cfg.seed_space)))
        pools.append(tuple(sorted(seeds)))
    return tuple(pools)


def length_plan(data: Dataset, cfg: Config, fake_total: int, n_a: int,
                available: int | None = None,
                excluded_lengths: frozenset[int] = frozenset(),
                knowledge: PlanningKnowledge | None = None) -> LengthPlan:
    """Prefer highly deviant lengths, then plan GRR/OLH net-support reports."""
    known = (knowledge if knowledge is not None else
             sample_planning_knowledge(data, cfg, cfg.seed_strategy_seed + 707))
    domain = tuple(range(cfg.length_min, cfg.length_max + 1))
    histogram = Counter(min(cfg.length_max, max(cfg.length_min, len(s))) for s in known.train)
    pi = {v: histogram[v] / len(known.train) for v in domain}
    references, H = length_reference(known, cfg.reference_coverage)
    
    
    reference_min = min(H.values())
    bmin = ceil(cfg.length_min_multiplier * cfg.alpha * fake_total)
    bmax = min(floor(cfg.length_max_multiplier * cfg.alpha * fake_total),
               fake_total if available is None else available)
    response = LengthPlan(reference_min=reference_min,
                          knowledge_fingerprint=known.fingerprint)
    if bmax < bmin or bmax == 0:
        response.attempts.append({"reason": "infeasible A budget bounds"})
        return response
    
    
    expected_support = {v: honest_length_support_mean(pi, n_a, cfg.epsilon,
                                                      domain, cfg.length_mechanism, v)
                        for v in domain}
    clean_mode = sorted(domain, key=lambda v: (-expected_support[v], v))[0]
    feasible_lengths = [v for v in domain if reference_min <= v < cfg.length_max
                        and v != clean_mode and v not in excluded_lengths]
    if not feasible_lengths:
        response.attempts.append({
            "reason": "No eligible length at or above global reference minimum, "
                      "different from the clean mode, and below the downstream upper bound",
            "reference_min": reference_min, "clean_mode": clean_mode})
        return response
    preference = sorted(feasible_lengths, key=lambda v: (
        -sum(abs(v - H[c]) for c in H), -expected_support[v], v))
    seed_pools = (sample_identity_seed_pools(cfg, bmax)
                  if cfg.length_mechanism == "olh" else ())
    delta_comp = cfg.length_success_delta / (len(domain) - 1)
    factor = (norm.ppf(1 - delta_comp) if cfg.probability_bound == "normal"
              else sqrt((1 - delta_comp) / delta_comp))
    g = floor(exp(cfg.epsilon)) + 1
    for t in preference:
        if t < 2:
            
            
            response.attempts.append({"length": t, "reference_min": reference_min,
                                      "reason": "Trace B needs at least one length-2 pair",
                                      "feasible": False})
            continue
        gaps: dict[int, int] = {}
        for rival in domain:
            if rival == t:
                continue
            mu, sigma = honest_length_moments(pi, n_a, cfg.epsilon, domain,
                                              cfg.length_mechanism, t, rival, len(data.train))
            gaps[rival] = max(0, ceil(mu + factor * sigma + 1))
        reports: list[Any] = []
        if cfg.length_mechanism == "grr":
            reports = [t] * max(gaps.values(), default=0)
        else:
            
            
            
            still_due = dict(gaps)
            used_seeds: set[int] = set()
            for identity, local_seeds in enumerate(seed_pools):
                if identity >= bmin and all(x == 0 for x in still_due.values()):
                    break
                best: tuple[tuple[int, int, int, int], tuple[int, int],
                            tuple[int, ...]] | None = None
                for seed in local_seeds:
                    report = (seed, bucket(seed, t, domain[0], g))
                    support = supported_lengths(report, domain, cfg.epsilon)
                    
                    
                    
                    improvement = olh_remaining_deficit_score(still_due, support)
                    rank = (improvement, int(seed not in used_seeds),
                            -len(support), -seed)
                    if best is None or rank > best[0]:
                        best = (rank, report, support)
                if best is None:
                    break
                _, report, support = best
                reports.append(report)
                used_seeds.add(report[0])
                for rival in still_due:
                    if rival not in support:
                        still_due[rival] = max(0, still_due[rival] - 1)
        required = len(reports)
        if required <= bmax and len(reports) < bmin:
            if cfg.length_mechanism == "grr":
                fill = t
            else:
                
                
                fill = None
            if fill is not None:
                reports.extend([fill] * (bmin - len(reports)))
        gains = length_net_gains(reports, t, domain, cfg.length_mechanism, cfg.epsilon)
        deficits = {rival: gaps[rival] - gains[rival] for rival in gaps
                    if gains[rival] < gaps[rival]}
        feasible = bmin <= len(reports) <= bmax and not deficits
        response.attempts.append({"length": t, "reference_min": reference_min,
                                  "deviation": sum(abs(t - H[c]) for c in H),
                                  "reference_weighted_lengths": dict(H),
                                  "seed_space": cfg.seed_space,
                                  "seeds_checked_per_identity": cfg.seed_search,
                                  "seed_pools_reused_across_lengths": (
                                      cfg.length_mechanism == "olh"),
                                  "seed_choice_score": "weighted_remaining_net_support_gap",
                                  "distinct_seeds": (len({r[0] for r in reports})
                                                     if cfg.length_mechanism == "olh" else None),
                                  "normal_expected_support": expected_support[t],
                                  "requirements": gaps, "net_gains": gains,
                                  "deficits": deficits, "planned": len(reports),
                                  "within_budget": bmin <= len(reports) <= bmax,
                                  "feasible": feasible})
        if feasible:
            response.target, response.reports, response.min_required = t, reports, required
            response.bounds = {k: float(v) for k, v in gaps.items()}
            response.net_gains = gains
            return response
    response.attempts.append({"reason": "No candidate satisfies every net-support "
                                        "inequality within the finite seed and identity budgets"})
    return response


@dataclass(frozen=True)
class Target:
    cls: int
    prefix: str
    full: str
    deviation: float
    margin: float
    nearest_margin: float
    extension_edges: int
    normalized_margin: float = 0.0
    preferred: bool = False
    class_distances: tuple[float, ...] = ()
    bottleneck_support_ratio: float | None = None
    path_count: int = 0
    
    full_deviation: float = 0.0
    full_margin: float = 0.0
    full_normalized_margin: float = 0.0
    full_nearest_margin: float = 0.0
    full_class_distances: tuple[float, ...] = ()


def initially_selected_classes_succeeded(
        initial_targets: dict[int, Target], final_targets: dict[int, Target],
        final_target_hits: dict[str, bool]) -> bool | None:
    """Whether EVERY originally selected CLASS succeeded after legal reselection.

    This is intentionally distinct from the unchanged initial-shape hit test:
    a legitimate replacement can succeed without reproducing the old shape,
    while an abandoned class can accidentally output its old shape without
    having an active, successful attack target.
    """
    if not initial_targets:
        return None
    return all(cls in final_targets and final_target_hits.get(str(cls), False)
               for cls in initial_targets)


@dataclass
class CandidatePools:
    priority: dict[int, list[Target]] = field(default_factory=dict)
    qualified: dict[int, list[Target]] = field(default_factory=dict)
    diagnostics: dict[int, dict[str, Any]] = field(default_factory=dict)
    search_complete: bool = True


def ref_dist(s: str, refs: tuple[tuple[str, float], ...], normalize: bool = False) -> float:
    return sum(w * (full_edit(s, r) / max(len(s), len(r)) if normalize else full_edit(s, r))
               for r, w in refs)


def class_diagnostics(s: str, cls: int,
                      references: dict[int, tuple[tuple[str, float], ...]]) -> tuple[float, float, float, float]:
    distances = {r: ref_dist(s, ref) for r, ref in references.items()}
    normalized = {r: ref_dist(s, ref, True) for r, ref in references.items()}
    other = [r for r in references if r != cls]
    margin = min(distances[r] for r in other) - distances[cls]
    nmargin = min(normalized[r] for r in other) - normalized[cls]
    near_self = min(full_edit(s, r) for r, _ in references[cls])
    near_other = min(full_edit(s, r) for c in other for r, _ in references[c])
    return distances[cls], margin, nmargin, near_other - near_self


def position_expected(data: Dataset | PlanningKnowledge, L: int, n_b: int,
                      epsilon: float) -> dict[int, dict[str, float]]:
    """GRR expectation with uniform sampling from each sequence's valid positions."""
    p, q = grr_prob(epsilon, len(PAIRS))
    valid = defaultdict(float)
    raw = defaultdict(lambda: defaultdict(float))
    for sequence, multiplicity in Counter(data.train).items():
        clipped = sequence[:L]
        if len(clipped) < 2:
            continue
        for pos in range(len(clipped) - 1):
            share = multiplicity / len(data.train) / (len(clipped) - 1)
            valid[pos] += share
            raw[pos][clipped[pos:pos + 2]] += share
    return {pos: {pair: n_b * (q * valid[pos] + (p - q) * raw[pos][pair])
                  for pair in PAIRS} for pos in range(L - 1)}


def pair_graph_from_counts(counts: dict[int, dict[str, float]], L: int,
                           how_many: int) -> dict[int, tuple[str, ...]]:
    graph = {}
    for pos in range(L - 1):
        
        
        ranked = sorted(PAIRS, key=lambda e: -counts.get(pos, {}).get(e, 0.0))
        graph[pos] = tuple(ranked[:how_many])
    return graph


def optimistic_pair_costs(expectations: dict[int, dict[str, float]],
                          cfg: Config) -> dict[int, dict[str, int]]:
    """Model-conditional NECESSARY B votes for a single pair to reach Top-6.

    Competing pairs use their optimistic lower bounds, and the target uses
    its optimistic upper bound. This is not a guarantee over random reports.
    """
    k = 2 * cfg.top_k
    result: dict[int, dict[str, int]] = {}
    for pos, means in expectations.items():
        lower = {e: max(0.0, means.get(e, 0.0) -
                        cfg.score_z * sqrt(max(0.0, means.get(e, 0.0))))
                 for e in PAIRS}
        result[pos] = {}
        for edge in PAIRS:
            others = sorted((lower[e] for e in PAIRS if e != edge), reverse=True)
            threshold = others[k - 1] if len(others) >= k else 0.0
            mu = max(0.0, means.get(edge, 0.0))
            upper = mu + cfg.score_z * sqrt(mu)
            
            result[pos][edge] = max(0, ceil(threshold - upper - 1e-12))
    return result


def candidate_pair_graph(expectations: dict[int, dict[str, float]], L: int,
                         cfg: Config, optimistic_b_budget: int
                         ) -> tuple[dict[int, tuple[str, ...]], dict[int, dict[str, int]]]:
    bounds = optimistic_pair_costs(expectations, cfg)
    return {
        pos: tuple(sorted(
            (edge for edge in PAIRS if bounds[pos][edge] <= optimistic_b_budget),
            key=lambda e: (-expectations[pos].get(e, 0.0), PAIRS.index(e))))
        for pos in range(L - 1)
    }, bounds


def expand_paths(paths: Iterable[str], graph: dict[int, tuple[str, ...]],
                 depth: int) -> list[str]:
    """depth=0 creates the first symbol; later depths extend each prefix."""
    if depth == 0:
        return list(dict.fromkeys(pair[0] for pair in graph.get(0, ())))
    allowed = graph.get(depth - 1, ())
    return list(dict.fromkeys(path + pair[1]
                              for path in paths for pair in allowed
                              if path[-1] == pair[0]))


def descendants(prefix: str, L: int, graph: dict[int, tuple[str, ...]],
                limit: int | None = None) -> list[str]:
    paths = [prefix]
    for pos in range(len(prefix) - 1, L - 1):
        pairs = graph.get(pos, ())
        paths = [s + pair[1] for s in paths for pair in pairs if pair[0] == s[-1]]
        if not paths:
            return []
        if limit is not None:
            paths = paths[:limit]
    return paths


def diverse_priority_targets(possible: list[Target], limit: int,
                             per_prefix_limit: int = 2
                             ) -> tuple[list[Target], dict[str, Any]]:
    """Choose a Pareto-stratified round-robin display shortlist.

    Four incomparable benefits are retained: larger COMPLETE-leaf deviation,
    larger qualified-prefix class margin, better weakest-link support,
    fewer extended edges.  Distinct prefix coverage remains the first step.
    The shortlist is only a *display/search source*, not the main-target rank.
    Strictly screened (preferred) items precede basic-only fallbacks.
    """
    def score(t: Target) -> tuple[float, float, float, float]:
        return (t.full_deviation, t.margin,
                float(t.bottleneck_support_ratio or 0.0),
                -float(t.extension_edges))

    def dominates(a: Target, b: Target) -> bool:
        x, y = score(a), score(b)
        return all(p >= q - 1e-12 for p, q in zip(x, y)) and any(
            p > q + 1e-12 for p, q in zip(x, y))

    unique_full: dict[str, Target] = {}
    for t in possible:
        unique_full.setdefault(t.full, t)
    
    
    grouped: dict[str, list[Target]] = defaultdict(list)
    for t in unique_full.values():
        grouped[t.prefix].append(t)
    prepared = []
    for prefix_index, prefix in enumerate(sorted(grouped)):
        candidates = grouped[prefix]
        chosen: list[Target] = []
        
        
        
        directions = [(prefix_index + axis) % 4 for axis in range(4)]
        for strictly_preferred in (True, False):
            subset = [t for t in candidates if t.preferred == strictly_preferred]
            rankings = [sorted(subset, key=lambda t, axis=axis:
                               (-score(t)[axis], t.full)) for axis in directions]
            while len(chosen) < per_prefix_limit and any(rankings):
                progress = False
                for ranked in rankings:
                    first = next((t for t in ranked if t not in chosen), None)
                    if first is not None:
                        chosen.append(first)
                        progress = True
                        if len(chosen) == per_prefix_limit:
                            break
                if not progress:
                    break
            if len(chosen) == per_prefix_limit:
                break
        prepared.extend(chosen)

    result: list[Target] = []
    layers = 0

    def append_pareto(items: list[Target]) -> None:
        nonlocal layers
        for preferred in (True, False):
            pending = [t for t in items if t.preferred == preferred
                       and t.full not in {x.full for x in result}]
            while pending and len(result) < limit:
                front = [t for t in pending if not any(
                    dominates(other, t) for other in pending if other != t)]
                if not front:
                    raise RuntimeError("Pareto front construction failed")
                layers += 1
                queues = [sorted(front, key=lambda t, axis=axis: (
                    -score(t)[axis], t.full)) for axis in range(4)]
                selected: set[str] = set()
                offset = (layers - 1) % 4
                while len(selected) < len(front) and len(result) < limit:
                    advanced = False
                    for direction in range(4):
                        queue = queues[(offset + direction) % 4]
                        first = next((t for t in queue if t.full not in selected), None)
                        if first is not None:
                            result.append(first)
                            selected.add(first.full)
                            advanced = True
                            if len(result) >= limit:
                                break
                    if not advanced:
                        break
                removed = {t.full for t in front}
                pending = [t for t in pending if t.full not in removed]

    
    
    champions = list(dict.fromkeys(
        next(t for t in prepared if t.prefix == prefix)
        for prefix in sorted({t.prefix for t in prepared})))
    append_pareto(champions)
    
    if len(result) < limit:
        append_pareto(prepared)
    return result, {"pareto_layers_examined": layers,
                    "priority_distinct_prefixes": len({t.prefix for t in result}),
                    "qualified_distinct_prefixes": len(grouped),
                    "priority_prefix_coverage": (
                        len({t.prefix for t in result}) / len(grouped) if grouped else 0),
                    "priority_preferred": sum(t.preferred for t in result),
                    "shortlist_policy": "distinct_prefixes_first_then_sibling_leaves",
                    "priority_objectives": ["full_leaf_deviation", "prefix_class_margin",
                                            "weakest_local_support", "few_extended_edges"]}


def target_pool(data: Dataset | PlanningKnowledge, cfg: Config, L: int,
                expectations: dict[int, dict[str, float]],
                references: dict[int, tuple[tuple[str, float], ...]],
                lengths: dict[int, int], optimistic_b_budget: int = 0,
                graph_override: dict[int, tuple[str, ...]] | None = None
                ) -> CandidatePools:
    """Uncapped legal paths through position-GRR budget-conditioned edges.

    All 70%-reference prefix/full-leaf margins must be positive. For the
    *initial* plan, prune only when SUM of model-conditional minimum B costs
    already exceeds the optimistic B-identity budget. After the actual B
    selection, its public graph is authoritative and historical B costs are 0.
    """
    graph, edge_lower = candidate_pair_graph(expectations, L, cfg, optimistic_b_budget)
    if graph_override is not None:
        graph = graph_override
    lower = (edge_lower if graph_override is None else
             {j: {e: 0 for e in PAIRS} for j in range(max(0, L - 1))})
    pools = CandidatePools()
    if L < 2:
        for cls in references:
            pools.priority[cls] = pools.qualified[cls] = []
            pools.diagnostics[cls] = {"reason": "no_length_two_position_pair",
                                       "search_complete": True}
        return pools
    
    
    
    core = pair_graph_from_counts(expectations, L, 2 * cfg.top_k)
    ranked_sixths = [sorted(expectations[pos].values(), reverse=True)[5]
                     if len(expectations[pos]) >= 6 else 0.0
                     for pos in range(L - 1)]
    deadline = (time.monotonic() + cfg.search_timeout_seconds
                if cfg.search_timeout_seconds > 0 else None)

    def legal_paths() -> Iterable[str]:
        first_chars = list(dict.fromkeys(edge[0] for edge in graph.get(0, ())))
        stack = [(first, 0) for first in reversed(first_chars)]
        while stack:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Candidate graph enumeration interrupted")
            prefix, spent = stack.pop()
            if len(prefix) == L:
                yield prefix
                continue
            pos = len(prefix) - 1
            extensions = []
            for edge in graph.get(pos, ()):
                if edge[0] != prefix[-1]:
                    continue
                cost = spent + lower[pos][edge]
                if graph_override is not None or cost <= optimistic_b_budget:
                    extensions.append((prefix + edge[1], cost))
            stack.extend(reversed(extensions))

    for cls in sorted(references):
        T = min(L, lengths[cls])
        prefix_cache: dict[str, tuple[float, float, float, float]] = {}
        examined = full_passed = nonpositive_prefix = nonpositive_full = 0
        found: list[Target] = []
        ref_shapes = {s for s, _ in references[cls]}
        try:
            for full in legal_paths():
                prefix = full[:T]
                if prefix not in prefix_cache:
                    prefix_cache[prefix] = class_diagnostics(prefix, cls, references)
                d, margin, nmargin, near = prefix_cache[prefix]
                examined += 1
                if margin <= 1e-12:
                    nonpositive_prefix += 1
                    continue
                fd, fm, fnm, fn = class_diagnostics(full, cls, references)
                if fm <= 1e-12:
                    nonpositive_full += 1
                    continue
                full_passed += 1
                ratios = [(expectations[pos].get(full[pos:pos + 2], 0.0) / sixth)
                          for pos, sixth in enumerate(ranked_sixths) if sixth > 0]
                preferred = nmargin > 1e-12 and near > 1e-12 and prefix not in ref_shapes
                found.append(Target(
                    cls, prefix, full, d, margin, near,
                    sum(full[pos:pos + 2] not in core[pos] for pos in range(L - 1)),
                    nmargin, preferred,
                    tuple(ref_dist(prefix, references[c]) for c in sorted(references)),
                    min(ratios) if len(ratios) == L - 1 else None,
                    0, fd, fm, fnm, fn,
                    tuple(ref_dist(full, references[c]) for c in sorted(references))))
        except TimeoutError:
            pools.search_complete = False
        counts = Counter(t.prefix for t in found)
        found = [replace(t, path_count=counts[t.prefix]) for t in found]
        found.sort(key=lambda t: (-t.full_deviation, -t.margin, t.full))
        pools.qualified[cls] = found
        pools.priority[cls] = found  
        pools.diagnostics[cls] = {
            "prefix_length": T, "examined_complete_paths": examined,
            "legal_complete_paths": examined if pools.search_complete else None,
            "paths_enumerated": examined,
            "examined_prefixes": len(prefix_cache),
            "nonpositive_margin": nonpositive_prefix,
            "nonpositive_complete_margin": nonpositive_full,
            "qualifying_prefixes": len(counts),
            "qualified_full_leaves": len(found),
            "candidate_pair_counts": {str(j): len(v) for j, v in graph.items()},
            "optimistic_b_identity_budget": optimistic_b_budget,
            "search_complete": pools.search_complete,
            "qualified_scope": "All budget-unexcluded position paths; no fixed cap"}
    return pools


def supports_to_top_k(items: tuple[str, ...], protected: Iterable[str],
                      mean: dict[str, float], population: int, k: int,
                      z: float) -> dict[str, int] | None:
    """Planning heuristic; all protected choices beat kth outside upper bound.

    Uses marginal +/- z sigma, NOT a joint probability guarantee. Incoming
    malicious reports support exactly one pair/node in B or C.
    """
    needed = tuple(dict.fromkeys(protected))
    if not set(needed) <= set(items) or len(needed) > k:
        return None
    outside = [v for v in items if v not in needed]
    if len(outside) <= k - len(needed):
        return {v: 0 for v in needed}
    upper = {v: mean.get(v, 0.0) + z * sqrt(max(0.0, mean.get(v, 0.0)))
             for v in outside}
    bound = sorted(upper.values(), reverse=True)[k - len(needed)]
    result = {}
    for v in needed:
        lower = mean.get(v, 0.0) - z * sqrt(max(0.0, mean.get(v, 0.0)))
        result[v] = max(0, ceil(bound + 1 - lower))
    return result


def guard_stage_votes(additions: dict[str, int] | None,
                      future_cost: int | None, remaining: int,
                      safety: int) -> tuple[dict[str, int], str | None]:
    """Spend only if current and all remaining C/D stages have a feasible plan.

    A missing forecast is an *infeasibility*, never a zero-cost reservation.
    """
    if additions is None:
        return {}, "current_topk_infeasible"
    if future_cost is None:
        return {}, "future_cd_infeasible"
    if sum(additions.values()) + future_cost + safety > remaining:
        return {}, "insufficient_remaining_budget"
    return additions, None


def position_topups(expectations: dict[int, dict[str, float]], full_targets: tuple[str, ...],
                    cfg: Config, n_b: int) -> dict[int, dict[str, int]] | None:
    attack: dict[int, dict[str, int]] = {}
    if not full_targets:
        return {j: {} for j in expectations}
    for pos in expectations:
        required = [path[pos:pos + 2] for path in full_targets]
        result = supports_to_top_k(PAIRS, required, expectations[pos], n_b,
                                   2 * cfg.top_k, cfg.score_z)
        if result is None:
            return None
        attack[pos] = result
    return attack


def apply_expected_pair_topups(expectations: dict[int, dict[str, float]],
                               injections: dict[int, dict[str, int]], L: int,
                               limit: int) -> dict[int, tuple[str, ...]]:
    aggregate = {pos: {s: score + injections.get(pos, {}).get(s, 0)
                       for s, score in counts.items()}
                 for pos, counts in expectations.items()}
    return pair_graph_from_counts(aggregate, L, limit)


def projected_oue_means(data: Dataset | PlanningKnowledge, candidates: tuple[str, ...],
                        n_d: int, cfg: Config, sample: list[int] | None = None
                        ) -> dict[tuple[str, int], float]:
    q = 1 / (exp(cfg.epsilon) + 1)
    p = 0.5
    width = max(map(len, candidates))
    if sample is None:
        population = Counter((s[:width], c) for s, c in zip(data.train, data.labels))
    else:
        population = Counter((data.train[i][:width], data.labels[i]) for i in sample)
    totals = Counter()
    for (sequence, label), multiplicity in population.items():
        chosen = min(candidates, key=lambda s: trace_distance(sequence[:len(s)], s))
        totals[(chosen, label)] += multiplicity
    denom = max(1, sum(population.values()))
    return {(shape, cls): n_d * (q + (p - q) * totals[(shape, cls)] / denom)
            for shape in candidates for cls in range(1, cfg.top_k + 1)}


@dataclass
class DPlan:
    increments: dict[tuple[str, int], int] = field(default_factory=dict)
    identities: int = 0
    max_ones: int = 0
    selected_order: list[int] = field(default_factory=list)
    feasible: bool = True


def oue_ones_cap(K: int, candidate_count: int, epsilon: float) -> int:
    """Attacker's chosen OUE sparsity cap; at least one slot is always allowed.

    This is NOT a hard property of honest OUE and cannot imply undetectability.
    """
    if K < 1 or candidate_count < 1 or epsilon <= 0:
        raise ValueError("OUE cap requires positive K, candidate count and epsilon")
    d = K * candidate_count
    return max(1, floor(0.5 + (d - 1) / (exp(epsilon) + 1)))


def d_ratio_reserve(honest_total: int, honest_a: int, honest_d: int,
                    fake_total: int, fake_a: int) -> tuple[float, int]:
    """Trace's L-dependent D reservation after A, not a D spending ceiling."""
    if (honest_total <= honest_a or honest_d < 0 or honest_d > honest_total - honest_a
            or fake_a < 0 or fake_a > fake_total):
        raise ValueError("Invalid honest D ratio or malicious A allocation")
    gamma = honest_d / (honest_total - honest_a)
    return gamma, ceil(gamma * (fake_total - fake_a))


def choose_oue_plan(candidates: tuple[str, ...], desired: dict[int, str],
                    projected_means: dict[tuple[str, int], float], cfg: Config
                    ) -> DPlan:
    """Sufficient conditions for Trace's global greedy candidate/class matching.

    Reverse-order construction handles later target cells as competitors.
    It can be conservative and does not claim a globally optimal OUE attack.
    """
    K, M = cfg.top_k, len(candidates)
    if (not 1 <= len(desired) <= K or len(set(desired.values())) != len(desired)
            or not set(desired.values()) <= set(candidates)):
        return DPlan(feasible=False)
    
    b = oue_ones_cap(K, M, cfg.epsilon)
    choices: list[DPlan] = []
    for order in permutations(sorted(desired)):
        increments: dict[tuple[str, int], int] = {}
        for index in range(len(order) - 1, -1, -1):
            cls = order[index]
            shape = desired[cls]
            prior_classes = set(order[:index])
            prior_shapes = {desired[x] for x in prior_classes}
            eligible = [(s, c) for s in candidates if s not in prior_shapes
                        for c in range(1, K + 1) if c not in prior_classes]
            competitors = [cell for cell in eligible if cell != (shape, cls)]
            upper = max((projected_means[cell] + increments.get(cell, 0)
                         + cfg.score_z * sqrt(max(0.0, projected_means[cell]))
                         for cell in competitors), default=0.0)
            lower = projected_means[(shape, cls)] - cfg.score_z * sqrt(
                max(0.0, projected_means[(shape, cls)]))
            increments[(shape, cls)] = max(0, ceil(upper + 1 - lower))
        votes = list(increments.values())
        identities = max(max(votes), ceil(sum(votes) / b)) if votes else 0
        choices.append(DPlan(increments, identities, b, list(order)))
    if not choices:
        return DPlan(feasible=False)
    return min(choices, key=lambda plan: (plan.identities, sum(plan.increments.values())))


def pack_oue(plan: DPlan, candidates: tuple[str, ...], K: int) -> list[set[tuple[str, int]]]:
    """Pack increments into at most b distinct active cells per fake identity."""
    if not plan.feasible:
        return []
    vectors: list[set[tuple[str, int]]] = [set() for _ in range(plan.identities)]
    for cell, votes in sorted(plan.increments.items(), key=lambda v: -v[1]):
        for _ in range(votes):
            positions = [i for i, bits in enumerate(vectors)
                         if len(bits) < plan.max_ones and cell not in bits]
            if not positions:
                raise ValueError("OUE packing infeasible at the requested identity bound")
            vectors[min(positions, key=lambda i: (len(vectors[i]), i))].add(cell)
    return vectors


def distribute_groups(n: int, cfg: Config, seed: int, length: int = 7
                      ) -> dict[str, list[int]]:
    """Original Trace sampling: A=.02N, B=.08N, C=.10 of post-AB per level.

    The source first selects 10% for knowledge estimation, then 20% of
    that for A and 80% for B.  The trie selects 10% of the post-knowledge
    population at *each* of `length` levels; remaining users enter D.
    Rounding is explicit and each genuine user appears at most once.
    """
    rng = np.random.default_rng(seed)
    users = list(map(int, rng.permutation(n)))
    knowledge_n = floor(0.10 * n)
    na = floor(0.20 * knowledge_n)
    nb = knowledge_n - na
    after_knowledge = n - knowledge_n
    level_n = floor(cfg.tree_level_fraction * after_knowledge)
    nc = length * level_n
    if min(na, nb, level_n) < 1 or na + nb + nc > n:
        raise ValueError("Not enough training users for disjoint four-way sampling")
    return {"A": users[:na], "B": users[na:na + nb],
            "C": users[na + nb:na + nb + nc], "D": users[na + nb + nc:]}


def split_users(users: list[int], L: int) -> list[list[int]]:
    return [list(map(int, part)) for part in np.array_split(np.array(users, dtype=int), L)]


@dataclass
class JointPlan:
    targets: dict[int, Target] = field(default_factory=dict)
    b_votes: dict[int, dict[str, int]] = field(default_factory=dict)
    predicted_cost: int | None = None
    checked_combinations: int = 0
    reason: str = ""
    total_combinations: int = 0
    search_limited: bool = False
    search_complete: bool = True
    search_diagnostics: dict[str, Any] = field(default_factory=dict)
    candidate_source: str = ""
    full_score: float | None = None
    d_ratio: float = 0.0
    d_ratio_reserve: int = 0
    d_locked: int = 0
    d_predicted: int = 0


class Planner:
    """Population-model planning; no access to current honest reports."""

    def __init__(self, data: Dataset, cfg: Config, seed: int = 9128,
                 knowledge: PlanningKnowledge | None = None):
        self.data, self.cfg = data, cfg
        self.knowledge = (knowledge if knowledge is not None else
                          sample_planning_knowledge(data, cfg, seed))
        self.references, self.ref_lengths = length_reference(
            self.knowledge, cfg.reference_coverage)
        
        
        self.sample = (None if len(self.knowledge.train) == len(data.train)
                       else list(self.knowledge.indices))
        self.empirical_sequences = Counter(self.knowledge.train)
        self.empirical_size = sum(self.empirical_sequences.values())
        self.prefix_frequencies: dict[int, Counter] = {}
        self.em_cache: dict[tuple[tuple[str, ...], int], dict[str, float]] = {}
        self.d_cache: dict[tuple[str, ...], dict[tuple[str, int], float]] = {}
        self.pools: dict[int, list[Target]] = {}
        self.qualified_pools: dict[int, list[Target]] = {}
        self.pool_diagnostics: dict[int, dict[str, Any]] = {}
        self.expectations: dict[int, dict[str, float]] = {}
        self.L = 0
        self.n_b = self.n_c = self.n_d = 0
        self.d_gamma = 0.0
        self.d_base_reserve = 0
        self.d_locked = 0
        self.optimistic_b_budget = 0
        self.pair_lower: dict[int, dict[str, int]] = {}
        self.candidate_graph: dict[int, tuple[str, ...]] = {}
        self.pool_search_complete = True

    def setup(self, L: int, n_b: int, n_c: int, n_d: int,
              *, fake_total: int | None = None, fake_a: int = 0,
              honest_a: int | None = None, build_candidates: bool = True) -> None:
        self.L, self.n_b, self.n_c, self.n_d = L, n_b, n_c, n_d
        self.d_gamma = 0.0
        self.d_base_reserve = 0
        self.d_locked = 0
        if fake_total is not None:
            self.d_gamma, self.d_base_reserve = d_ratio_reserve(
                len(self.data.train),
                floor(0.2 * floor(0.1 * len(self.data.train)))
                if honest_a is None else honest_a,
                n_d, fake_total, fake_a)
        self.expectations = position_expected(self.knowledge, L, n_b,
                                              self.cfg.epsilon)
        self.optimistic_b_budget = max(0, (fake_total or 0) - fake_a -
                                       self.d_base_reserve)
        self.candidate_graph, self.pair_lower = candidate_pair_graph(
            self.expectations, L, self.cfg, self.optimistic_b_budget)
        candidate_pools = (
            target_pool(self.knowledge, self.cfg, L, self.expectations, self.references,
                        self.ref_lengths, self.optimistic_b_budget)
            if build_candidates else CandidatePools())
        self.pools = candidate_pools.priority
        self.qualified_pools = candidate_pools.qualified
        self.pool_diagnostics = candidate_pools.diagnostics
        self.pool_search_complete = candidate_pools.search_complete
        self.em_cache.clear()
        self.d_cache.clear()

    def after_b_graph(self, actual_graph: dict[int, tuple[str, ...]]) -> None:
        """Build all lawful dynamic replacement leaves from public B results."""
        candidate_pools = target_pool(
            self.knowledge, self.cfg, self.L, self.expectations, self.references,
            self.ref_lengths, self.optimistic_b_budget,
            graph_override=actual_graph)
        self.pools = candidate_pools.priority
        self.qualified_pools = candidate_pools.qualified
        self.pool_diagnostics = candidate_pools.diagnostics
        self.pool_search_complete = candidate_pools.search_complete

    def em_means(self, candidates: tuple[str, ...], depth: int) -> dict[str, float]:
        key = (candidates, depth)
        if key not in self.em_cache:
            total = np.zeros(len(candidates))
            if depth not in self.prefix_frequencies:
                frequencies = Counter()
                for sequence, multiplicity in self.empirical_sequences.items():
                    frequencies[sequence[:depth + 1]] += multiplicity
                self.prefix_frequencies[depth] = frequencies
            for sequence, multiplicity in self.prefix_frequencies[depth].items():
                total += multiplicity * em_probs(sequence, candidates, self.cfg.epsilon)
            pop = len(split_users(list(range(self.n_c)), self.L)[depth])
            self.em_cache[key] = dict(zip(candidates, total * pop / self.empirical_size))
        return self.em_cache[key]

    def d_means(self, candidates: tuple[str, ...]) -> dict[tuple[str, int], float]:
        if candidates not in self.d_cache:
            self.d_cache[candidates] = projected_oue_means(
                self.knowledge, candidates, self.n_d, self.cfg)
        return self.d_cache[candidates]

    def future_cost(self, graph: dict[int, tuple[str, ...]],
                    targets: dict[int, Target], start_depth: int = 0,
                    survivors: list[str] | None = None
                    ) -> tuple[int, tuple[str, ...], DPlan] | None:
        """Simulate expected C selection from the currently *public* frontier."""
        if not targets or len(targets) > self.cfg.top_k:
            return None
        parents = [""] if survivors is None else survivors[:]
        used = 0
        for depth in range(start_depth, self.L):
            options = expand_paths(parents, graph, depth)
            if not options:
                return None
            protect = [target.full[:depth + 1] for target in targets.values()]
            mean = self.em_means(tuple(options), depth)
            addition = supports_to_top_k(tuple(options), protect, mean,
                                         self.n_c // self.L, 2 * self.cfg.top_k,
                                         self.cfg.score_z)
            if addition is None:
                return None
            used += sum(addition.values())
            scores = {node: mean[node] + addition.get(node, 0) for node in options}
            parents = top_nodes_in_creation_order(options, scores,
                                                  2 * self.cfg.top_k)
            if not set(protect) <= set(parents):
                return None
        final = tuple(parents)
        if len(final) < self.cfg.top_k:
            return None
        d_plan = choose_oue_plan(final, {c: t.full for c, t in targets.items()},
                                 self.d_means(final), self.cfg)
        if not d_plan.feasible:
            return None
        return used + max(d_plan.identities, self.d_base_reserve, self.d_locked), final, d_plan

    def joint_initial(self, remaining: int) -> JointPlan:
        """Uncapped, subset-aware B/C/D search; complete or explicitly interrupted.

        Each optimistic B edge lower bound is a NECESSARY condition only.
        The exact union of (position,pair) edges and all C/D projections are
        evaluated jointly before a candidate is admitted.
        """
        safe = ceil(self.cfg.safety_share * floor(self.cfg.beta * len(self.data.train)))
        available_classes = [c for c in sorted(self.references)
                             if self.qualified_pools.get(c)]
        checked = 0
        budget_pruned = 0
        bound_pruned = 0
        total_upper = 0
        best: JointPlan | None = None
        interrupted = False
        deadline = (time.monotonic() + self.cfg.search_timeout_seconds
                    if self.cfg.search_timeout_seconds > 0 else None)
        class_pools = {c: sorted(self.qualified_pools[c],
                                 key=lambda t: (-t.full_deviation, t.full))
                       for c in available_classes}

        def check(combo: tuple[Target, ...]) -> None:
            nonlocal checked, budget_pruned, best
            checked += 1
            if len({t.full for t in combo}) != len(combo):
                return
            required = {(pos, t.full[pos:pos + 2]) for t in combo
                        for pos in range(self.L - 1)}
            if (any(sum(pos == p for p, _ in required) > 2 * self.cfg.top_k
                    for pos in range(self.L - 1))
                    or sum(self.pair_lower[p][edge] for p, edge in required)
                    > self.optimistic_b_budget):
                budget_pruned += 1
                return
            additions = position_topups(self.expectations,
                                        tuple(t.full for t in combo), self.cfg, self.n_b)
            if additions is None:
                return
            b_cost = sum(sum(v.values()) for v in additions.values())
            if b_cost + self.d_base_reserve + safe > remaining:
                budget_pruned += 1
                return
            graph = apply_expected_pair_topups(self.expectations, additions, self.L,
                                               2 * self.cfg.top_k)
            if any(any(t.full[p:p + 2] not in graph[p]
                       for p in range(self.L - 1)) for t in combo):
                return
            targets = {t.cls: t for t in combo}
            future = self.future_cost(graph, targets)
            if future is None:
                return
            total = b_cost + future[0] + safe
            if total > remaining:
                budget_pruned += 1
                return
            full_score = sum(t.full_deviation for t in combo)
            d_predicted = future[2].identities
            candidate = JointPlan(
                targets=targets, b_votes=additions, predicted_cost=total,
                reason="Predicted feasible under model-conditional bounds",
                candidate_source="all_qualified_no_hard_cap",
                full_score=full_score, d_ratio=self.d_gamma,
                d_ratio_reserve=self.d_base_reserve,
                d_locked=max(self.d_base_reserve, d_predicted),
                d_predicted=d_predicted)
            tie = (total, -min(t.margin for t in combo),
                   -min(float(t.bottleneck_support_ratio or 0.0) for t in combo),
                   tuple(t.full for t in combo))
            best_tie = ((best.predicted_cost or float("inf")),
                        -min(t.margin for t in best.targets.values()),
                        -min(float(t.bottleneck_support_ratio or 0.0)
                             for t in best.targets.values()),
                        tuple(t.full for _, t in sorted(best.targets.items()))) if best else ()
            if (best is None or len(candidate.targets) > len(best.targets)
                    or (len(candidate.targets) == len(best.targets) and
                        (full_score > (best.full_score or 0.0) + 1e-8 or
                         (abs(full_score - (best.full_score or 0.0)) <= 1e-8
                          and tie < best_tie)))):
                best = candidate

        try:
            
            
            for size in range(len(available_classes), 0, -1):
                for subset in combinations(available_classes, size):
                    pools = [class_pools[c] for c in subset]
                    max_tail = [0.0] * (size + 1)
                    for i in range(size - 1, -1, -1):
                        max_tail[i] = max_tail[i + 1] + pools[i][0].full_deviation
                    total_upper += int(np.prod([len(p) for p in pools], dtype=object))

                    def walk(depth: int, chosen: list[Target], score: float,
                             used_edges: frozenset[tuple[int, str]], lb: int) -> None:
                        nonlocal bound_pruned, interrupted
                        if deadline is not None and time.monotonic() >= deadline:
                            interrupted = True
                            raise TimeoutError("BCD search interrupted by explicit time limit")
                        if (best is not None and len(best.targets) == size and
                                score + max_tail[depth] < (best.full_score or 0.0) - 1e-8):
                            bound_pruned += 1
                            return
                        if depth == size:
                            check(tuple(chosen))
                            return
                        for target in pools[depth]:
                            if any(t.full == target.full for t in chosen):
                                continue
                            edges = frozenset((j, target.full[j:j + 2])
                                              for j in range(self.L - 1))
                            union = used_edges | edges
                            added = union - used_edges
                            new_lb = lb + sum(self.pair_lower[j][e] for j, e in added)
                            if new_lb > self.optimistic_b_budget:
                                bound_pruned += 1
                                continue
                            if any(sum(p == j for p, _ in union) > 2 * self.cfg.top_k
                                   for j in range(self.L - 1)):
                                bound_pruned += 1
                                continue
                            chosen.append(target)
                            walk(depth + 1, chosen, score + target.full_deviation,
                                 union, new_lb)
                            chosen.pop()

                    walk(0, [], 0.0, frozenset(), 0)
                if best is not None and len(best.targets) == size:
                    break
        except TimeoutError:
            interrupted = True

        complete = self.pool_search_complete and not interrupted
        result = best if best is not None else JointPlan(
            d_ratio=self.d_gamma, d_ratio_reserve=self.d_base_reserve,
            d_locked=self.d_base_reserve,
            reason=("No model-feasible active target set" if complete
                    else "Search incomplete and no feasible target found"))
        result.checked_combinations = checked
        result.total_combinations = total_upper
        result.search_complete = complete
        result.search_limited = not complete
        result.search_diagnostics = {
            "search_complete": complete,
            "candidate_search_complete": self.pool_search_complete,
            "checked_combinations": checked, "upper_bound_combinations": total_upper,
            "optimistic_b_budget": self.optimistic_b_budget,
            "bound_pruned_branches": bound_pruned,
            "budget_pruned_combinations": budget_pruned,
            "qualified_pool_sizes": {str(c): len(self.qualified_pools.get(c, ()))
                                     for c in sorted(self.references)},
            "partial_class_subsets_allowed": True,
            "search_timeout_seconds": self.cfg.search_timeout_seconds,
            "no_fixed_candidate_or_combination_caps": True}
        return result

    def _reselect_exact_group(self, targets: dict[int, Target], missing: set[int],
                              depth: int, survivors: list[str],
                              graph: dict[int, tuple[str, ...]], remaining: int,
                              events: list[dict[str, Any]]) -> dict[int, Target]:
        """Attempt joint recovery of *all* supplied missing classes.

        At B's conclusion use depth=-1 and no previous frontier. At the end
        of C depth=ell, the public survivors are strings of length ell+1.
        """
        if not missing or not self.cfg.allow_dynamic_reselect:
            return targets
        frozen = {cls: target for cls, target in targets.items() if cls not in missing}
        missing_classes = sorted(missing)
        if any(not self.qualified_pools.get(cls) and not self.pools.get(cls)
               for cls in missing_classes):
            events.append({"event": "reselect_failure", "depth": depth,
                           "classes": sorted(missing), "reason": "empty original pool",
                           "search_limited": not self.pool_search_complete,
                           "search_complete": self.pool_search_complete})
            return frozen

        best: dict[int, Target] | None = None
        best_source = None
        best_phase = None
        global_key: tuple[float, ...] | None = None
        best_prefix_changes = 0
        checked_total = 0
        truncated = False
        any_survivor = False
        frozen_fulls = {t.full for t in frozen.values()}
        tried_combinations: set[tuple[str, ...]] = set()
        
        
        
        deadline = (time.monotonic() + self.cfg.search_timeout_seconds
                    if self.cfg.search_timeout_seconds > 0 else None)
        for phase in ("same_prefix", "replacement_prefix"):
            if truncated:
                break
            for source, pools in (("all_qualified", self.qualified_pools),):
                plausible: list[list[Target]] = []
                for cls in missing_classes:
                    old = targets.get(cls)
                    valid = [t for t in pools.get(cls, ())
                             if t.margin > 1e-12 and t.full_margin > 1e-12
                             and t.full not in frozen_fulls
                             and all(t.full[pos:pos + 2] in graph.get(pos, ())
                                     for pos in range(self.L - 1))
                             and (depth < 0 or t.full[:depth + 1] in survivors)]
                    if phase == "same_prefix":
                        valid = [t for t in valid if old is not None
                                 and t.prefix == old.prefix]
                    valid.sort(key=lambda t: (
                        -t.full_deviation, -t.margin,
                        -float(t.bottleneck_support_ratio or 0.0),
                        t.extension_edges, t.full))
                    plausible.append(valid)
                if any(not options for options in plausible):
                    continue
                any_survivor = True
                for replacements in product(*plausible):
                    if deadline is not None and time.monotonic() >= deadline:
                        truncated = True
                        break
                    signature = tuple(t.full for t in replacements)
                    if signature in tried_combinations:
                        continue
                    tried_combinations.add(signature)
                    checked_total += 1
                    proposed = dict(frozen)
                    proposed.update({t.cls: t for t in replacements})
                    if len({t.full for t in proposed.values()}) != len(proposed):
                        continue
                    forecast = self.future_cost(graph, proposed, depth + 1,
                                                survivors if depth >= 0 else None)
                    if forecast is None:
                        continue
                    safety = ceil(self.cfg.safety_share * floor(self.cfg.beta * len(self.data.train)))
                    if forecast[0] + safety > remaining:
                        continue
                    changed = sum(int(targets.get(t.cls) is None or
                                      targets[t.cls].prefix != t.prefix)
                                  for t in replacements)
                    
                    
                    
                    retained_prefixes = len(replacements) - changed
                    key = (float(retained_prefixes),
                           sum(t.full_deviation for t in proposed.values()),
                           -float(forecast[0]),
                           min(t.margin for t in proposed.values()),
                           min(float(t.bottleneck_support_ratio or 0.0)
                               for t in proposed.values()),
                           -sum(t.extension_edges for t in proposed.values()),
                           -float(changed))
                    if global_key is None or key > global_key:
                        best, global_key = proposed, key
                        best_source, best_phase = source, phase
                        best_prefix_changes = changed
            
            
            if best is not None and phase == "same_prefix":
                break
        if best is None:
            events.append({"event": "reselect_failure", "depth": depth,
                           "classes": sorted(missing),
                           "reason": ("no surviving legal descendants" if not any_survivor else
                                      "no feasible remaining joint C/D plan in searched pools"),
                           "checked_combinations": checked_total,
                           "search_limited": truncated or not self.pool_search_complete,
                           "search_complete": not truncated and self.pool_search_complete})
            return frozen
        events.append({"event": "dynamic_reselect", "depth": depth,
                       "classes": sorted(missing),
                       "source": best_source,
                       "selection_phase": best_phase,
                       "prefixes_changed": best_prefix_changes,
                       "prefixes_preserved": len(missing_classes) - best_prefix_changes,
                       "full_score": global_key[1],
                       "future_cost_with_d_reserve": -global_key[2],
                       "checked_combinations": checked_total,
                       "search_limited": truncated or not self.pool_search_complete,
                       "search_complete": not truncated and self.pool_search_complete,
                       "new_targets": {str(c): t.full for c, t in best.items()}})
        return best

    def dynamic_reselect(self, targets: dict[int, Target], missing: set[int],
                         depth: int, survivors: list[str],
                         graph: dict[int, tuple[str, ...]], remaining: int,
                         events: list[dict[str, Any]]) -> dict[int, Target]:
        """Recover the largest feasible subset when several classes are pruned.

        A class with no legal surviving replacement must not suppress other
        classes' recovery opportunities.  Once the number of recoverable
        classes is maximized, compare the joint full-leaf deviation; only
        after that compare remaining C/D cost.  Already submitted identities
        are never recovered.  Future stages see only the returned survivors.
        """
        if not missing:
            return targets
        frozen = {c: t for c, t in targets.items() if c not in missing}
        if not self.cfg.allow_dynamic_reselect:
            events.append({"event": "reselect_failure", "depth": depth,
                           "classes": sorted(missing),
                           "reason": "dynamic_reselection_disabled"})
            return frozen
        if len(missing) == 1:
            return self._reselect_exact_group(targets, missing, depth, survivors,
                                              graph, remaining, events)

        checked_subsets = 0
        chosen: dict[int, Target] | None = None
        chosen_events: list[dict[str, Any]] = []
        chosen_key: tuple[Any, ...] | None = None
        
        
        all_complete = self.pool_search_complete
        for size in range(len(missing), 0, -1):
            for subset in combinations(sorted(missing), size):
                checked_subsets += 1
                trial_targets = dict(frozen)
                trial_targets.update({c: targets[c] for c in subset if c in targets})
                trial_events: list[dict[str, Any]] = []
                trial = self._reselect_exact_group(
                    trial_targets, set(subset), depth, survivors, graph,
                    remaining, trial_events)
                all_complete &= all(e.get("search_complete", True) and
                                    not e.get("search_limited", False)
                                    for e in trial_events)
                if not set(subset) <= set(trial):
                    continue
                success_event = next((e for e in reversed(trial_events)
                                      if e["event"] == "dynamic_reselect"), None)
                if success_event is None:
                    continue
                preserved = len(subset) - success_event["prefixes_changed"]
                key = (preserved,
                       sum(t.full_deviation for t in trial.values()),
                       -success_event["future_cost_with_d_reserve"],
                       min(t.margin for t in trial.values()),
                       -sum(t.extension_edges for t in trial.values()),
                       tuple(t.full for _, t in sorted(trial.items())))
                if chosen_key is None or key > chosen_key:
                    chosen, chosen_events, chosen_key = trial, trial_events, key
            
            
            if chosen is not None:
                break
        if chosen is None:
            events.append({"event": "reselect_failure", "depth": depth,
                           "classes": sorted(missing),
                           "reason": "no_feasible_recoverable_subset",
                           "checked_class_subsets": checked_subsets,
                           "search_limited": not all_complete,
                           "search_complete": all_complete})
            return frozen
        recovered = set(chosen) - set(frozen)
        dropped = missing - recovered
        for event in chosen_events:
            if event["event"] == "dynamic_reselect":
                event["recovered_classes"] = sorted(recovered)
                event["checked_class_subsets"] = checked_subsets
                event["search_complete"] = bool(event.get("search_complete", True)) and all_complete
                event["search_limited"] = not event["search_complete"]
                event["maximum_recovered_class_count_proven"] = event["search_complete"]
        events.extend(chosen_events)
        if dropped:
            events.append({"event": "reselect_failure", "depth": depth,
                           "classes": sorted(dropped),
                           "reason": "partial_joint_recovery_only",
                           "recovered_classes": sorted(recovered),
                           "checked_class_subsets": checked_subsets,
                           "search_limited": not all_complete,
                           "search_complete": all_complete})
        return chosen

    def budget_reselect(self, targets: dict[int, Target], current_depth: int,
                        parents: list[str], graph: dict[int, tuple[str, ...]],
                        remaining: int, events: list[dict[str, Any]]
                        ) -> dict[int, Target]:
        """Before CURRENT C reports: keep originals if feasible, else replan.

        Lexicographic policy: retained active classes > original FULL targets
        > original PREFIXES > full-leaf J. Previously inactive classes are
        never activated; the D locked identities cannot be spent in C.
        """
        if not targets or not self.cfg.allow_dynamic_reselect:
            return targets
        safe = ceil(self.cfg.safety_share
                    * floor(self.cfg.beta * len(self.data.train)))
        classes = sorted(targets)
        best: dict[int, Target] | None = None
        best_key: tuple[Any, ...] | None = None
        best_forecast = 0
        checked_subsets = checked_combos = 0
        deadline = (time.monotonic() + self.cfg.search_timeout_seconds
                    if self.cfg.search_timeout_seconds > 0 else None)
        interrupted = False
        for size in range(len(classes), 0, -1):
            for subset in combinations(classes, size):
                checked_subsets += 1
                choices: list[list[Target]] = []
                for cls in subset:
                    old = targets[cls]
                    valid = [t for t in self.qualified_pools.get(cls, ())
                             if t.margin > 1e-12 and t.full_margin > 1e-12
                             and (current_depth == 0 or
                                  t.full[:current_depth] in parents)
                             and all(t.full[p:p + 2] in graph.get(p, ())
                                     for p in range(self.L - 1))]
                    valid.sort(key=lambda t: (
                        t.full != old.full, t.prefix != old.prefix,
                        -t.full_deviation, t.full))
                    choices.append(valid)
                if any(not pool for pool in choices):
                    continue
                for combo in product(*choices):
                    if deadline is not None and time.monotonic() >= deadline:
                        interrupted = True
                        break
                    checked_combos += 1
                    if len({t.full for t in combo}) != len(combo):
                        continue
                    proposed = {t.cls: t for t in combo}
                    forecast = self.future_cost(graph, proposed, current_depth, parents)
                    if forecast is None or forecast[0] + safe > remaining:
                        continue
                    original_count = sum(proposed[c].full == targets[c].full for c in proposed)
                    prefix_count = sum(proposed[c].prefix == targets[c].prefix
                                       for c in proposed)
                    key = (size, original_count, prefix_count,
                           sum(t.full_deviation for t in combo), -forecast[0],
                           tuple(t.full for t in combo))
                    if best_key is None or key > best_key:
                        best, best_key = proposed, key
                        best_forecast = forecast[0]
                if interrupted:
                    break
            if interrupted or best is not None:
                break
        if best is not None:
            events.append({"event": "c_budget_replan", "depth": current_depth,
                           "active_classes": sorted(best),
                           "disabled_classes": sorted(set(targets) - set(best)),
                           "checked_class_subsets": checked_subsets,
                           "checked_combinations": checked_combos,
                           "search_complete": not interrupted and self.pool_search_complete,
                           "preserved_original_full": best_key[1],
                           "preserved_original_prefix": best_key[2],
                           "complete_leaf_objective_J": best_key[3],
                           "planned_remaining_joint_cost": best_forecast,
                           "d_locked_minimum": self.d_locked})
            return best
        events.append({"event": "c_budget_replan_failure",
                       "depth": current_depth, "remaining": remaining,
                       "checked_class_subsets": checked_subsets,
                       "checked_combinations": checked_combos,
                       "search_complete": not interrupted and self.pool_search_complete})
        
        
        return {}


def honest_oue_counts(data: Dataset, ids: list[int], candidates: tuple[str, ...],
                      cfg: Config, rng: np.random.Generator) -> dict[tuple[str, int], int]:
    """Trace's OUE: one true (closest shape, class) cell; all other cells q."""
    q = 1 / (exp(cfg.epsilon) + 1)
    p = 0.5
    K = cfg.top_k
    counts = {(s, c): 0 for s in candidates for c in range(1, K + 1)}
    if not candidates:
        return counts
    for user in ids:
        sequence, label = data.train[user], data.labels[user]
        closest = min(candidates, key=lambda s: trace_distance(sequence[:len(s)], s))
        for s in candidates:
            for c in range(1, K + 1):
                counts[(s, c)] += int(rng.random() < (p if (s, c) == (closest, label) else q))
    return counts


def trace_label_greedy(counts: dict[tuple[str, int], int],
                       candidates: tuple[str, ...], K: int) -> list[str | None]:
    """Exactly one representative per class, matching Trace's nested greedy rule."""
    result: list[str | None] = [None] * K
    for _ in range(K):
        best = (0, "", None)
        for s in candidates:
            if s in result:
                continue
            for c in range(1, K + 1):
                if result[c - 1] is None and counts[(s, c)] > best[0]:
                    best = (counts[(s, c)], s, c)
        if best[2] is None:
            break
        result[best[2] - 1] = best[1]
    return result


def trace_report(data: Dataset, cfg: Config, run_seed: int, poisoning: bool,
                 fixed_length: int | None = None,
                 length_attack_plan: LengthPlan | None = None,
                 evaluate_test: bool = True,
                 stress_prune_classes: int = 0,
                 planning_knowledge: PlanningKnowledge | None = None) -> dict[str, Any]:
    """Execute actual A/B/C/D; `fixed_length` is diagnostic-only for a clean run.

    It reconstructs an offline honest B/C/D baseline conditional on the
    attacked run's public L, without pretending a length overwrite is an attack.
    """
    if fixed_length is not None and (poisoning or not 1 <= fixed_length <= 10):
        raise ValueError("fixed_length is allowed only for a clean counterfactual")
    if length_attack_plan is not None and not poisoning:
        raise ValueError("length_attack_plan is only allowed in a poisoning trial")
    if stress_prune_classes not in (0, 1, 2) or (stress_prune_classes and not poisoning):
        raise ValueError("Only poisoned validation runs may force 1/2-class pruning")
    data.validate()
    knowledge = (planning_knowledge if planning_knowledge is not None else
                 sample_planning_knowledge(data, cfg, run_seed + 707))
    if (length_attack_plan is not None and length_attack_plan.knowledge_fingerprint is not None
            and length_attack_plan.knowledge_fingerprint != knowledge.fingerprint):
        raise ValueError("The A length plan and B/C/D planner must share exactly "
                         "the same round-specific planning knowledge sample")
    N = len(data.train)
    M = floor(cfg.beta * N) if poisoning else 0
    
    groups = distribute_groups(N, cfg, run_seed + 1, length=1)
    domain = tuple(range(cfg.length_min, cfg.length_max + 1))
    honest = {name: len(ids) for name, ids in groups.items()}
    fake_used = {name: 0 for name in "ABCD"}
    events: list[dict[str, Any]] = []
    remaining = M
    planner = Planner(data, cfg, knowledge=knowledge)

    
    rng_a = np.random.default_rng(run_seed + 101)
    a_counts: Counter = Counter()
    if cfg.length_mechanism == "olh":
        for slot, user in enumerate(groups["A"]):
            value = min(cfg.length_max, max(cfg.length_min, len(data.train[user])))
            report = olh_report(value, slot, cfg.epsilon, domain, rng_a)
            a_counts.update(supported_lengths(report, domain, cfg.epsilon))
    else:
        for user in groups["A"]:
            value = min(cfg.length_max, max(cfg.length_min, len(data.train[user])))
            a_counts.update([randomized_response(value, domain, cfg.epsilon, rng_a)])

    a_plan = (length_attack_plan if length_attack_plan is not None else
              (length_plan(data, cfg, M, len(groups["A"]), knowledge=knowledge) if poisoning
               else LengthPlan()))
    if a_plan.reports:
        if cfg.length_mechanism == "olh":
            for report in a_plan.reports:
                a_counts.update(supported_lengths(report, domain, cfg.epsilon))
        else:
            a_counts.update(a_plan.reports)
    fake_used["A"] = len(a_plan.reports)
    if cfg.length_mechanism == "olh":
        grouped_a = Counter(a_plan.reports)
        a_report_details = [
            {"seed": h, "bucket": b, "times": times,
             "supports": list(supported_lengths((h, b), domain, cfg.epsilon))}
            for (h, b), times in sorted(grouped_a.items())]
    else:
        a_report_details = [{"length": str(v), "times": times}
                            for v, times in Counter(a_plan.reports).items()]
    remaining -= fake_used["A"]
    natural_L = select_length_output(a_counts, domain)
    L = fixed_length if fixed_length is not None else natural_L
    events.append({"event": "length_selection", "selected": L, "natural_length": natural_L,
                   "counterfactual_fixed_length": fixed_length is not None,
                   "attacker_target": a_plan.target,
                   "target_hit": a_plan.target == L if a_plan.target is not None else None,
                   "fake_used": fake_used["A"]})
    groups = distribute_groups(N, cfg, run_seed + 1, length=L)
    honest = {name: len(ids) for name, ids in groups.items()}

    
    
    planner.setup(L, len(groups["B"]), len(groups["C"]), len(groups["D"]),
                  fake_total=M if poisoning else None,
                  fake_a=fake_used["A"], honest_a=len(groups["A"]),
                  build_candidates=poisoning and 2 <= L <= 9)
    joint = planner.joint_initial(remaining) if poisoning and 2 <= L <= 9 else JointPlan(
        reason="No active B/C/D target or downstream length not viable",
        d_ratio=planner.d_gamma, d_ratio_reserve=planner.d_base_reserve,
        d_locked=planner.d_base_reserve)
    
    
    planner.d_locked = joint.d_locked
    events.append({"event": "d_reserve_locked", "gamma_d": planner.d_gamma,
                   "base_ratio_reserve": planner.d_base_reserve,
                   "d_joint_predicted": joint.d_predicted,
                   "d_locked": joint.d_locked,
                   "available_after_a": remaining,
                   "feasible": bool(joint.targets)})
    rng_b = np.random.default_rng(run_seed + 202)
    b_counts: dict[int, Counter] = {j: Counter() for j in range(max(0, L - 1))}
    for user in groups["B"]:
        s = data.train[user][:L]
        if len(s) < 2:
            continue
        pos = int(rng_b.integers(len(s) - 1))
        reported = randomized_response(s[pos:pos + 2], PAIRS, cfg.epsilon, rng_b)
        b_counts[pos][reported] += 1
    b_needed = sum(sum(v.values()) for v in joint.b_votes.values())
    if b_needed + planner.d_locked <= remaining:
        for pos, additions in joint.b_votes.items():
            b_counts[pos].update(additions)
        fake_used["B"] = b_needed
        remaining -= b_needed
    elif joint.targets:
        events.append({"event": "b_budget_failure", "needed": b_needed, "remaining": remaining})
        joint = JointPlan(reason="Insufficient identities for B reports",
                          d_ratio=planner.d_gamma, d_ratio_reserve=planner.d_base_reserve,
                          d_locked=planner.d_base_reserve)
        planner.d_locked = planner.d_base_reserve
    b_graph = {pos: tuple(top_names(dict(b_counts[pos]), 2 * cfg.top_k))
               for pos in range(max(0, L - 1))}
    events.append({"event": "b_pruning", "pairs": {str(k): list(v) for k, v in b_graph.items()},
                   "fake_used": fake_used["B"], "planned_cost": joint.predicted_cost})
    b_initial_survival = {str(cls): all(
        target.full[pos:pos + 2] in b_graph[pos] for pos in range(L - 1))
        for cls, target in joint.targets.items()}

    initial_targets = dict(joint.targets)
    targets = dict(joint.targets)
    if poisoning and 2 <= L <= 9:
        planner.after_b_graph(b_graph)
    if targets:
        missing = {cls for cls, t in targets.items()
                   if any(t.full[pos:pos + 2] not in b_graph[pos] for pos in range(L - 1))}
        if missing:
            targets = planner.dynamic_reselect(targets, missing, -1, [], b_graph,
                                               remaining, events)

    
    
    
    c_groups = split_users(groups["C"], L)
    rng_c = np.random.default_rng(run_seed + 303)
    parents: list[str] = [""]
    c_states: list[dict[str, Any]] = []
    stress_forced_count = 0
    stress_depth = max(0, L // 2 - 1)
    for depth in range(L):
        nodes = expand_paths(parents, b_graph, depth)
        if not nodes:
            events.append({"event": "c_empty", "depth": depth})
            parents = []
            break
        candidates = tuple(nodes)
        missing_before = {c for c, t in targets.items() if t.full[:depth + 1] not in nodes}
        if missing_before:
            targets = planner.dynamic_reselect(targets, missing_before, depth - 1,
                                               parents if depth > 0 else [], b_graph,
                                               remaining, events)
        protected = [t.full[:depth + 1] for t in targets.values()]
        projected = planner.em_means(candidates, depth)
        c_additions = {}
        reserved: int | None = None
        planning_failure = None
        future = None
        if targets:
            suggested = supports_to_top_k(candidates, protected, projected,
                                          len(c_groups[depth]), 2 * cfg.top_k,
                                          cfg.score_z)
            if suggested is not None:
                forecast_nodes = top_nodes_in_creation_order(
                    nodes, {s: projected[s] + suggested.get(s, 0) for s in nodes},
                    2 * cfg.top_k)
                future = planner.future_cost(b_graph, targets, depth + 1, forecast_nodes)
            reserved = future[0] if future is not None else None
            c_additions, planning_failure = guard_stage_votes(
                suggested, reserved, remaining, ceil(cfg.safety_share * M))
            if planning_failure is not None:
                
                
                alternative = planner.budget_reselect(
                    targets, depth, parents, b_graph, remaining, events)
                if alternative != targets:
                    targets = alternative
                    protected = [t.full[:depth + 1] for t in targets.values()]
                    suggested = supports_to_top_k(
                        candidates, protected, projected, len(c_groups[depth]),
                        2 * cfg.top_k, cfg.score_z)
                    future = None
                    if suggested is not None:
                        forecast_nodes = top_nodes_in_creation_order(
                            nodes, {s: projected[s] + suggested.get(s, 0) for s in nodes},
                            2 * cfg.top_k)
                        future = planner.future_cost(
                            b_graph, targets, depth + 1, forecast_nodes)
                    reserved = future[0] if future is not None else None
                    c_additions, planning_failure = guard_stage_votes(
                        suggested, reserved, remaining,
                        ceil(cfg.safety_share * M))
            if planning_failure is not None:
                events.append({"event": "c_plan_guard", "depth": depth,
                               "reason": planning_failure, "future_reserved": reserved,
                               "requested": sum(suggested.values()) if suggested else 0,
                               "remaining": remaining,
                               "protected": protected})
        count = sum(c_additions.values())

        honest_counts = Counter()
        for user in c_groups[depth]:
            probs = em_probs(data.train[user], candidates, cfg.epsilon)
            voted = candidates[int(rng_c.choice(len(candidates), p=probs))]
            honest_counts[voted] += 1
        honest_counts.update(c_additions)
        fake_used["C"] += count
        remaining -= count
        winners = top_nodes_in_creation_order(
            nodes, {s: honest_counts[s] for s in nodes}, 2 * cfg.top_k)
        if stress_prune_classes and depth == stress_depth and targets:
            
            
            
            original_winners = winners[:]
            target_prefixes = {cls: t.full[:depth + 1]
                               for cls, t in targets.items()}
            unique_nodes = [node for cls, node in sorted(target_prefixes.items())
                            if node in winners and list(target_prefixes.values()).count(node) == 1]
            removed = list(dict.fromkeys(unique_nodes))[:stress_prune_classes]
            stress_forced_count = len(removed)
            if removed:
                winners = [s for s in winners if s not in removed]
                for substitute in top_names({s: honest_counts[s] for s in nodes}, len(nodes)):
                    if len(winners) >= len(original_winners):
                        break
                    if substitute not in removed and substitute not in winners:
                        winners.append(substitute)
            events.append({"event": "stress_forced_prune", "depth": depth,
                           "requested_class_count": stress_prune_classes,
                           "forced_node_count": stress_forced_count,
                           "removed_nodes": removed,
                           "counterfactual": True})
            
            
            kept = set(winners)
            winners = [node for node in nodes if node in kept]
        c_states.append({"depth": depth, "candidates": len(nodes),
                         "survivors": winners, "fake_used": count,
                         "protected": protected,
                         "future_reserved": reserved,
                         "d_locked_minimum": planner.d_locked,
                         "d_predicted_identities": (
                             future[2].identities if future is not None else None),
                         "planning_failure": planning_failure,
                         "planned_votes": dict(c_additions)})
        parents = winners

        missing_after = {cls for cls, t in targets.items()
                         if t.full[:depth + 1] not in parents}
        if missing_after:
            targets = planner.dynamic_reselect(targets, missing_after, depth,
                                               parents, b_graph, remaining, events)

    
    
    leaves = tuple(parents) if c_states and c_states[-1]["depth"] == L - 1 else ()
    d_additions: list[set[tuple[str, int]]] = []
    d_plan = DPlan(feasible=False)
    d_available_after_c = remaining
    if poisoning and d_available_after_c < planner.d_locked:
        raise AssertionError("B/C spent identities locked as D's minimum reserve")
    d_free_after_c = d_available_after_c - planner.d_locked
    if poisoning:
        events.append({"event": "d_final_budget", "locked_minimum": planner.d_locked,
                       "unused_free_from_earlier_stages": d_free_after_c,
                       "total_available": d_available_after_c,
                       "honest_candidate_count": len(leaves),
                       "actual_oue_ones_cap": (
                           oue_ones_cap(cfg.top_k, len(leaves), cfg.epsilon)
                           if leaves else None)})
    if poisoning and len(leaves) >= cfg.top_k and targets:
        if all(t.full in leaves for t in targets.values()):
            d_plan = choose_oue_plan(leaves, {c: t.full for c, t in targets.items()},
                                     planner.d_means(leaves), cfg)
            if d_plan.feasible and d_plan.identities <= d_available_after_c:
                d_additions = pack_oue(d_plan, leaves, cfg.top_k)
                fake_used["D"] = len(d_additions)
                remaining -= fake_used["D"]
            else:
                events.append({"event": "d_budget_failure", "required": d_plan.identities,
                               "remaining": remaining, "feasible": d_plan.feasible})
        else:
            events.append({"event": "d_missing_target_leaf",
                           "remaining_targets": {str(c): t.full for c, t in targets.items()}})
    rng_d = np.random.default_rng(run_seed + 404)
    d_counts = honest_oue_counts(data, groups["D"], leaves, cfg, rng_d)
    for bits in d_additions:
        for cell in bits:
            d_counts[cell] += 1
    representatives = trace_label_greedy(d_counts, leaves, cfg.top_k) if leaves else [None] * cfg.top_k
    hit = {str(c): representatives[c - 1] == t.full for c, t in targets.items()}
    initial_hit = {str(c): representatives[c - 1] == t.full
                   for c, t in initial_targets.items()}
    initial_classes = set(initial_targets)
    final_active_classes = set(targets)
    deactivated_classes = sorted(initial_classes - final_active_classes)
    changed_classes = sorted(c for c in initial_classes & final_active_classes
                             if initial_targets[c].full != targets[c].full)
    
    initial_active_all_hit = (all(initial_hit.values()) if initial_hit else None)
    original_classes_success = initially_selected_classes_succeeded(
        initial_targets, targets, hit)
    final_active_all_hit = (all(hit.values()) if hit else None)
    quality, normalized_quality = {}, {}
    
    
    evaluation_references, _ = length_reference(data, cfg.reference_coverage)
    for cls in range(1, cfg.top_k + 1):
        shape = representatives[cls - 1]
        quality[str(cls)] = (ref_dist(shape, evaluation_references[cls])
                             if shape is not None else None)
        normalized_quality[str(cls)] = (ref_dist(shape, evaluation_references[cls], True)
                                         if shape is not None else None)
    return {"seed": run_seed, "attacked": poisoning, "epsilon": cfg.epsilon,
            "length": L, "natural_honest_length": natural_L,
            "counterfactual_fixed_length": fixed_length is not None,
            "length_plan": {"target": a_plan.target,
                                        "reference_length_lower_bound": a_plan.reference_min,
                                        "min_required": a_plan.min_required,
                                        "attempts": a_plan.attempts,
                                        "net_gains": a_plan.net_gains,
                                        "submitted_reports": a_report_details},
            "honest_groups": honest, "fake_total": M,
            "planning_knowledge": knowledge.diagnostics(),
            "shape_distance_reference_source": "full_honest_population_posthoc_only",
            "fake_used": fake_used, "fake_remaining": remaining,
            "references": {str(cls): [{"shape": s, "weight": w}
                                       for s, w in planner.references[cls]]
                           for cls in range(1, cfg.top_k + 1)},
            "reference_lengths": {str(k): v for k, v in planner.ref_lengths.items()},
            "reference_coverage_policy": cfg.reference_coverage,
            "candidate_pools": {str(cls): [asdict(t) for t in pool]
                                for cls, pool in planner.pools.items()},
            "qualified_candidate_pools": {str(cls): [asdict(t) for t in pool]
                                          for cls, pool in planner.qualified_pools.items()},
            "candidate_pool_diagnostics": {
                str(cls): value for cls, value in planner.pool_diagnostics.items()},
            "targets": {str(c): asdict(t) for c, t in targets.items()},
            "initial_targets": {str(c): asdict(t) for c, t in initial_targets.items()},
            "target_success": hit, "all_target_success": len(hit) == cfg.top_k and all(hit.values()),
            "initial_target_hit": initial_hit,
            "initial_active_target_count": len(initial_classes),
            "final_active_target_count": len(final_active_classes),
            "all_initial_active_targets_hit": initial_active_all_hit,
            "all_initial_active_classes_succeeded_after_reselection": original_classes_success,
            "all_final_active_targets_hit": final_active_all_hit,
            "initial_active_targets_hit_count": sum(initial_hit.values()),
            "final_active_targets_hit_count": sum(hit.values()),
            "initial_target_deactivated_classes": deactivated_classes,
            "initial_target_deactivation_rate": (
                len(deactivated_classes) / len(initial_classes) if initial_classes else None),
            "initial_target_changed_classes": changed_classes,
            "active_target_hit_rate": (sum(hit.values()) / len(hit) if hit else None),
            "target_status_by_class": {
                str(c): ("not_initially_selected" if c not in initial_targets
                         else "current_target_hit" if c in targets and hit[str(c)]
                         else "target_deactivated" if c not in targets
                         else "current_target_missed")
                for c in range(1, cfg.top_k + 1)},
            "representatives": representatives,
            "accuracy": accuracy(data, representatives) if evaluate_test else None,
            "recall_by_class": recall_by_class(data, representatives) if evaluate_test else None,
            "test_evaluation_performed": evaluate_test,
            "stress_counterfactual": stress_prune_classes > 0,
            "stress_requested_classes": stress_prune_classes,
            "stress_forced_node_count": stress_forced_count,
            "b_initial_survival": b_initial_survival,
            "c_final_survival": {str(cls): target.full in leaves
                                 for cls, target in targets.items()},
            "valid_representative_count": sum(s is not None for s in representatives),
            "no_output": all(s is None for s in representatives),
            "shape_distance": quality, "normalized_shape_distance": normalized_quality,
            "b_planned_votes": {str(j): dict(v) for j, v in joint.b_votes.items()},
            "b_kept": {str(j): list(v) for j, v in b_graph.items()},
            "c_levels": c_states,
            "d_reserved_minimum": planner.d_locked,
            "d_gamma": planner.d_gamma,
            "d_ratio_reserved": planner.d_base_reserve,
            "d_available_after_c": d_available_after_c,
            "d_free_after_c": d_free_after_c,
            "d_borrowed_from_free_pool": max(0, fake_used["D"] - planner.d_locked),
            "d_unused_after_final": remaining,
            
            "d_ones_per_vector": (oue_ones_cap(cfg.top_k, len(leaves), cfg.epsilon)
                                  if leaves else 0),
            "d_ones_cap_policy": "max(floor(E[Z]), 1)",
            "d_expected_honest_ones": (0.5 + (cfg.top_k * len(leaves) - 1)
                                       / (exp(cfg.epsilon) + 1)) if leaves else None,
            "d_actual_ones_per_vector": [len(bits) for bits in d_additions],
            "d_planned_increments": {f"{s}:{c}": n
                                     for (s, c), n in d_plan.increments.items()},
            "d_class_order": d_plan.selected_order,
            "d_malicious_vectors": [[f"{s}:{c}" for s, c in sorted(bits)]
                                    for bits in d_additions],
            "events": events, "initial_joint_plan": {
                "reason": joint.reason, "predicted_cost": joint.predicted_cost,
                "complete_leaf_objective_J": joint.full_score,
                "objective": "maximize_sum_of_complete_leaf_weighted_edit_distances",
                "gamma_d": joint.d_ratio, "d_ratio_reserve": joint.d_ratio_reserve,
                "d_locked_minimum": joint.d_locked,
                "d_predicted_identity_cost": joint.d_predicted,
                "checked_combinations": joint.checked_combinations,
                "total_combinations": joint.total_combinations,
                "search_complete": joint.search_complete,
                "search_limited": joint.search_limited,
                "search_diagnostics": joint.search_diagnostics,
                "candidate_source": joint.candidate_source}}


CHECKPOINT_FORMAT_VERSION = 1


def _dataset_fingerprint(data: Dataset) -> str:
    """Bind saved trials to the precise training/test data and their order."""
    digest = sha256()
    for title, sequences, labels in (("train", data.train, data.labels),
                                     ("test", data.test, data.test_labels)):
        digest.update(f"{title}:{len(sequences)}\n".encode("ascii"))
        for shape, label in zip(sequences, labels):
            digest.update(f"{label}:{len(shape)}:{shape}\n".encode("ascii"))
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Never expose a partially written trial or aggregate JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                   dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_trial_checkpoints(checkpoint_dir: Path, data: Dataset,
                               cfg: Config, seed: int, resume: bool) -> str:
    """Refuse to mix different datasets, parameters, or RNG bases."""
    signature_source = {"format_version": CHECKPOINT_FORMAT_VERSION,
                        "dataset_fingerprint": _dataset_fingerprint(data),
                        "config": asdict(cfg), "base_seed": seed}
    signature = sha256(json.dumps(signature_source, sort_keys=True,
                                  ensure_ascii=False).encode("utf8")).hexdigest()
    manifest = checkpoint_dir / "manifest.json"
    if manifest.exists():
        if not resume:
            raise ValueError(f"Checkpoints already exist at {checkpoint_dir}; "
                             "use --resume or select a new --output")
        existing = json.loads(manifest.read_text(encoding="utf8"))
        if existing.get("signature") != signature:
            raise ValueError("Checkpoint configuration, base seed, or dataset "
                             "differs from this run; use a new --output")
    else:
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} is not empty "
                             "and has no valid manifest; choose a new --output")
        _write_json_atomic(manifest, {"signature": signature, **signature_source})
    return signature


def paired_experiment(data: Dataset, cfg: Config, runs: int, seed: int,
                      trial_start: int = 0, checkpoint_dir: Path | None = None,
                      resume: bool = False) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs must be positive")
    if trial_start < 0:
        raise ValueError("trial_start must be nonnegative")
    if resume and checkpoint_dir is None:
        raise ValueError("resume requires a checkpoint directory")
    signature = (_prepare_trial_checkpoints(checkpoint_dir, data, cfg, seed, resume)
                 if checkpoint_dir is not None else None)
    
    
    
    records = []
    selections = []
    resumed_trials = 0
    for index in range(runs):
        
        
        trial_index = trial_start + index
        trial_seed = seed + trial_index * 10007
        saved_trial = (checkpoint_dir / f"trial_{trial_index:08d}.json"
                       if checkpoint_dir is not None else None)
        if saved_trial is not None and saved_trial.exists():
            if not resume:
                raise ValueError(f"Checkpoint already exists at {saved_trial}; "
                                 "use --resume or select a new --output")
            stored = json.loads(saved_trial.read_text(encoding="utf8"))
            if (stored.get("signature") != signature or
                    stored.get("trial_index") != trial_index or
                    stored.get("trial_seed") != trial_seed):
                raise ValueError(f"Incompatible checkpoint: {saved_trial}")
            record = stored["record"]
            if (record.get("trial_index") != trial_index or
                    record.get("trial_seed") != trial_seed):
                raise ValueError(f"Corrupted trial checkpoint: {saved_trial}")
            records.append(record)
            selections.append(record["a_analytical_selection"])
            resumed_trials += 1
            print(f"trial={index + 1}/{runs} global={trial_index} "
                  "restored_from_checkpoint", flush=True)
            continue
        
        
        
        knowledge = sample_planning_knowledge(data, cfg, trial_seed + 707)
        
        
        
        
        
        
        
        trial_cfg = replace(cfg, seed_strategy_seed=(
            cfg.seed_strategy_seed + trial_index * 10009))
        length_groups = distribute_groups(len(data.train), trial_cfg,
                                          trial_seed + 1, length=1)
        candidate_plan = length_plan(data, trial_cfg, floor(cfg.beta * len(data.train)),
                                     len(length_groups["A"]), knowledge=knowledge)
        a_selection = {
            "status": ("analytically_feasible" if candidate_plan.target is not None
                       else "no_analytically_feasible_target"),
            "selected_length": candidate_plan.target,
            "reference_length_lower_bound": candidate_plan.reference_min,
            "malicious_identities": len(candidate_plan.reports),
            "required_net_support": candidate_plan.bounds,
            "achieved_net_support": candidate_plan.net_gains,
            "search_attempts": candidate_plan.attempts,
            "knowledge_fingerprint": knowledge.fingerprint,
            "seed_pool_random_seed": trial_cfg.seed_strategy_seed,
            "seed_pool_policy": "vary_by_global_trial_reuse_across_configurations",
            "pre_execution_random_validation": False,
            "caution": "Analytic feasibility is model-dependent, not an observed success guarantee",
        }
        selections.append(a_selection)
        clean = trace_report(data, trial_cfg, trial_seed, False,
                             planning_knowledge=knowledge)
        attack = trace_report(data, trial_cfg, trial_seed, True,
                              length_attack_plan=candidate_plan,
                              planning_knowledge=knowledge)
        same_length_clean = trace_report(data, trial_cfg, trial_seed, False,
                                         fixed_length=attack["length"],
                                         planning_knowledge=knowledge)
        record = {"trial_index": trial_index, "trial_seed": trial_seed,
                  "clean": clean, "same_length_clean": same_length_clean,
                  "attack": attack, "a_analytical_selection": a_selection,
                  "post_length_accuracy_change": (attack["accuracy"]
                                                  - same_length_clean["accuracy"]),
                  "accuracy_change": attack["accuracy"] - clean["accuracy"]}
        
        
        if saved_trial is not None:
            _write_json_atomic(saved_trial, {"signature": signature,
                                             "trial_index": trial_index,
                                             "trial_seed": trial_seed,
                                             "record": record})
        records.append(record)
        print(f"trial={index + 1}/{runs} clean={clean['accuracy']:.4f} "
              f"attack={attack['accuracy']:.4f} "
              f"A={attack['length_plan']['target']}->{attack['length']} "
              f"malicious={sum(attack['fake_used'].values())}/{attack['fake_total']}",
              flush=True)
    diffs = [record["accuracy_change"] for record in records]
    def attack_mean_present(key: str) -> float | None:
        values = [x["attack"].get(key) for x in records]
        measured = [float(value) for value in values if value is not None]
        return float(np.mean(measured)) if measured else None

    summary = {"runs": runs, "mean_clean_accuracy": float(np.mean([x["clean"]["accuracy"] for x in records])),
               "mean_attack_accuracy": float(np.mean([x["attack"]["accuracy"] for x in records])),
               "mean_same_length_clean_accuracy": float(np.mean([
                   x["same_length_clean"]["accuracy"] for x in records])),
               "mean_accuracy_change": float(np.mean(diffs)),
               "mean_post_length_accuracy_change": float(np.mean([
                   x["post_length_accuracy_change"] for x in records])),
               "length_target_hit_rate": float(np.mean([
                   x["attack"]["length_plan"]["target"] is not None and
                   x["attack"]["length_plan"]["target"] == x["attack"]["length"]
                   for x in records])),
               "all_shape_targets_hit_rate": float(np.mean([
                   x["attack"]["all_target_success"] for x in records])),
               
               
               "all_three_classes_hit_rate": float(np.mean([
                   x["attack"]["all_target_success"] for x in records])),
               "trials_with_initial_active_targets": sum(
                   x["attack"].get("initial_active_target_count", 0) > 0
                   for x in records),
               "trials_with_final_active_targets": sum(
                   x["attack"].get("final_active_target_count", 0) > 0
                   for x in records),
               "all_initial_active_targets_hit_rate": attack_mean_present(
                   "all_initial_active_targets_hit"),
               "all_initial_active_classes_success_rate_after_reselection": attack_mean_present(
                   "all_initial_active_classes_succeeded_after_reselection"),
               "all_final_active_targets_hit_rate": attack_mean_present(
                   "all_final_active_targets_hit"),
               "mean_active_target_hit_rate": attack_mean_present("active_target_hit_rate"),
               "mean_initial_target_deactivation_rate": attack_mean_present(
                   "initial_target_deactivation_rate"),
               "mean_initial_active_target_count": attack_mean_present(
                   "initial_active_target_count"),
               "mean_final_active_target_count": attack_mean_present(
                   "final_active_target_count"),
               "attack_no_output_rate": float(np.mean([
                   x["attack"]["no_output"] for x in records]))}
    return {"config": asdict(cfg), "summary": summary,
            "trial_start": trial_start,
            "checkpointing": {"enabled": checkpoint_dir is not None,
                              "directory": (str(checkpoint_dir)
                                            if checkpoint_dir is not None else None),
                              "resumed_trials": resumed_trials},
            "a_analytical_selection": (selections[0] if runs == 1 else
                                       {"status": "per_trial", "per_trial": selections}),
            "a_analytical_selections": selections,
            "independent_bcd_validation": {"status": "removed_by_design",
                                           "used_for_selection": False},
            "trials": records}


def main() -> None:
    p = argparse.ArgumentParser(description="PrivShape/Trace four-stage local poisoning experiment")
    p.add_argument("--data-dir", type=Path, help="Directory of data.txt, label.txt, test_data.txt, test_label.txt")
    p.add_argument("--demo", action="store_true", help="Run on a synthetic smoke-test dataset")
    p.add_argument("--demo-users", type=int, default=1200)
    p.add_argument("--epsilon", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=.20)
    p.add_argument("--length-mechanism", choices=("olh", "grr"), default="olh")
    p.add_argument("--length-bound", choices=("normal", "cantelli"), default="normal",
                   help="A-stage model-conditional pairwise bound; Cantelli is "
                        "conservative only under the assumed honest-user "
                        "distribution and variance model, not a guarantee "
                        "of real-population attack success")
    p.add_argument("--length-delta", type=float, default=.05)
    p.add_argument("--seed-search", type=int, default=100,
                   help="Number of distinct random OLH seed candidates PER A identity")
    p.add_argument("--seed-space", type=int, default=2**31,
                   help="Seed values are sampled independently from [0, seed-space)")
    p.add_argument("--reference-coverage", type=float, default=.70,
                   help="Within each class, cover at least this share of honest shapes")
    p.add_argument("--score-z", type=float, default=2.0,
                   help="B/C/D marginal planning margin; not a joint confidence level")
    p.add_argument("--planning-sample", type=int, default=0,
                   help="0 (default) uses the entire normal training distribution; "
                        "positive N is an explicit approximate planning sample")
    p.add_argument("--search-timeout-seconds", type=float, default=0.0,
                   help="Optional runtime interruption (0: exhaustive); interrupted "
                        "searches are flagged incomplete, never called optimal")
    p.add_argument("--safety-share", type=float, default=.02,
                   help="Optional extra fake-identity buffer; use 0 for no buffer")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=20231025)
    p.add_argument("--seed-strategy-seed", type=int, default=20231025,
                   help="Base RNG seed for A candidate pools: each global trial "
                        "gets a new reproducible pool, shared across compared "
                        "beta/epsilon/planning-sample settings")
    p.add_argument("--trial-start", type=int, default=0,
                   help="First global trial index for parallel workers: assign "
                        "disjoint ranges and distinct --output files")
    p.add_argument("--resume", action="store_true",
                   help="Reuse compatible per-trial checkpoints; unfinished global "
                        "trial indices are recomputed")
    p.add_argument("--no-dynamic-reselect", action="store_true")
    p.add_argument("--output", type=Path, default=Path("trace_attack_result.json"))
    args = p.parse_args()
    if not args.demo and args.data_dir is None:
        p.error("Provide --data-dir with your preprocessed Trace files, or --demo")
    if (args.epsilon <= 0 or not 0 <= args.beta <= 1 or args.seed_search < 1
            or args.seed_space < args.seed_search
            or not 0 < args.reference_coverage <= 1
            or args.search_timeout_seconds < 0 or not 0 <= args.safety_share <= 1
            or not 0 < args.length_delta < 1 or args.score_z < 0
            or args.planning_sample < 0 or args.trial_start < 0
            or args.seed < 0 or args.seed_strategy_seed < 0):
        p.error("Invalid epsilon/beta/delta/seed-space/coverage/timeout/buffer")
    if args.output.exists() and not args.resume:
        p.error(f"Output already exists: {args.output}; select another --output "
                "or use --resume")
    checkpoint_dir = Path(str(args.output) + ".trials")
    if args.resume and args.output.exists() and not (checkpoint_dir / "manifest.json").exists():
        p.error("Cannot resume a legacy result without trial checkpoints; "
                "select a new --output")
    data = demo_dataset(args.demo_users, seed=args.seed) if args.demo else load_dataset(args.data_dir)
    cfg = Config(epsilon=args.epsilon, beta=args.beta,
                 length_mechanism=args.length_mechanism, seed_search=args.seed_search,
                 seed_space=args.seed_space,
                 seed_strategy_seed=args.seed_strategy_seed,
                 reference_coverage=args.reference_coverage,
                 probability_bound=args.length_bound,
                 length_success_delta=args.length_delta,
                 score_z=args.score_z, planning_sample=args.planning_sample,
                 search_timeout_seconds=args.search_timeout_seconds,
                 safety_share=args.safety_share,
                 allow_dynamic_reselect=not args.no_dynamic_reselect)
    result = paired_experiment(data, cfg, args.runs, args.seed,
                               trial_start=args.trial_start,
                               checkpoint_dir=checkpoint_dir, resume=args.resume)
    _write_json_atomic(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Detailed log: {args.output}")


if __name__ == "__main__":
    main()
