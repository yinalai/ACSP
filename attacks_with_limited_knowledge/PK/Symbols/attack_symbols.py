

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field, replace
from copy import deepcopy
try:
    from cdifflib import CSequenceMatcher
except ImportError:
    from difflib import SequenceMatcher as CSequenceMatcher
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
import symbols_distance as sd


SYMBOLS = "abcdefghij"
SYMBOL_SIZE = 6
ABC = SYMBOLS[:SYMBOL_SIZE]


PAIRS = tuple(a + b for a in ABC for b in ABC)
SYMBOL_DIST = dict(sd.get_distance(SYMBOL_SIZE))
SYMBOLS_PROTOCOL_VERSION = "symbols_c_baseline_separate_150s_replanning_20261102"
MASK32 = 0xFFFFFFFF
P1, P2, P3, P4, P5 = 0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1


def configure_symbol_size(size: int) -> None:
   
    global SYMBOL_SIZE, ABC, PAIRS, SYMBOL_DIST
    if not 3 <= size <= 10:
        raise ValueError("Symbols' supplied distance function supports alphabet sizes 3..10")
    if size == SYMBOL_SIZE:
        return
    SYMBOL_SIZE = size
    ABC = SYMBOLS[:size]
    PAIRS = tuple(a + b for a in ABC for b in ABC)
    SYMBOL_DIST = dict(sd.get_distance(size))
    full_edit.cache_clear()
    trace_distance.cache_clear()


def rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & MASK32


def xxh32(payload: bytes, seed: int = 0) -> int:
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
    attack_mode: str = "fk"  
    alpha: float = 0.02  
    
    
    tree_level_fraction: float = 0.70
    length_mechanism: str = "grr"
    length_min: int = 1
    length_max: int = 14
    length_success_delta: float = 0.05
    probability_bound: str = "normal"  
    seed_search: int = 100  
    seed_space: int = 2**31
    seed_strategy_seed: int = 20231025
    reference_coverage: float = 0.70
    
    length_min_multiplier: float = 1.0
    length_max_multiplier: float = 2.0
    top_k: int = 6
    symbol_size: int = 6
    branch_factor: int = 3  
    
    
    
    
    search_timeout_seconds: float | None = None
    
    
    
    
    planning_sample: int = 0
    
    
    planning_fraction: float | None = None
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
        if set(self.labels) != set(range(1, 7)) or not set(self.test_labels) <= set(range(1, 7)):
            raise ValueError("Symbols classification expects labels 1 through 6")
        for s in self.train + self.test:
            if not s or any(c not in ABC for c in s):
                raise ValueError(f"Invalid compressed SAX sequence: {s!r}")
            if any(a == b for a, b in zip(s, s[1:])):
                raise ValueError(f"Adjacent repeats remain in compressed SAX: {s!r}")


@dataclass(frozen=True)
class PlanningKnowledge:
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
            "per_class_counts": {str(cls): self.labels.count(cls) for cls in range(1, 7)},
            "shared_by": ["A_length", "70pct_class_references", "B_position_pairs",
                          "C_prefix_expected_votes", "D_label_expected_votes"],
        }


def planning_knowledge_size(cfg: Config, population: int) -> int:
    if cfg.planning_fraction is not None:
        if not 0 < cfg.planning_fraction <= 1 or cfg.planning_sample != 0:
            raise ValueError("planning_fraction must be in (0,1] and cannot "
                             "be combined with a positive planning_sample")
        return max(1, floor(population * cfg.planning_fraction + 1e-9))
    if cfg.planning_sample < 0:
        raise ValueError("planning_sample must be nonnegative")
    return population if cfg.planning_sample == 0 else min(population, cfg.planning_sample)








SEARCH_TIME_PROFILES: dict[str, dict[str, float]] = {
    "fixed_100pct_budget": {
        "initial_candidates": 300, "joint_initial": 45,
        "after_b_candidates": 180, "b_reselect": 60,
        "c_dynamic": 60, "c_budget": 90, "c_layer": 150,
        "d_reselect": 60,
    },
}


def search_time_profile(cfg: Config, knowledge: PlanningKnowledge
                        ) -> tuple[str, dict[str, float | None]]:
    name = "fixed_100pct_budget"
    if cfg.search_timeout_seconds is None:
        return name, dict(SEARCH_TIME_PROFILES[name])
    if cfg.search_timeout_seconds < 0:
        raise ValueError("search_timeout_seconds must not be negative")
    if cfg.search_timeout_seconds == 0:
        return "unlimited_override", {key: None for key in SEARCH_TIME_PROFILES[name]}
    seconds = float(cfg.search_timeout_seconds)
    return "uniform_override", {
        key: (2 * seconds if key == "c_layer" else seconds)
        for key in SEARCH_TIME_PROFILES[name]}


def sample_planning_knowledge(data: Dataset, cfg: Config, seed: int
                              ) -> PlanningKnowledge:
    total = len(data.train)
    size = planning_knowledge_size(cfg, total)
    if size >= total:
        indices = tuple(range(total))
        sample_seed: int | None = None
    else:
        indices = tuple(int(i) for i in
                        np.random.default_rng(seed).permutation(total)[:size])
        sample_seed = seed
    known = PlanningKnowledge(tuple(data.train[i] for i in indices),
                              tuple(data.labels[i] for i in indices), indices,
                              total, sample_seed)
    if set(known.labels) != set(range(1, 7)):
        raise ValueError("The randomly selected planning knowledge sample does not "
                         "contain all six Symbols classes; increase --planning-sample "
                         "or change --seed. No unobserved users are added to the sample.")
    return known


def sample_pk_knowledge(data: Dataset, cfg: Config, seed: int) -> PlanningKnowledge:
    total = len(data.train)
    count = floor(cfg.beta * total)
    if not 0 < count < total:
        raise ValueError("PK requires at least one controlled and one uncontrolled real user")
    ids = tuple(int(x) for x in np.random.default_rng(seed).permutation(total)[:count])
    known = PlanningKnowledge(tuple(data.train[i] for i in ids),
                              tuple(data.labels[i] for i in ids), ids, total, seed)
    return known


def load_dataset(directory: Path) -> Dataset:
    def read_text(name: str) -> list[str]:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; provide four preprocessed Symbols .txt files")
        return [x.strip() for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
    data = Dataset(read_text("data.txt"), [int(x) for x in read_text("label.txt")],
                   read_text("test_data.txt"), [int(x) for x in read_text("test_label.txt")])
    data.validate()
    return data


def demo_dataset(n: int = 1200, n_test: int = 180, seed: int = 29) -> Dataset:

    rng = np.random.default_rng(seed)
    centers = tuple("".join(ABC[(i + shift) % len(ABC)] for i in range(5))
                    for shift in range(6))

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

    labels = [1 + (i % 6) for i in range(n)]
    test_labels = [1 + (i % 6) for i in range(n_test)]
    ds = Dataset([draw(c) for c in labels], labels,
                 [draw(c) for c in test_labels], test_labels)
    ds.validate()
    return ds


@lru_cache(maxsize=500000)
def full_edit(s: str, t: str) -> float:
    return float(sd.similar_match_dtw(SYMBOL_DIST, s, t))


@lru_cache(maxsize=100000)
def trace_distance(s: str, t: str) -> float:
    return full_edit(s, t)


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Attack-planning search time budget exhausted")


def em_probs(sequence: str, candidates: tuple[str, ...], epsilon: float,
             search_deadline: float | None = None) -> np.ndarray:
    if not candidates:
        raise ValueError("Cannot match into an empty trie level")
    scores = []
    for c in candidates:
        _check_deadline(search_deadline)
        scores.append(1 / (trace_distance(sequence[:len(c)], c) + 0.1))
    scores = np.array(scores, dtype=float)
    spread = float(scores.max() - scores.min())
    if spread == 0:
        return np.full(len(candidates), 1 / len(candidates))
    weights = np.exp(epsilon * (scores - scores.min()) / spread / 2)
    return weights / weights.sum()


def predict_label(representatives: list[str | None], sequence: str) -> int:
    if not representatives or any(not isinstance(s, str) or not s
                                  for s in representatives):
        raise ValueError("Need nonempty class representatives (Symbols match_clustering)")
    distances = [trace_distance(sequence, s) for s in representatives]
    smallest = min(distances)
    
    
    best_ratio, selected = 0.0, 1
    for i, s in enumerate(representatives):
        if distances[i] == smallest:
            ratio = CSequenceMatcher(None, sequence, s).ratio()
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
            for cls in range(1, 7)}


def top_names(values: dict[str, float], limit: int) -> list[str]:
    
    
    return [k for k, _ in sorted(values.items(), key=lambda kv: -kv[1])[:limit]]


def top_nodes_in_creation_order(nodes: Iterable[str], scores: dict[str, float],
                                limit: int) -> list[str]:
   
    original = list(dict.fromkeys(nodes))
    kept = set(top_names({node: scores[node] for node in original}, limit))
    return [node for node in original if node in kept]


def honest_length_moments(pi: dict[int, float], n_a: int, epsilon: float,
                          domain: tuple[int, ...], mechanism: str,
                          target: int, rival: int, population_n: int) -> tuple[float, float]:
    
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


def pk_length_estimator(pi: dict[int, float], cfg: Config,
                        domain: tuple[int, ...],
                        slots: tuple[tuple[int, int, int | None], ...],
                        replaced: int,
                        unknown_population_size: int | None = None):
   
    controlled = [uid for _, uid, value in slots if value is not None]
    if not 0 <= replaced <= len(controlled):
        raise ValueError("Invalid number of replaced A identities")
    removed = set(controlled[:replaced])
    kept = [(slot, ({value: 1.0} if value is not None else pi))
            for slot, uid, value in slots if uid not in removed]
    g = len(domain) if cfg.length_mechanism == "grr" else floor(exp(cfg.epsilon)) + 1
    p, q = grr_prob(cfg.epsilon, g)

    def bin_of(slot: int, value: int) -> int:
        return value if cfg.length_mechanism == "grr" else bucket(
            slot, value, domain[0], g)

    def supports(query: int) -> float:
        return sum(q + (p - q) * sum(
            weight for value, weight in histogram.items()
            if bin_of(slot, value) == bin_of(slot, query))
            for slot, histogram in kept)

    def moments(target: int, rival: int) -> tuple[float, float]:
        mean, variance = 0.0, 0.0
        unknown_rows = []
        for slot, histogram in kept:
            target_bin, rival_bin = bin_of(slot, target), bin_of(slot, rival)
            if target_bin == rival_bin:
                continue
            contribution = (p - q) * sum(
                weight * ((bin_of(slot, value) == rival_bin)
                          - (bin_of(slot, value) == target_bin))
                for value, weight in histogram.items())
            second = 2 * q + (p - q) * sum(
                weight for value, weight in histogram.items()
                if bin_of(slot, value) in (target_bin, rival_bin))
            mean += contribution
            variance += max(0.0, second - contribution * contribution)
            if histogram is pi:
                unknown_rows.append([(p - q) * (
                    (bin_of(slot, v) == rival_bin) -
                    (bin_of(slot, v) == target_bin)) for v in domain])
        if (len(unknown_rows) > 1 and unknown_population_size is not None
                and unknown_population_size > 1):
            a = np.asarray(unknown_rows, dtype=float)
            weights = np.asarray([pi.get(v, 0.0) for v in domain])
            avgs = a @ weights
            unknown_mean = float(avgs.sum())
            sums = a.sum(axis=0)
            pair_a = float(np.dot(weights, sums ** 2 - (a ** 2).sum(axis=0)))
            pair_avg = float(unknown_mean ** 2 - (avgs ** 2).sum())
            variance -= (pair_a - pair_avg) / (unknown_population_size - 1)
        return mean, sqrt(max(0.0, variance))

    return {length: supports(length) for length in domain}, moments


def length_reference(data: Dataset | PlanningKnowledge, coverage: float = .70,
                     allow_missing: bool = False
                     ) -> tuple[dict[int, tuple[tuple[str, float], ...]], dict[int, int]]:
    if not 0 < coverage <= 1:
        raise ValueError("Reference coverage must be in (0, 1]")
    refs: dict[int, tuple[tuple[str, float], ...]] = {}
    highest: dict[int, int] = {}
    for cls in range(1, 7):
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
            if allow_missing:
                continue
            raise ValueError(f"Class {cls} has no reference shapes")
        refs[cls] = tuple((s, n / total) for s, n in chosen)
        highest[cls] = ceil(sum(n * len(s) for s, n in chosen) / total)
    return refs, highest


def select_length_output(votes: Counter, domain: tuple[int, ...]) -> int:
    
    
    
    ranked = sorted(votes, key=lambda v: -votes[v])
    if not ranked:
        return domain[0]
    return ranked[1] if ranked[0] == domain[-1] and len(ranked) > 1 else ranked[0]


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
    feasible: bool = False
    strategy: str = "none"


def length_net_gains(reports: Iterable[Any], target: int,
                     domain: tuple[int, ...], mechanism: str,
                     epsilon: float) -> dict[int, int]:
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
    collisions = set(supported)
    return sum(max(0, due) for rival, due in deficits.items()
               if rival not in collisions)


def sample_identity_seed_pools(cfg: Config, count: int) -> tuple[tuple[int, ...], ...]:
   
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
                knowledge: PlanningKnowledge | None = None,
                pk_a_slots: tuple[tuple[int, int, int | None], ...] | None = None,
                _pk_assumed: int | None = None,
                _pk_best_effort: bool = False) -> LengthPlan:
   
    configure_symbol_size(cfg.symbol_size)
    known = (knowledge if knowledge is not None else
             sample_planning_knowledge(data, cfg, cfg.seed_strategy_seed + 707))
    if pk_a_slots is not None and available is None:
        available = sum(value is not None for _, _, value in pk_a_slots)
    if pk_a_slots is not None and _pk_assumed is None:
        
        cap = available
        candidate = length_plan(data, cfg, fake_total, n_a, available,
                                excluded_lengths, known, pk_a_slots, cap)
        if candidate.target is not None:
            return candidate
        recovery = length_plan(data, cfg, fake_total, n_a, available,
                               excluded_lengths, known, pk_a_slots, cap,
                               _pk_best_effort=True)
        if recovery.target is not None:
            recovery.attempts.append({
                "reason": "pk_a_fixed_capacity_best_effort",
                "assumed_replacements": cap,
                "theoretical_feasible": recovery.feasible})
        return recovery
    domain = tuple(range(cfg.length_min, cfg.length_max + 1))
    histogram = Counter(min(cfg.length_max, max(cfg.length_min, len(s))) for s in known.train)
    pi = {v: histogram[v] / len(known.train) for v in domain}
    references, H = length_reference(known, cfg.reference_coverage,
                                     allow_missing=cfg.attack_mode == "pk")
    
    
    reference_min = min(H.values())
    bmin = ceil(cfg.length_min_multiplier * cfg.alpha * fake_total)
    if pk_a_slots is not None and available is not None:
        bmin = bmax = available
    else:
        bmax = min(floor(cfg.length_max_multiplier * cfg.alpha * fake_total),
                   fake_total if available is None else available)
    response = LengthPlan(reference_min=reference_min,
                          knowledge_fingerprint=known.fingerprint)
    if bmax < bmin or bmax == 0:
        response.attempts.append({"reason": "infeasible A budget bounds"})
        return response
    
    
    if pk_a_slots is None:
        expected_support = {v: honest_length_support_mean(
            pi, n_a, cfg.epsilon, domain, cfg.length_mechanism, v) for v in domain}
        pk_moments = None
    else:
        expected_support, pk_moments = pk_length_estimator(
            pi, cfg, domain, pk_a_slots, _pk_assumed,
            len(data.train) - len(known.indices))
    clean_mode = select_length_output(Counter(expected_support), domain)
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
    best_effort = None
    for t in preference:
        if t < 2:
            
            
            response.attempts.append({"length": t, "reference_min": reference_min,
                                      "reason": "Symbols B needs at least one length-2 pair",
                                      "feasible": False})
            continue
        gaps: dict[int, int] = {}
        for rival in domain:
            if rival == t:
                continue
            mu, sigma = (pk_moments(t, rival) if pk_moments is not None else
                         honest_length_moments(pi, n_a, cfg.epsilon, domain,
                                               cfg.length_mechanism, t, rival, len(data.train)))
            gaps[rival] = max(0, ceil(mu + factor * sigma + 1))
        reports: list[Any] = []
        if cfg.length_mechanism == "grr":
            reports = [t] * (bmax if _pk_best_effort else
                             max(gaps.values(), default=0))
        else:
            
            
            
            still_due = dict(gaps)
            used_seeds: set[int] = set()
            for identity, local_seeds in enumerate(seed_pools):
                if not _pk_best_effort and identity >= bmin and all(
                        x == 0 for x in still_due.values()):
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
        if _pk_best_effort:
            
            theoretical_min = max(gaps.values(), default=0)
            score = (theoretical_min, sum(max(0, v) for v in deficits.values()),
                     -sum(abs(t - H[c]) for c in H), t)
            if best_effort is None or score < best_effort[0]:
                best_effort = (score, t, reports, gaps, gains, feasible)
            continue
        if feasible:
            response.target, response.reports, response.min_required = t, reports, required
            response.bounds = {k: float(v) for k, v in gaps.items()}
            response.net_gains = gains
            response.feasible = True
            response.strategy = "analytically_feasible"
            return response
    if best_effort is not None:
        score, target, reports, gaps, gains, feasible = best_effort
        response.target, response.reports, response.min_required = (
            target, reports, score[0])
        response.bounds = {k: float(v) for k, v in gaps.items()}
        response.net_gains = gains
        response.feasible = feasible
        response.strategy = ("pk_a_full_budget_recovery" if feasible else
                             "pk_best_effort_min_theoretical_demand")
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


def all_class_diagnostics(s: str,
                          references: dict[int, tuple[tuple[str, float], ...]],
                          search_deadline: float | None = None
                          ) -> tuple[dict[int, tuple[float, float, float, float]],
                                     tuple[float, ...]]:
    distances: dict[int, float] = {}
    normalized: dict[int, float] = {}
    nearest: dict[int, float] = {}
    for cls, refs in references.items():
        raw = []
        for ref, _ in refs:
            _check_deadline(search_deadline)
            raw.append(full_edit(s, ref))
        distances[cls] = sum(weight * value
                             for (_, weight), value in zip(refs, raw))
        normalized[cls] = sum(weight * (value / max(len(s), len(ref)))
                              for (ref, weight), value in zip(refs, raw))
        nearest[cls] = min(raw)
    diagnostics = {}
    for cls in references:
        others = [other for other in references if other != cls]
        diagnostics[cls] = (
            distances[cls],
            min(distances[other] for other in others) - distances[cls],
            min(normalized[other] for other in others) - normalized[cls],
            min(nearest[other] for other in others) - nearest[cls],
        )
    return diagnostics, tuple(distances[cls] for cls in sorted(references))


def position_expected(data: Dataset | PlanningKnowledge, L: int, n_b: int,
                      epsilon: float) -> dict[int, dict[str, float]]:
    
    p, q = grr_prob(epsilon, len(PAIRS))
    raw = defaultdict(lambda: defaultdict(float))
    for sequence, multiplicity in Counter(data.train).items():
        clipped = sequence[:L].ljust(L, sequence[-1])
        for pos in range(L - 1):
            share = multiplicity / len(data.train) / (L - 1)
            raw[pos][clipped[pos:pos + 2]] += share
    return {pos: {pair: n_b * (q / (L - 1) + (p - q) * raw[pos][pair])
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
   
    k = cfg.branch_factor * cfg.top_k
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
                graph_override: dict[int, tuple[str, ...]] | None = None,
                search_deadline: float | None = None
                ) -> CandidatePools:
  
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
    
    
    
    
    
    
    
    core = pair_graph_from_counts(expectations, L, cfg.branch_factor * cfg.top_k)
    rank_index = cfg.branch_factor * cfg.top_k - 1
    ranked_last_kept = [sorted(expectations[pos].values(), reverse=True)[rank_index]
                     if len(expectations[pos]) > rank_index else 0.0
                     for pos in range(L - 1)]
    
    
    deadline = search_deadline
    if deadline is None and cfg.search_timeout_seconds is not None and cfg.search_timeout_seconds > 0:
        deadline = time.monotonic() + cfg.search_timeout_seconds
    classes = tuple(sorted(references))
    prefix_lengths = {cls: min(L, lengths[cls]) for cls in classes}
    ref_shapes = {cls: {s for s, _ in references[cls]} for cls in classes}
    
    
    adjacency: dict[int, dict[str, tuple[str, ...]]] = {}
    for pos, edges in graph.items():
        grouped: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            grouped[edge[0]].append(edge)
        adjacency[pos] = {symbol: tuple(outgoing)
                          for symbol, outgoing in grouped.items()}
    found_by_class: dict[int, list[Target]] = {cls: [] for cls in classes}
    stats = {cls: {"prefixes_checked": 0, "rejected_prefixes": 0,
                   "complete_leaves_checked": 0, "rejected_full_leaves": 0}
             for cls in classes}
    pruned_entire_subtrees = 0
    first_chars = list(dict.fromkeys(edge[0] for edge in graph.get(0, ())))
    
    
    initial = {cls: None for cls in classes}
    stack = [(first, 0, initial) for first in reversed(first_chars)]
    try:
        while stack:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Candidate graph enumeration interrupted")
            prefix, spent, active = stack.pop()
            reaching_T = [cls for cls in active
                          if active[cls] is None and len(prefix) == prefix_lengths[cls]]
            profile = None
            if reaching_T:
                profile = all_class_diagnostics(prefix, references, deadline)
                active = active.copy()
                for cls in reaching_T:
                    stats[cls]["prefixes_checked"] += 1
                    metrics = profile[0][cls]
                    if metrics[1] <= 1e-12:
                        stats[cls]["rejected_prefixes"] += 1
                        del active[cls]
                    else:
                        active[cls] = (metrics, profile[1])
            if not active:
                pruned_entire_subtrees += 1
                continue
            if len(prefix) == L:
                full_profile = profile or all_class_diagnostics(prefix, references, deadline)
                full_metrics, full_distances = full_profile
                for cls, (prefix_metrics, prefix_distances) in active.items():
                    stats[cls]["complete_leaves_checked"] += 1
                    fd, fm, fnm, fn = full_metrics[cls]
                    if fm <= 1e-12:
                        stats[cls]["rejected_full_leaves"] += 1
                        continue
                    d, margin, nmargin, near = prefix_metrics
                    original_prefix = prefix[:prefix_lengths[cls]]
                    ratios = [expectations[pos].get(prefix[pos:pos + 2], 0.0) / cutoff
                              for pos, cutoff in enumerate(ranked_last_kept)
                              if cutoff > 0]
                    found_by_class[cls].append(Target(
                        cls, original_prefix, prefix, d, margin, near,
                        sum(prefix[pos:pos + 2] not in core[pos]
                            for pos in range(L - 1)),
                        nmargin,
                        nmargin > 1e-12 and near > 1e-12
                        and original_prefix not in ref_shapes[cls],
                        prefix_distances,
                        min(ratios) if len(ratios) == L - 1 else None,
                        0, fd, fm, fnm, fn, full_distances))
                continue
            pos = len(prefix) - 1
            extensions = []
            for edge in adjacency.get(pos, {}).get(prefix[-1], ()):
                cost = spent + lower[pos][edge]
                if graph_override is not None or cost <= optimistic_b_budget:
                    extensions.append((prefix + edge[1], cost, active))
            stack.extend(reversed(extensions))
    except TimeoutError:
        pools.search_complete = False

    for cls in classes:
        found = found_by_class[cls]
        counts = Counter(t.prefix for t in found)
        found = [replace(t, path_count=counts[t.prefix]) for t in found]
        found.sort(key=lambda t: (-t.full_deviation, -t.margin, t.full))
        pools.qualified[cls] = found
        pools.priority[cls] = found  
        measured = stats[cls]
        
        
        
        all_leaves_counted = (pools.search_complete and
                              measured["rejected_prefixes"] == 0)
        pools.diagnostics[cls] = {
            "prefix_length": prefix_lengths[cls],
            "examined_complete_paths": measured["complete_leaves_checked"],
            "legal_complete_paths": (measured["complete_leaves_checked"]
                                     if all_leaves_counted else None),
            "paths_enumerated": measured["complete_leaves_checked"],
            "examined_prefixes": measured["prefixes_checked"],
            "nonpositive_margin": measured["rejected_prefixes"],
            "nonpositive_complete_margin": measured["rejected_full_leaves"],
            "qualifying_prefixes": len(counts),
            "qualified_full_leaves": len(found),
            "pruned_prefix_branches_for_class": measured["rejected_prefixes"],
            "pruned_subtrees_for_all_classes": pruned_entire_subtrees,
            "shared_reference_profiles": True,
            "candidate_pair_counts": {str(j): len(v) for j, v in graph.items()},
            "optimistic_b_identity_budget": optimistic_b_budget,
            "search_complete": pools.search_complete,
            "qualified_scope": "All budget-unexcluded position paths; exact prefix pruning; no fixed cap"}
    return pools


def supports_to_top_k(items: tuple[str, ...], protected: Iterable[str],
                      mean: dict[str, float], population: int, k: int,
                      z: float) -> dict[str, int] | None:
    
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
                                   cfg.branch_factor * cfg.top_k, cfg.score_z)
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
                        n_d: int, cfg: Config, sample: list[int] | None = None,
                        search_deadline: float | None = None
                        ) -> dict[tuple[str, int], float]:
    q = 1 / (exp(cfg.epsilon) + 1)
    p = 0.5
    if sample is None:
        population = Counter(zip(data.train, data.labels))
    else:
        population = Counter((data.train[i], data.labels[i]) for i in sample)
    totals = Counter()
    for (sequence, label), multiplicity in population.items():
        def cost(shape: str) -> float:
            _check_deadline(search_deadline)
            return trace_distance(sequence, shape)

        chosen = min(candidates, key=cost)
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
    initial_required: int | None = None
    failure_reason: str | None = None
    padding_reports: int = 0
    budget_sufficient: bool | None = None
    pk_c_raw_requirements: tuple[tuple[int, int, int], ...] = ()


def pk_fixed_votes(required: dict[Any, int], cap: int, fallback: Any) -> dict[Any, int]:
    if cap < 0 or any(v < 0 for v in required.values()):
        raise ValueError("Invalid PK fixed-report budget")
    total = sum(required.values())
    if not total:
        return {fallback: cap} if cap else {}
    if total <= cap:
        votes = dict(required)
        votes[fallback] = votes.get(fallback, 0) + cap - total
        return {k: v for k, v in votes.items() if v}
    scaled = {k: required[k] * cap // total for k in required}
    left = cap - sum(scaled.values())
    order = sorted(required, key=lambda k: (-(required[k] * cap % total), str(k)))
    for key in order[:left]:
        scaled[key] += 1
    return {k: v for k, v in scaled.items() if v}


def oue_ones_cap(K: int, candidate_count: int, epsilon: float) -> int:
    if K < 1 or candidate_count < 1 or epsilon <= 0:
        raise ValueError("OUE cap requires positive K, candidate count and epsilon")
    d = K * candidate_count
    return max(1, floor(0.5 + (d - 1) / (exp(epsilon) + 1)))


def d_ratio_reserve(honest_total: int, honest_a: int, honest_d: int,
                    fake_total: int, fake_a: int) -> tuple[float, int]:
    if (honest_total <= honest_a or honest_d < 0 or honest_d > honest_total - honest_a
            or fake_a < 0 or fake_a > fake_total):
        raise ValueError("Invalid honest D ratio or malicious A allocation")
    gamma = honest_d / (honest_total - honest_a)
    return gamma, ceil(gamma * (fake_total - fake_a))


def choose_oue_plan(candidates: tuple[str, ...], desired: dict[int, str],
                    projected_means: dict[tuple[str, int], float], cfg: Config,
                    search_deadline: float | None = None
                    ) -> DPlan:
    K, M = cfg.top_k, len(candidates)
    if (not 1 <= len(desired) <= K or len(set(desired.values())) != len(desired)
            or not set(desired.values()) <= set(candidates)):
        return DPlan(feasible=False)
    
    b = oue_ones_cap(K, M, cfg.epsilon)
    choices: list[DPlan] = []
    for order in permutations(sorted(desired)):
        _check_deadline(search_deadline)
        increments: dict[tuple[str, int], int] = {}
        for index in range(len(order) - 1, -1, -1):
            _check_deadline(search_deadline)
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
    
    rng = np.random.default_rng(seed)
    users = list(map(int, rng.permutation(n)))
    knowledge_n = floor(0.10 * n)
    na = floor(0.20 * knowledge_n)
    nb = knowledge_n - na
    after_knowledge = n - knowledge_n
    if length < 1:
        raise ValueError("The number of prefix-tree levels must be positive")
    level_n = floor(cfg.tree_level_fraction * after_knowledge / length)
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
    pk_budget_sufficient: bool | None = None
    pk_total_shortfall: int = 0
    pk_stage_demands: dict[str, Any] = field(default_factory=dict)


class Planner:

    def __init__(self, data: Dataset, cfg: Config, seed: int = 9128,
                 knowledge: PlanningKnowledge | None = None):
        configure_symbol_size(cfg.symbol_size)
        self.data, self.cfg = data, cfg
        self.knowledge = (knowledge if knowledge is not None else
                          sample_planning_knowledge(data, cfg, seed))
        self.search_profile_name, self.search_limits = search_time_profile(
            cfg, self.knowledge)
        self.search_time_spent: dict[str, float] = defaultdict(float)
        self.search_deadline_overruns: dict[str, float] = defaultdict(float)
        self.search_calls: Counter = Counter()
        
        
        self.c_layer_search_spent: dict[int, float] = defaultdict(float)
        self.c_layer_search_calls: dict[int, Counter] = defaultdict(Counter)
        self.c_baseline_spent: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        self.c_baseline_calls: dict[int, Counter] = defaultdict(Counter)
        self._active_search_deadline: float | None = None
        self._deadline_stack: list[float | None] = []
        self.future_cache: OrderedDict = OrderedDict()
        self.future_cache_hits = self.future_cache_misses = 0
        self.references, self.ref_lengths = length_reference(
            self.knowledge, cfg.reference_coverage,
            allow_missing=cfg.attack_mode == "pk")
        
        
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
        self.n_b = self.n_c = self.n_d = self.c_per_level = 0
        self.d_gamma = 0.0
        self.d_base_reserve = 0
        self.d_locked = 0
        self.optimistic_b_budget = 0
        self.pair_lower: dict[int, dict[str, int]] = {}
        self.candidate_graph: dict[int, tuple[str, ...]] = {}
        self.pool_search_complete = True
        self.pk_stage_caps: dict[str, Any] | None = None
        self.pk_stage_users: dict[str, Any] | None = None
        self.pk_known = {uid: (seq, cls) for uid, seq, cls in zip(
            self.knowledge.indices, self.knowledge.train, self.knowledge.labels)}
        self.pk_b_cache: dict[int, dict[int, dict[str, float]]] = {}
        self.pk_c_prefix_cache: dict[tuple[tuple[str, ...], int], np.ndarray] = {}
        self.pk_d_choice_cache: dict[tuple[str, ...], list[tuple[str, int]]] = {}

    def _start_search(self, kind: str,
                      *, layer: int | None = None) -> tuple[float | None, float]:
        started = time.monotonic()
        self.search_calls[kind] += 1
        seconds = self.search_limits[kind]
        if layer is not None:
            if layer < 0 or kind not in ("c_dynamic", "c_budget"):
                raise ValueError("Only C search events may consume a C layer budget")
            self.c_layer_search_calls[layer][kind] += 1
            layer_cap = self.search_limits["c_layer"]
            remaining = (None if layer_cap is None else
                         max(0.0, layer_cap - self.c_layer_search_spent[layer]))
        else:
            remaining = None
        if remaining is not None:
            seconds = remaining if seconds is None else min(seconds, remaining)
        deadline = (started + seconds) if seconds is not None else None
        self._deadline_stack.append(self._active_search_deadline)
        if deadline is not None and self._active_search_deadline is not None:
            deadline = min(deadline, self._active_search_deadline)
        self._active_search_deadline = deadline
        return deadline, started

    def _finish_search(self, kind: str, started: float,
                       *, layer: int | None = None) -> None:
        finished = time.monotonic()
        elapsed = finished - started
        self.search_time_spent[kind] += elapsed
        if layer is not None:
            self.c_layer_search_spent[layer] += elapsed
        if self._active_search_deadline is not None:
            self.search_deadline_overruns[kind] += max(
                0.0, finished - self._active_search_deadline)
        self._active_search_deadline = self._deadline_stack.pop()

    def search_diagnostics(self) -> dict[str, Any]:
        return {"profile": self.search_profile_name,
                "pk_fixed_full_replacement": self.pk_stage_caps is not None,
                "knowledge_ratio": len(self.knowledge.train) / len(self.data.train),
                "knowledge_size": len(self.knowledge.train),
                "honest_training_size": len(self.data.train),
                "time_limits_seconds": dict(self.search_limits),
                "actual_seconds_by_kind": {key: round(value, 4)
                                           for key, value in self.search_time_spent.items()},
                "deadline_overrun_seconds_by_kind": {
                    key: round(value, 4)
                    for key, value in self.search_deadline_overruns.items()},
                "deadline_policy": "cooperative_check_inside_cost_projections",
                "strict_wall_clock_preemption": False,
                "calls_by_kind": dict(self.search_calls),
                "c_layer_search_seconds": {
                    str(layer): round(seconds, 4)
                    for layer, seconds in sorted(self.c_layer_search_spent.items())},
                "c_layer_search_calls": {
                    str(layer): dict(calls)
                    for layer, calls in sorted(self.c_layer_search_calls.items())},
                "c_baseline_seconds_by_depth": {
                    str(depth): {kind: round(seconds, 4)
                                 for kind, seconds in costs.items()}
                    for depth, costs in sorted(self.c_baseline_spent.items())},
                "c_baseline_calls_by_depth": {
                    str(depth): dict(calls)
                    for depth, calls in sorted(self.c_baseline_calls.items())},
                "future_cost_cache_hits": self.future_cache_hits,
                "future_cost_cache_misses": self.future_cache_misses}

    def _measure_c_baseline(self, depth: int, kind: str, action):
        if self._deadline_stack:
            return action()
        started = time.monotonic()
        self.c_baseline_calls[depth][kind] += 1
        try:
            return action()
        finally:
            self.c_baseline_spent[depth][kind] += time.monotonic() - started

    def baseline_em_means(self, candidates: tuple[str, ...], depth: int
                          ) -> dict[str, float]:
        return self._measure_c_baseline(
            depth, "em_means", lambda: self.em_means(candidates, depth))

    def baseline_future_cost(self, layer: int,
                             graph: dict[int, tuple[str, ...]],
                             targets: dict[int, Target], start_depth: int,
                             survivors: list[str] | None
                             ) -> tuple[int, tuple[str, ...], DPlan] | None:
        return self._measure_c_baseline(
            layer, "future_cost", lambda: self.future_cost(
                graph, targets, start_depth, survivors))

    def setup(self, L: int, n_b: int, n_c: int, n_d: int,
              *, fake_total: int | None = None, fake_a: int = 0,
              honest_a: int | None = None, build_candidates: bool = True,
              pk_stage_caps: dict[str, Any] | None = None,
              pk_stage_users: dict[str, Any] | None = None) -> None:
        if L <= 0 or n_c % L:
            raise ValueError("C normal users must form L equal-sized disjoint groups")
        self.L, self.n_b, self.n_c, self.n_d = L, n_b, n_c, n_d
        self.pk_stage_caps = pk_stage_caps
        self.pk_stage_users = pk_stage_users
        self.pk_b_cache.clear()
        self.pk_c_prefix_cache.clear()
        self.pk_d_choice_cache.clear()
        self.c_per_level = n_c // L
        self.d_gamma = 0.0
        self.d_base_reserve = 0
        self.d_locked = 0
        if fake_total is not None:
            self.d_gamma, self.d_base_reserve = d_ratio_reserve(
                len(self.data.train),
                floor(0.2 * floor(0.1 * len(self.data.train)))
                if honest_a is None else honest_a,
                n_d, fake_total, fake_a)
        if pk_stage_caps is not None:
            self.d_base_reserve = ceil(self.d_gamma * pk_stage_caps["D"])
        self.expectations = (self.pk_b_means(len(pk_stage_users["B"])) if pk_stage_caps is not None else
                             position_expected(self.knowledge, L, n_b, self.cfg.epsilon))
        self.optimistic_b_budget = (pk_stage_caps["B"] if pk_stage_caps is not None
                                    else max(0, (fake_total or 0) - fake_a -
                                             self.d_base_reserve))
        self.candidate_graph, self.pair_lower = candidate_pair_graph(
            self.expectations, L, self.cfg, self.optimistic_b_budget)
        if build_candidates:
            deadline, started = self._start_search("initial_candidates")
            try:
                candidate_pools = target_pool(
                    self.knowledge, self.cfg, L, self.expectations,
                    self.references, self.ref_lengths, self.optimistic_b_budget,
                    search_deadline=deadline)
            finally:
                self._finish_search("initial_candidates", started)
        else:
            candidate_pools = CandidatePools()
        self.pools = candidate_pools.priority
        self.qualified_pools = candidate_pools.qualified
        self.pool_diagnostics = candidate_pools.diagnostics
        self.pool_search_complete = candidate_pools.search_complete
        self.em_cache.clear()
        self.d_cache.clear()
        self.future_cache.clear()

    def pk_b_means(self, replaced: int) -> dict[int, dict[str, float]]:

        if self.pk_stage_caps is None or self.pk_stage_users is None:
            raise ValueError("PK B predictions require real group membership")
        ids = self.pk_stage_users["B"]
        if not 0 <= replaced <= len(ids):
            raise ValueError("Invalid PK B replacement count")
        if replaced not in self.pk_b_cache:
            means = position_expected(self.knowledge, self.L,
                                      self.n_b - len(ids), self.cfg.epsilon)
            p, q = grr_prob(self.cfg.epsilon, len(PAIRS))
            for uid in ids[replaced:]:
                _check_deadline(self._active_search_deadline)
                seq = self.pk_known[uid][0]
                clipped = seq[:self.L].ljust(self.L, seq[-1])
                for pos in range(self.L - 1):
                    true_pair = clipped[pos:pos + 2]
                    for pair in PAIRS:
                        means[pos][pair] += (q + (p - q) * (pair == true_pair)) / (self.L - 1)
            self.pk_b_cache[replaced] = means
        return self.pk_b_cache[replaced]

    def pk_b_topups(self, targets: tuple[str, ...], detail: dict[str, Any] | None = None
                     ) -> tuple[dict[int, dict[str, int]], dict[int, dict[str, float]]] | None:
        if not targets or any(len(target) < 2 or any(
                target[pos:pos + 2] not in PAIRS for pos in range(len(target) - 1))
                for target in targets):
            return None
        cap = self.pk_stage_caps["B"]
        _check_deadline(self._active_search_deadline)
        projected = self.pk_b_means(cap)
        additions = position_topups(projected, targets, self.cfg, self.n_b)
        if additions is None:
            return None
        flat = {(pos, pair): amount for pos, votes in additions.items()
                for pair, amount in votes.items()}
        if detail is not None:
            detail.update(raw_required=sum(flat.values()), capacity=cap)
        allocated = pk_fixed_votes(flat, cap, (0, targets[0][:2]))
        result: dict[int, dict[str, int]] = {}
        for (pos, pair), amount in allocated.items():
            result.setdefault(pos, {})[pair] = amount
        assert sum(sum(v.values()) for v in result.values()) == cap
        return result, projected

    def pk_c_topups(self, candidates: tuple[str, ...], protected: list[str], depth: int,
                    detail: dict[str, Any] | None = None
                     ) -> tuple[dict[str, int] | None, dict[str, float]]:
        if (not set(protected) <= set(candidates) or
                len(set(protected)) > self.cfg.branch_factor * self.cfg.top_k):
            return None, {}
        cap = self.pk_stage_caps["C"][depth]
        _check_deadline(self._active_search_deadline)
        projected = self.em_means(candidates, depth, cap)
        additions = supports_to_top_k(
            candidates, protected, projected, self.c_per_level,
            self.cfg.branch_factor * self.cfg.top_k, self.cfg.score_z)
        if additions is None:
            return None, projected
        if detail is not None:
            detail.update(raw_required=sum(additions.values()), capacity=cap, depth=depth)
        result = pk_fixed_votes(additions, cap, protected[0] if protected else candidates[-1])
        assert sum(result.values()) == cap
        return result, projected

    def pk_d_plan(self, leaves: tuple[str, ...], targets: dict[int, str]) -> DPlan:
        if (not 1 <= len(targets) <= self.cfg.top_k or
                len(set(targets.values())) != len(targets) or
                not set(targets.values()) <= set(leaves)):
            return DPlan(feasible=False, failure_reason="invalid_d_target_binding")
        cap = self.pk_stage_caps["D"]
        _check_deadline(self._active_search_deadline)
        plan = choose_oue_plan(leaves, targets, self.d_means(leaves, cap),
                               self.cfg, search_deadline=self._active_search_deadline)
        if not plan.feasible:
            plan.failure_reason = "pk_d_invalid_plan"
            return plan
        required = plan.identities
        plan.budget_sufficient = required <= cap
        plan.initial_required = required
        if required > cap:
            plan.failure_reason = "pk_d_best_effort_under_capacity"
            fallback = (targets[sorted(targets)[0]], sorted(targets)[0])
            plan.increments = pk_fixed_votes(plan.increments, cap, fallback)
        plan.identities = cap
        plan.padding_reports = max(0, cap - required)
        return plan

    def pk_d_budget_reselect(self, leaves: tuple[str, ...], targets: dict[int, Target],
                             events: list[dict[str, Any]]) -> tuple[dict[int, Target], DPlan]:
        cap = self.pk_stage_caps["D"]
        original = (self.pk_d_plan(leaves, {c: t.full for c, t in targets.items()})
                    if targets and all(t.full in leaves for t in targets.values())
                    else DPlan(feasible=False, failure_reason="missing_target_leaf"))
        if (not self.cfg.allow_dynamic_reselect or
                (original.feasible and (original.initial_required or 0) <= cap)):
            return targets, original
        deadline, started = self._start_search("d_reselect")
        checked = 0
        interrupted = False
        best: tuple[dict[int, Target], DPlan] | None = None
        best_key: tuple[Any, ...] | None = None
        classes = sorted(targets)
        try:
            for size in range(len(classes), 0, -1):
                for subset in combinations(classes, size):
                    choices = []
                    for cls in subset:
                        valid = [t for t in self.qualified_pools.get(cls, ())
                                 if t.full in leaves and t.margin > 1e-12
                                 and t.full_margin > 1e-12]
                        valid.sort(key=lambda t: (t.full != targets[cls].full,
                                                  t.prefix != targets[cls].prefix,
                                                  -t.full_deviation, t.full))
                        choices.append(valid)
                    if any(not group for group in choices):
                        continue
                    for combo in product(*choices):
                        if deadline is not None and time.monotonic() >= deadline:
                            interrupted = True
                            break
                        if len({t.full for t in combo}) != len(combo):
                            continue
                        checked += 1
                        proposed = {t.cls: t for t in combo}
                        try:
                            plan = self.pk_d_plan(
                                leaves, {c: t.full for c, t in proposed.items()})
                        except TimeoutError:
                            interrupted = True
                            break
                        if not plan.feasible:
                            continue
                        raw = plan.initial_required or 0
                        shortage = max(0, raw - cap)
                        unchanged = sum(t.full == targets[t.cls].full for t in combo)
                        prefixes = sum(t.prefix == targets[t.cls].prefix for t in combo)
                        key = (int(shortage == 0),
                               size if shortage == 0 else -shortage,
                               size, unchanged, prefixes,
                               sum(t.full_deviation for t in combo),
                               tuple(t.full for t in combo))
                        if best_key is None or key > best_key:
                            best, best_key = (proposed, plan), key
                    if interrupted:
                        break
                if interrupted or (best_key is not None and best_key[0] == 1
                                   and len(best[0]) == size):
                    break
        finally:
            self._finish_search("d_reselect", started)
        chosen, plan = best if best is not None else (targets, original)
        events.append({"event": "pk_d_budget_reselect",
                       "targets_before": {str(c): t.full for c, t in targets.items()},
                       "targets_after": {str(c): t.full for c, t in chosen.items()},
                       "raw_required_before": original.initial_required,
                       "raw_required_after": plan.initial_required,
                       "capacity": cap, "checked_combinations": checked,
                       "search_complete": not interrupted and self.pool_search_complete,
                       "budget_sufficient": plan.budget_sufficient})
        return chosen, plan

    def after_b_graph(self, actual_graph: dict[int, tuple[str, ...]]) -> None:

        deadline, started = self._start_search("after_b_candidates")
        try:
            candidate_pools = target_pool(
                self.knowledge, self.cfg, self.L, self.expectations, self.references,
                self.ref_lengths, self.optimistic_b_budget,
                graph_override=actual_graph, search_deadline=deadline)
        finally:
            self._finish_search("after_b_candidates", started)
        self.future_cache.clear()
        self.pools = candidate_pools.priority
        self.qualified_pools = candidate_pools.qualified
        self.pool_diagnostics = candidate_pools.diagnostics
        self.pool_search_complete = candidate_pools.search_complete

    def em_means(self, candidates: tuple[str, ...], depth: int,
                 replaced: int = 0) -> dict[str, float]:
        _check_deadline(self._active_search_deadline)
        key = (candidates, depth, replaced if self.pk_stage_caps else 0)
        if key not in self.em_cache:
            total = np.zeros(len(candidates))
            if depth not in self.prefix_frequencies:
                frequencies = Counter()
                for sequence, multiplicity in self.empirical_sequences.items():
                    _check_deadline(self._active_search_deadline)
                    frequencies[sequence[:depth + 1]] += multiplicity
                self.prefix_frequencies[depth] = frequencies
            for sequence, multiplicity in self.prefix_frequencies[depth].items():
                _check_deadline(self._active_search_deadline)
                total += multiplicity * em_probs(
                    sequence, candidates, self.cfg.epsilon,
                    search_deadline=self._active_search_deadline)
            if self.pk_stage_caps is None:
                result = total * self.c_per_level / self.empirical_size
            else:
                controlled = self.pk_stage_users["C"][depth]
                if not 0 <= replaced <= len(controlled):
                    raise ValueError("Invalid PK C replacement count")
                result = total * self.pk_stage_caps["C_honest"][depth] / self.empirical_size
                prefix_key = (candidates, depth)
                if prefix_key not in self.pk_c_prefix_cache:
                    sums = np.zeros((len(controlled) + 1, len(candidates)))
                    for idx, uid in enumerate(controlled):
                        _check_deadline(self._active_search_deadline)
                        sums[idx + 1] = (sums[idx] + em_probs(
                            self.pk_known[uid][0], candidates, self.cfg.epsilon,
                            search_deadline=self._active_search_deadline))
                    self.pk_c_prefix_cache[prefix_key] = sums
                sums = self.pk_c_prefix_cache[prefix_key]
                result += sums[-1] - sums[replaced]
            self.em_cache[key] = dict(zip(candidates, result))
        return self.em_cache[key]

    def d_means(self, candidates: tuple[str, ...],
                replaced: int = 0) -> dict[tuple[str, int], float]:
        _check_deadline(self._active_search_deadline)
        key = (candidates, replaced if self.pk_stage_caps else 0)
        if key not in self.d_cache:
            if self.pk_stage_caps is None:
                means = projected_oue_means(
                    self.knowledge, candidates, self.n_d, self.cfg,
                    search_deadline=self._active_search_deadline)
            else:
                controlled = self.pk_stage_users["D"]
                if not 0 <= replaced <= len(controlled):
                    raise ValueError("Invalid PK D replacement count")
                means = projected_oue_means(
                    self.knowledge, candidates, self.n_d - len(controlled), self.cfg,
                    search_deadline=self._active_search_deadline)
                q = 1 / (exp(self.cfg.epsilon) + 1)
                delta = 0.5 - q
                if candidates not in self.pk_d_choice_cache:
                    choices = []
                    for uid in controlled:
                        _check_deadline(self._active_search_deadline)
                        seq, label = self.pk_known[uid]
                        closest = min(candidates, key=lambda s: trace_distance(seq, s))
                        choices.append((closest, label))
                    self.pk_d_choice_cache[candidates] = choices
                observed = Counter(self.pk_d_choice_cache[candidates][replaced:])
                count = len(controlled) - replaced
                means = {cell: v + count * q + delta * observed[cell]
                         for cell, v in means.items()}
            self.d_cache[key] = means
        return self.d_cache[key]

    def future_cost(self, graph: dict[int, tuple[str, ...]],
                    targets: dict[int, Target], start_depth: int = 0,
                    survivors: list[str] | None = None
                    ) -> tuple[int, tuple[str, ...], DPlan] | None:
        _check_deadline(self._active_search_deadline)
        key = (self.L, start_depth, self.d_locked, self.d_base_reserve,
               tuple((cls, target.full) for cls, target in targets.items()),
               tuple((pos, tuple(edges)) for pos, edges in sorted(graph.items())),
               None if survivors is None else tuple(survivors))
        if key in self.future_cache:
            self.future_cache_hits += 1
            self.future_cache.move_to_end(key)
            return self.future_cache[key]
        self.future_cache_misses += 1
        result = self._future_cost_uncached(graph, targets, start_depth, survivors)
        _check_deadline(self._active_search_deadline)
        self.future_cache[key] = result
        if len(self.future_cache) > 4096:
            self.future_cache.popitem(last=False)
        return result

    def _future_cost_uncached(self, graph: dict[int, tuple[str, ...]],
                             targets: dict[int, Target], start_depth: int,
                             survivors: list[str] | None
                             ) -> tuple[int, tuple[str, ...], DPlan] | None:

        if not targets or len(targets) > self.cfg.top_k:
            return None
        parents = [""] if survivors is None else survivors[:]
        used = 0
        pk_c_requirements: list[tuple[int, int, int]] = []
        for depth in range(start_depth, self.L):
            _check_deadline(self._active_search_deadline)
            options = expand_paths(parents, graph, depth)
            if not options:
                return None
            protect = [target.full[:depth + 1] for target in targets.values()]
            if self.pk_stage_caps is None:
                mean = self.em_means(tuple(options), depth)
                addition = supports_to_top_k(
                    tuple(options), protect, mean, self.c_per_level,
                    self.cfg.branch_factor * self.cfg.top_k, self.cfg.score_z)
            else:
                detail: dict[str, Any] = {}
                addition, mean = self.pk_c_topups(tuple(options), protect, depth, detail)
                if detail:
                    pk_c_requirements.append((depth, detail["raw_required"],
                                              detail["capacity"]))
            if addition is None:
                return None
            if (self.pk_stage_caps is not None and
                    sum(addition.values()) > self.pk_stage_caps["C"][depth]):
                return None
            used += sum(addition.values())
            scores = {node: mean[node] + addition.get(node, 0) for node in options}
            parents = top_nodes_in_creation_order(options, scores,
                                                  self.cfg.branch_factor * self.cfg.top_k)
            if not set(protect) <= set(parents):
                return None
        final = tuple(parents)
        if len(final) < self.cfg.top_k:
            return None
        d_plan = (self.pk_d_plan(final, {c: t.full for c, t in targets.items()})
                  if self.pk_stage_caps is not None else
                  choose_oue_plan(final, {c: t.full for c, t in targets.items()},
                                  self.d_means(final), self.cfg,
                                  search_deadline=self._active_search_deadline))
        if not d_plan.feasible:
            return None
        if self.pk_stage_caps is not None:
            d_plan.pk_c_raw_requirements = tuple(pk_c_requirements)
        if self.pk_stage_caps is not None and d_plan.identities > self.pk_stage_caps["D"]:
            return None
        return used + max(d_plan.identities, self.d_base_reserve, self.d_locked), final, d_plan

    def joint_initial(self, remaining: int) -> JointPlan:
        safe = ceil(self.cfg.safety_share * floor(self.cfg.beta * len(self.data.train)))
        available_classes = [c for c in sorted(self.references)
                             if self.qualified_pools.get(c)]
        checked = 0
        budget_pruned = 0
        bound_pruned = 0
        tail_upper_bound_skips = 0
        total_upper = 0
        best: JointPlan | None = None
        best_effort: JointPlan | None = None
        interrupted = False
        deadline, started = self._start_search("joint_initial")
        class_pools = {c: sorted(self.qualified_pools[c],
                                 key=lambda t: (-t.full_deviation, t.full))
                       for c in available_classes}

        
        
        
        @lru_cache(maxsize=8192)
        def target_edges(full: str) -> frozenset[tuple[int, str]]:
            return frozenset((pos, full[pos:pos + 2])
                             for pos in range(self.L - 1))

        def check(combo: tuple[Target, ...]) -> None:
            nonlocal checked, budget_pruned, best, best_effort
            checked += 1
            if len({t.full for t in combo}) != len(combo):
                return
            required = set().union(*(target_edges(t.full) for t in combo))
            if (any(sum(pos == p for p, _ in required) > self.cfg.branch_factor * self.cfg.top_k
                    for pos in range(self.L - 1))
                    or (self.pk_stage_caps is None and
                        sum(self.pair_lower[p][edge] for p, edge in required)
                        > self.optimistic_b_budget)):
                budget_pruned += 1
                return
            if self.pk_stage_caps is None:
                expectations = self.expectations
                additions = position_topups(
                    expectations, tuple(t.full for t in combo), self.cfg, self.n_b)
            else:
                b_detail: dict[str, Any] = {}
                result = self.pk_b_topups(tuple(t.full for t in combo), b_detail)
                if result is None:
                    budget_pruned += 1
                    return
                additions, expectations = result
            if additions is None:
                return
            b_cost = sum(sum(v.values()) for v in additions.values())
            if self.pk_stage_caps is not None and b_cost > self.pk_stage_caps["B"]:
                budget_pruned += 1
                return
            if self.pk_stage_caps is None and b_cost + self.d_base_reserve + safe > remaining:
                budget_pruned += 1
                return
            graph = apply_expected_pair_topups(expectations, additions, self.L,
                                               self.cfg.branch_factor * self.cfg.top_k)
            if any(any(t.full[p:p + 2] not in graph[p]
                       for p in range(self.L - 1)) for t in combo):
                return
            targets = {t.cls: t for t in combo}
            future = self.future_cost(graph, targets)
            if future is None:
                return
            total = b_cost + future[0] + (safe if self.pk_stage_caps is None else 0)
            if self.pk_stage_caps is None and total > remaining:
                budget_pruned += 1
                return
            full_score = sum(t.full_deviation for t in combo)
            d_predicted = future[2].identities
            pk_shortfall = 0
            pk_demands: dict[str, Any] = {}
            if self.pk_stage_caps is not None:
                pk_demands = {
                    "B": b_detail,
                    "C": [{"depth": depth, "raw_required": raw, "capacity": cap}
                          for depth, raw, cap in future[2].pk_c_raw_requirements],
                    "D": {"raw_required": future[2].initial_required,
                          "capacity": self.pk_stage_caps["D"]}}
                pk_shortfall = max(0, b_detail["raw_required"] - b_detail["capacity"])
                pk_shortfall += sum(max(0, raw - cap) for _, raw, cap
                                    in future[2].pk_c_raw_requirements)
                pk_shortfall += max(0, (future[2].initial_required or 0)
                                    - self.pk_stage_caps["D"])
            candidate = JointPlan(
                targets=targets, b_votes=additions, predicted_cost=total,
                reason=("PK fixed slots; theoretical demand fits each stage/layer"
                        if self.pk_stage_caps is not None and pk_shortfall == 0 else
                        "PK fixed slots; theoretical demand exceeds a stage/layer"
                        if self.pk_stage_caps is not None else
                        "Predicted feasible under model-conditional bounds"),
                candidate_source="all_qualified_no_hard_cap",
                full_score=full_score, d_ratio=self.d_gamma,
                d_ratio_reserve=self.d_base_reserve,
                d_locked=max(self.d_base_reserve, d_predicted),
                d_predicted=d_predicted,
                pk_budget_sufficient=(pk_shortfall == 0 if self.pk_stage_caps is not None else None),
                pk_total_shortfall=pk_shortfall, pk_stage_demands=pk_demands)
            incumbent = (best if pk_shortfall == 0 or self.pk_stage_caps is None
                         else best_effort)
            tie = (total, -min(t.margin for t in combo),
                   -min(float(t.bottleneck_support_ratio or 0.0) for t in combo),
                   tuple(t.full for t in combo))
            best_tie = ((incumbent.predicted_cost or float("inf")),
                        -min(t.margin for t in incumbent.targets.values()),
                        -min(float(t.bottleneck_support_ratio or 0.0)
                             for t in incumbent.targets.values()),
                        tuple(t.full for _, t in sorted(incumbent.targets.items()))) if incumbent else ()
            if (incumbent is None or len(candidate.targets) > len(incumbent.targets)
                    or (len(candidate.targets) == len(incumbent.targets) and
                        (pk_shortfall < incumbent.pk_total_shortfall
                         or (pk_shortfall == incumbent.pk_total_shortfall and
                             (full_score > (incumbent.full_score or 0.0) + 1e-8
                              or (abs(full_score - (incumbent.full_score or 0.0)) <= 1e-8
                                  and tie < best_tie)))))):
                if self.pk_stage_caps is not None and pk_shortfall:
                    best_effort = candidate
                else:
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
                        nonlocal bound_pruned, interrupted, tail_upper_bound_skips
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
                        for index, target in enumerate(pools[depth]):
                            
                            
                            
                            
                            
                            
                            if (best is not None and len(best.targets) == size and
                                    score + target.full_deviation + max_tail[depth + 1]
                                    < (best.full_score or 0.0) - 1e-8):
                                skipped = len(pools[depth]) - index
                                bound_pruned += skipped
                                tail_upper_bound_skips += skipped
                                break
                            if any(t.full == target.full for t in chosen):
                                continue
                            edges = target_edges(target.full)
                            union = used_edges | edges
                            added = union - used_edges
                            new_lb = lb + sum(self.pair_lower[j][e] for j, e in added)
                            if self.pk_stage_caps is None and new_lb > self.optimistic_b_budget:
                                bound_pruned += 1
                                continue
                            if any(sum(p == j for p, _ in union) > self.cfg.branch_factor * self.cfg.top_k
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

        if best is None and best_effort is not None:
            best = best_effort
        if best is None and self.pk_stage_caps is not None:
            fallback_targets: dict[int, Target] = {}
            occupied: set[str] = set()
            for cls in available_classes:
                candidate = next((t for t in class_pools[cls]
                                  if t.full not in occupied), None)
                if candidate is not None:
                    fallback_targets[cls] = candidate
                    occupied.add(candidate.full)
            if fallback_targets:
                try:
                    guessed = (self.pk_b_topups(tuple(t.full for t in fallback_targets.values()))
                               if not interrupted else None)
                except TimeoutError:
                    guessed = None
                    interrupted = True
                best = JointPlan(targets=fallback_targets,
                                 b_votes=guessed[0] if guessed else {},
                                 reason="PK fixed-capacity best effort; no sufficient joint forecast",
                                 candidate_source="qualified_fixed_capacity_fallback",
                                 full_score=sum(t.full_deviation for t in fallback_targets.values()),
                                 d_ratio=self.d_gamma, d_ratio_reserve=self.d_base_reserve,
                                 d_locked=self.pk_stage_caps["D"],
                                 d_predicted=self.pk_stage_caps["D"])
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
            "tail_upper_bound_skipped_candidates": tail_upper_bound_skips,
            "target_edge_cache_hits": target_edges.cache_info().hits,
            "budget_pruned_combinations": budget_pruned,
            "qualified_pool_sizes": {str(c): len(self.qualified_pools.get(c, ()))
                                     for c in sorted(self.references)},
            "partial_class_subsets_allowed": True,
            "search_timeout_seconds": self.cfg.search_timeout_seconds,
            "pk_fixed_full_replacement": self.pk_stage_caps is not None,
            "no_fixed_candidate_or_combination_caps": True}
        self._finish_search("joint_initial", started)
        return result

    def _reselect_exact_group(self, targets: dict[int, Target], missing: set[int],
                              depth: int, survivors: list[str],
                              graph: dict[int, tuple[str, ...]], remaining: int,
                              events: list[dict[str, Any]]) -> dict[int, Target]:
        
        
        return self._reselect_exact_group_inner(
            targets, missing, depth, survivors, graph, remaining,
            events, self._active_search_deadline)

    def _reselect_exact_group_inner(self, targets: dict[int, Target],
                                    missing: set[int], depth: int,
                                    survivors: list[str],
                                    graph: dict[int, tuple[str, ...]],
                                    remaining: int, events: list[dict[str, Any]],
                                    deadline: float | None) -> dict[int, Target]:
        if not missing or not self.cfg.allow_dynamic_reselect:
            return targets
        frozen = {cls: target for cls, target in targets.items() if cls not in missing}
        missing_classes = sorted(missing)
        if deadline is not None and time.monotonic() >= deadline:
            events.append({"event": "reselect_failure", "depth": depth,
                           "classes": missing_classes,
                           "reason": "c_dynamic_time_budget_exhausted",
                           "search_limited": True, "search_complete": False})
            return frozen
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
                    try:
                        forecast = self.future_cost(
                            graph, proposed, depth + 1,
                            survivors if depth >= 0 else None)
                    except TimeoutError:
                        truncated = True
                        break
                    if forecast is None:
                        continue
                    safety = ceil(self.cfg.safety_share * floor(self.cfg.beta * len(self.data.train)))
                    if self.pk_stage_caps is None and forecast[0] + safety > remaining:
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
                         events: list[dict[str, Any]], *,
                         layer: int | None = None) -> dict[int, Target]:
        if not missing:
            return targets
        
        
        
        kind = "b_reselect" if layer is None else "c_dynamic"
        _, started = self._start_search(kind, layer=layer)
        try:
            return self._dynamic_reselect_inner(
                targets, missing, depth, survivors, graph, remaining, events)
        finally:
            self._finish_search(kind, started, layer=layer)

    def _dynamic_reselect_inner(self, targets: dict[int, Target],
                                missing: set[int], depth: int,
                                survivors: list[str],
                                graph: dict[int, tuple[str, ...]],
                                remaining: int, events: list[dict[str, Any]]
                                ) -> dict[int, Target]:
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
                if (self._active_search_deadline is not None and
                        time.monotonic() >= self._active_search_deadline):
                    all_complete = False
                    break
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
            if (self._active_search_deadline is not None and
                    time.monotonic() >= self._active_search_deadline):
                all_complete = False
                break
            
            
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
        deadline, started = self._start_search("c_budget", layer=current_depth)
        try:
            return self._budget_reselect_inner(
                targets, current_depth, parents, graph, remaining,
                events, deadline)
        finally:
            self._finish_search("c_budget", started, layer=current_depth)

    def _budget_reselect_inner(self, targets: dict[int, Target],
                               current_depth: int, parents: list[str],
                               graph: dict[int, tuple[str, ...]],
                               remaining: int, events: list[dict[str, Any]],
                               deadline: float | None) -> dict[int, Target]:
        if not targets or not self.cfg.allow_dynamic_reselect:
            return targets
        if deadline is not None and time.monotonic() >= deadline:
            events.append({"event": "c_budget_replan_failure",
                           "depth": current_depth, "remaining": remaining,
                           "reason": "c_budget_time_budget_exhausted",
                           "checked_class_subsets": 0,
                           "checked_combinations": 0,
                           "search_complete": False})
            return targets if self.pk_stage_caps is not None else {}
        safe = ceil(self.cfg.safety_share
                    * floor(self.cfg.beta * len(self.data.train)))
        classes = sorted(targets)
        best: dict[int, Target] | None = None
        best_key: tuple[Any, ...] | None = None
        best_forecast = 0
        best_pk_shortfall: int | None = None
        checked_subsets = checked_combos = 0
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
                    try:
                        forecast = self.future_cost(
                            graph, proposed, current_depth, parents)
                    except TimeoutError:
                        interrupted = True
                        break
                    if (forecast is None or
                            (self.pk_stage_caps is None and forecast[0] + safe > remaining)):
                        continue
                    original_count = sum(proposed[c].full == targets[c].full for c in proposed)
                    prefix_count = sum(proposed[c].prefix == targets[c].prefix
                                       for c in proposed)
                    pk_shortfall = (sum(max(0, raw - cap) for _, raw, cap
                                        in forecast[2].pk_c_raw_requirements)
                                    + max(0, (forecast[2].initial_required or 0)
                                          - self.pk_stage_caps["D"])
                                    if self.pk_stage_caps is not None else 0)
                    merit = (size, original_count, prefix_count,
                             sum(t.full_deviation for t in combo), -forecast[0],
                             tuple(t.full for t in combo))
                    key = ((-pk_shortfall,) + merit
                           if self.pk_stage_caps is not None else merit)
                    if best_key is None or key > best_key:
                        best, best_key = proposed, key
                        best_forecast = forecast[0]
                        best_pk_shortfall = pk_shortfall
                if interrupted:
                    break
            if interrupted or (best is not None and
                               (self.pk_stage_caps is None or best_pk_shortfall == 0)):
                break
        if best is not None:
            offset = int(self.pk_stage_caps is not None)
            events.append({"event": "c_budget_replan", "depth": current_depth,
                           "active_classes": sorted(best),
                           "disabled_classes": sorted(set(targets) - set(best)),
                           "checked_class_subsets": checked_subsets,
                           "checked_combinations": checked_combos,
                           "search_complete": not interrupted and self.pool_search_complete,
                           "preserved_original_full": best_key[1 + offset],
                           "preserved_original_prefix": best_key[2 + offset],
                           "complete_leaf_objective_J": best_key[3 + offset],
                           "pk_total_forecast_shortfall": best_pk_shortfall,
                           "planned_remaining_joint_cost": best_forecast,
                           "d_locked_minimum": self.d_locked})
            return best
        events.append({"event": "c_budget_replan_failure",
                       "depth": current_depth, "remaining": remaining,
                       "checked_class_subsets": checked_subsets,
                       "checked_combinations": checked_combos,
                       "search_complete": not interrupted and self.pool_search_complete})
        
        
        return targets if self.pk_stage_caps is not None else {}


def honest_oue_counts(data: Dataset, ids: list[int], candidates: tuple[str, ...],
                      cfg: Config, rng: np.random.Generator,
                      skip_users: set[int] | None = None) -> dict[tuple[str, int], int]:
    q = 1 / (exp(cfg.epsilon) + 1)
    p = 0.5
    K = cfg.top_k
    counts = {(s, c): 0 for s in candidates for c in range(1, K + 1)}
    if not candidates:
        return counts
    for user in ids:
        sequence, label = data.train[user], data.labels[user]
        closest = min(candidates, key=lambda s: trace_distance(sequence, s))
        for s in candidates:
            for c in range(1, K + 1):
                vote = int(rng.random() < (p if (s, c) == (closest, label) else q))
                if not skip_users or user not in skip_users:
                    counts[(s, c)] += vote
    return counts


def trace_label_greedy(counts: dict[tuple[str, int], int],
                       candidates: tuple[str, ...], K: int) -> list[str | None]:
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
    configure_symbol_size(cfg.symbol_size)
    if fixed_length is not None and (poisoning or not cfg.length_min <= fixed_length <= cfg.length_max):
        raise ValueError("fixed_length is allowed only for a clean counterfactual")
    if length_attack_plan is not None and not poisoning:
        raise ValueError("length_attack_plan is only allowed in a poisoning trial")
    if stress_prune_classes not in (0, 1, 2) or (stress_prune_classes and not poisoning):
        raise ValueError("Only poisoned validation runs may force 1/2-class pruning")
    data.validate()
    knowledge = (planning_knowledge if planning_knowledge is not None else
                 (sample_pk_knowledge(data, cfg, run_seed + 707)
                  if poisoning and cfg.attack_mode == "pk" else
                  sample_planning_knowledge(data, cfg, run_seed + 707)))
    if (length_attack_plan is not None and length_attack_plan.knowledge_fingerprint is not None
            and length_attack_plan.knowledge_fingerprint != knowledge.fingerprint):
        raise ValueError("The A length plan and B/C/D planner must share exactly "
                         "the same round-specific planning knowledge sample")
    N = len(data.train)
    M = floor(cfg.beta * N) if poisoning else 0
    pk = poisoning and cfg.attack_mode == "pk"
    controlled = set(knowledge.indices) if pk else set()
    if pk and len(controlled) != M:
        raise ValueError("PK planning knowledge must equal the fixed compromised identities")
    
    groups = distribute_groups(N, cfg, run_seed + 1, length=1)
    domain = tuple(range(cfg.length_min, cfg.length_max + 1))
    honest = {name: len(ids) for name, ids in groups.items()}
    fake_used = {name: 0 for name in "ABCD"}
    events: list[dict[str, Any]] = []
    remaining = M
    planner = Planner(data, cfg, knowledge=knowledge)

    
    rng_a = np.random.default_rng(run_seed + 101)
    a_counts: Counter = Counter()
    a_cap = sum(u in controlled for u in groups["A"]) if pk else M
    known_lengths = {uid: min(cfg.length_max, max(cfg.length_min, len(seq)))
                     for uid, seq in zip(knowledge.indices, knowledge.train)} if pk else {}
    pk_a_slots = (tuple((slot, user, known_lengths.get(user))
                        for slot, user in enumerate(groups["A"])) if pk else None)
    a_plan = (length_attack_plan if length_attack_plan is not None else
              (length_plan(data, cfg, M,
                           len(groups["A"]) - a_cap if pk else len(groups["A"]),
                           available=a_cap if pk else None, knowledge=knowledge,
                           pk_a_slots=pk_a_slots)
               if poisoning else LengthPlan()))
    if pk and len(a_plan.reports) < a_cap:
        fallback_length = (a_plan.target if a_plan.target is not None else
                           max(cfg.length_min, min(cfg.length_max, 2)))
        used = len(a_plan.reports)
        if cfg.length_mechanism == "grr":
            a_plan.reports.extend([fallback_length] * (a_cap - used))
        else:
            domain_g = floor(exp(cfg.epsilon)) + 1
            pools = sample_identity_seed_pools(cfg, a_cap)
            a_plan.reports.extend((seeds[0], bucket(seeds[0], fallback_length,
                                                       cfg.length_min, domain_g))
                                  for seeds in pools[used:])
        if a_plan.target is None:
            a_plan.strategy = "pk_fixed_unqualified_length_fallback"
    if pk and len(a_plan.reports) > a_cap:
        raise AssertionError("PK A plan exceeds controlled A users")
    a_replaced = (set(list(u for u in groups["A"] if u in controlled)
                      [:len(a_plan.reports)]) if pk else set())
    if cfg.length_mechanism == "olh":
        for slot, user in enumerate(groups["A"]):
            value = min(cfg.length_max, max(cfg.length_min, len(data.train[user])))
            report = olh_report(value, slot, cfg.epsilon, domain, rng_a)
            if user not in a_replaced:
                a_counts.update(supported_lengths(report, domain, cfg.epsilon))
    else:
        for user in groups["A"]:
            value = min(cfg.length_max, max(cfg.length_min, len(data.train[user])))
            vote = randomized_response(value, domain, cfg.epsilon, rng_a)
            if user not in a_replaced:
                a_counts.update([vote])
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
    
    
    
    protocol_failure = ("symbols_B_requires_L_at_least_2" if L < 2 else None)
    if protocol_failure is not None:
        events.append({"event": "protocol_unavailable", "stage": "B",
                       "reason": protocol_failure, "actual_length": L,
                       "length_changed": False})
    groups = distribute_groups(N, cfg, run_seed + 1, length=L)
    honest = {name: len(ids) for name, ids in groups.items()}
    c_groups = split_users(groups["C"], L)
    pk_caps = ({"A": a_cap, "B": sum(u in controlled for u in groups["B"]),
                "C": [sum(u in controlled for u in group) for group in c_groups],
                "C_honest": [sum(u not in controlled for u in group)
                             for group in c_groups],
                "D": sum(u in controlled for u in groups["D"])} if pk else None)
    pk_stage_users = ({"B": [u for u in groups["B"] if u in controlled],
                       "C": [[u for u in layer if u in controlled]
                             for layer in c_groups],
                       "D": [u for u in groups["D"] if u in controlled]}
                      if pk else None)

    
    
    planner.setup(L, len(groups["B"]), len(groups["C"]), len(groups["D"]),
                  fake_total=M if poisoning else None,
                  fake_a=fake_used["A"], honest_a=len(groups["A"]),
                  build_candidates=poisoning and 2 <= L < cfg.length_max and len(planner.references) >= 2,
                  pk_stage_caps=pk_caps, pk_stage_users=pk_stage_users)
    joint = planner.joint_initial(remaining) if poisoning and 2 <= L < cfg.length_max else JointPlan(
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
    b_needed = sum(sum(v.values()) for v in joint.b_votes.values())
    if pk and protocol_failure is None:
        cap = pk_caps["B"]
        chosen_pair = (next(iter(joint.targets.values())).full[:2]
                       if joint.targets else PAIRS[-1])
        flat = {(pos, pair): amount for pos, pairs in joint.b_votes.items()
                for pair, amount in pairs.items()}
        allocated = pk_fixed_votes(flat, cap, (0, chosen_pair))
        joint.b_votes = {}
        for (pos, pair), amount in allocated.items():
            joint.b_votes.setdefault(pos, {})[pair] = amount
        b_needed = cap
        if not joint.targets:
            events.append({"event": "pk_b_unqualified_best_effort", "replaced": cap})
    b_feasible = (pk or b_needed + planner.d_locked <= remaining)
    b_replaced = (set(list(u for u in groups["B"] if u in controlled)[:b_needed])
                  if pk and b_feasible else set())
    if protocol_failure is None:
        for user in groups["B"]:
            original = data.train[user]
            s = original[:L].ljust(L, original[-1])
            pos = int(rng_b.integers(L - 1))
            reported = randomized_response(s[pos:pos + 2], PAIRS, cfg.epsilon, rng_b)
            if user not in b_replaced:
                b_counts[pos][reported] += 1
    if b_feasible:
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
    b_graph = {pos: tuple(top_names(dict(b_counts[pos]), cfg.branch_factor * cfg.top_k))
               for pos in range(max(0, L - 1))}
    events.append({"event": "b_pruning", "pairs": {str(k): list(v) for k, v in b_graph.items()},
                   "fake_used": fake_used["B"], "planned_cost": joint.predicted_cost})
    b_initial_survival = {str(cls): all(
        target.full[pos:pos + 2] in b_graph[pos] for pos in range(L - 1))
        for cls, target in joint.targets.items()}

    initial_targets = dict(joint.targets)
    targets = dict(joint.targets)
    if pk:
        planner.expectations = planner.pk_b_means(fake_used["B"])
    
    
    if targets and poisoning and 2 <= L < cfg.length_max:
        planner.after_b_graph(b_graph)
    if targets:
        missing = {cls for cls, t in targets.items()
                   if any(t.full[pos:pos + 2] not in b_graph[pos] for pos in range(L - 1))}
        if missing:
            targets = planner.dynamic_reselect(targets, missing, -1, [], b_graph,
                                               remaining, events)

    
    
    
    if any(len(group) != planner.c_per_level for group in c_groups):
        raise AssertionError("Actual Symbols C groups disagree with projected layer sizes")
    rng_c = np.random.default_rng(run_seed + 303)
    parents: list[str] = [""]
    c_states: list[dict[str, Any]] = []
    pk_c_layer_replacements = [0] * L
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
                                               remaining, events, layer=depth)
        protected = [t.full[:depth + 1] for t in targets.values()]
        
        
        
        projected = planner.baseline_em_means(candidates, depth) if targets else {}
        c_additions = {}
        reserved: int | None = None
        planning_failure = None
        future = None
        if targets:
            pk_detail: dict[str, Any] = {}
            if pk:
                suggested, projected = planner._measure_c_baseline(
                    depth, "pk_c_topups", lambda: planner.pk_c_topups(
                        candidates, protected, depth, pk_detail))
            else:
                suggested = supports_to_top_k(candidates, protected, projected,
                                              len(c_groups[depth]), cfg.branch_factor * cfg.top_k,
                                              cfg.score_z)
            if suggested is not None:
                forecast_nodes = top_nodes_in_creation_order(
                    nodes, {s: projected[s] + suggested.get(s, 0) for s in nodes},
                    cfg.branch_factor * cfg.top_k)
                future = planner.baseline_future_cost(
                    depth, b_graph, targets, depth + 1, forecast_nodes)
            if pk:
                raw = pk_detail.get("raw_required")
                
                future_shortfall = (None if future is None else
                                    sum(max(0, demand - cap) for _, demand, cap
                                        in future[2].pk_c_raw_requirements)
                                    + max(0, (future[2].initial_required or 0)
                                          - pk_caps["D"]))
                if (suggested is None or future is None or
                        (raw is not None and raw > pk_caps["C"][depth]) or
                        (future_shortfall is not None and future_shortfall > 0)):
                    old_targets = {str(c): t.full for c, t in targets.items()}
                    alternative = planner.budget_reselect(
                        targets, depth, parents, b_graph, remaining, events)
                    if alternative != targets:
                        targets = alternative
                        protected = [t.full[:depth + 1] for t in targets.values()]
                        pk_detail = {}
                        if targets:
                            suggested, projected = planner._measure_c_baseline(
                                depth, "pk_c_topups", lambda: planner.pk_c_topups(
                                    candidates, protected, depth, pk_detail))
                        else:
                            suggested = None
                        future = None
                        if suggested is not None:
                            forecast_nodes = top_nodes_in_creation_order(
                                nodes, {s: projected[s] + suggested.get(s, 0) for s in nodes},
                                cfg.branch_factor * cfg.top_k)
                            future = planner.baseline_future_cost(
                                depth, b_graph, targets, depth + 1, forecast_nodes)
                    events.append({"event": "pk_c_pre_submission_budget_check",
                                   "depth": depth, "raw_required_before": raw,
                                   "fixed_capacity": pk_caps["C"][depth],
                                   "future_forecast_available_before": future_shortfall is not None,
                                   "future_shortfall_before": future_shortfall,
                                   "targets_before": old_targets,
                                   "targets_after": {str(c): t.full for c, t in targets.items()},
                                   "raw_required_after": pk_detail.get("raw_required"),
                                   "fallback_to_full_quota": True})
            reserved = future[0] if future is not None else None
            if (pk and suggested is not None and
                    sum(suggested.values()) > pk_caps["C"][depth]):
                c_additions, planning_failure = {}, "pk_c_layer_capacity"
            elif pk:
                c_additions = suggested or {}
                planning_failure = None if suggested is not None else "pk_c_no_sufficient_plan"
            else:
                c_additions, planning_failure = guard_stage_votes(
                    suggested, reserved, remaining, ceil(cfg.safety_share * M))
            if planning_failure is not None and not pk:
                
                
                alternative = planner.budget_reselect(
                    targets, depth, parents, b_graph, remaining, events)
                if alternative != targets:
                    targets = alternative
                    protected = [t.full[:depth + 1] for t in targets.values()]
                    if pk:
                        suggested, projected = planner._measure_c_baseline(
                            depth, "pk_c_topups", lambda: planner.pk_c_topups(
                                candidates, protected, depth))
                    else:
                        suggested = supports_to_top_k(
                            candidates, protected, projected, len(c_groups[depth]),
                            cfg.branch_factor * cfg.top_k, cfg.score_z)
                    future = None
                    if suggested is not None:
                        forecast_nodes = top_nodes_in_creation_order(
                            nodes, {s: projected[s] + suggested.get(s, 0) for s in nodes},
                            cfg.branch_factor * cfg.top_k)
                        future = planner.baseline_future_cost(
                            depth, b_graph, targets, depth + 1, forecast_nodes)
                    reserved = future[0] if future is not None else None
                    if (pk and suggested is not None and
                            sum(suggested.values()) > pk_caps["C"][depth]):
                        c_additions, planning_failure = {}, "pk_c_layer_capacity"
                    else:
                        c_additions, planning_failure = guard_stage_votes(
                            suggested, reserved, remaining,
                            ceil(cfg.safety_share * M))
            if planning_failure is not None:
                events.append({"event": "c_plan_guard", "depth": depth,
                               "reason": planning_failure, "future_reserved": reserved,
                               "requested": sum(suggested.values()) if suggested else 0,
                               "remaining": remaining,
                               "protected": protected})
        if pk:
            cap = pk_caps["C"][depth]
            fallback = next((node for node in protected if node in candidates), candidates[-1])
            c_additions = pk_fixed_votes(c_additions, cap, fallback)
            if not targets and cap:
                events.append({"event": "pk_c_unqualified_best_effort", "depth": depth,
                               "replaced": cap})
        count = sum(c_additions.values())
        if pk and count > pk_caps["C"][depth]:
            events.append({"event": "pk_c_group_limit", "depth": depth,
                           "requested": count, "capacity": pk_caps["C"][depth]})
            c_additions = {}
            count = 0
        pk_c_layer_replacements[depth] = count if pk else 0
        c_replaced = (set(list(u for u in c_groups[depth] if u in controlled)[:count])
                      if pk else set())

        honest_counts = Counter()
        for user in c_groups[depth]:
            probs = em_probs(data.train[user], candidates, cfg.epsilon)
            voted = candidates[int(rng_c.choice(len(candidates), p=probs))]
            if user not in c_replaced:
                honest_counts[voted] += 1
        honest_counts.update(c_additions)
        if pk and sum(honest_counts.values()) != len(c_groups[depth]):
            raise AssertionError("PK C layer must replace, not append, reports")
        fake_used["C"] += count
        remaining -= count
        winners = top_nodes_in_creation_order(
            nodes, {s: honest_counts[s] for s in nodes}, cfg.branch_factor * cfg.top_k)
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
                         "received_reports": sum(honest_counts.values()),
                         "honest_users": len(c_groups[depth]),
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
                                               parents, b_graph, remaining, events,
                                               layer=depth)

    
    
    leaves = tuple(parents) if c_states and c_states[-1]["depth"] == L - 1 else ()
    d_additions: list[set[tuple[str, int]]] = []
    d_plan = DPlan(feasible=False)
    d_available_after_c = pk_caps["D"] if pk else remaining
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
    if pk and poisoning and len(leaves) >= cfg.top_k and targets:
        targets, d_plan = planner.pk_d_budget_reselect(leaves, targets, events)
    if poisoning and len(leaves) >= cfg.top_k and targets:
        if all(t.full in leaves for t in targets.values()):
            d_plan = (d_plan if pk else choose_oue_plan(
                          leaves, {c: t.full for c, t in targets.items()},
                          planner.d_means(leaves), cfg))
            if d_plan.feasible and d_plan.identities <= d_available_after_c:
                d_additions = pack_oue(d_plan, leaves, cfg.top_k)
                fake_used["D"] = len(d_additions)
                remaining -= fake_used["D"]
            else:
                events.append({"event": "d_budget_failure", "required": d_plan.identities,
                               "initial_required": d_plan.initial_required,
                               "fixed_replacement_count": (pk_caps["D"] if pk else None),
                               "reason": d_plan.failure_reason,
                               "stage_capacity": d_available_after_c,
                               "remaining": remaining, "feasible": d_plan.feasible})
        else:
            events.append({"event": "d_missing_target_leaf",
                           "remaining_targets": {str(c): t.full for c, t in targets.items()}})
    if pk and leaves and len(d_additions) != pk_caps["D"]:
        viable = next(((t.full, c) for c, t in sorted(targets.items())
                       if t.full in leaves), (leaves[-1], 1))
        d_additions = [{viable} for _ in range(pk_caps["D"])]
        fake_used["D"] = len(d_additions)
        remaining -= fake_used["D"]
        events.append({"event": "pk_d_unqualified_best_effort", "cell": viable,
                       "replaced": len(d_additions)})
    rng_d = np.random.default_rng(run_seed + 404)
    d_replaced = (set(list(u for u in groups["D"] if u in controlled)
                      [:len(d_additions)]) if pk else set())
    if pk:
        if (len(a_replaced) != fake_used["A"] or
                len(b_replaced) != fake_used["B"] or
                len(d_replaced) != fake_used["D"] or
                any(used > cap for used, cap in zip(
                    pk_c_layer_replacements, pk_caps["C"])) or
                pk_caps["A"] + pk_caps["B"] + sum(pk_caps["C"]) + pk_caps["D"] != M):
            raise AssertionError("PK replacement violated real-user stage/layer allocation")
        if (fake_used["A"] != pk_caps["A"] or
                fake_used["B"] != (pk_caps["B"] if L >= 2 else 0) or
                any(state["fake_used"] != pk_caps["C"][state["depth"]]
                    for state in c_states) or
                (bool(leaves) and fake_used["D"] != pk_caps["D"])):
            raise AssertionError("PK must replace every protocol-reportable controlled slot")
    pk_unreportable = ({"B": pk_caps["B"] - fake_used["B"],
                        "C_not_reached": sum(pk_caps["C"][len(c_states):]),
                        "D_no_candidates": (0 if leaves else pk_caps["D"])}
                       if pk else None)
    if pk and remaining != sum(pk_unreportable.values()):
        raise AssertionError("PK unused slots must be protocol-unreportable")
    d_counts = (honest_oue_counts(data, groups["D"], leaves, cfg, rng_d,
                                  skip_users=d_replaced)
                if pk else honest_oue_counts(data, groups["D"], leaves, cfg, rng_d))
    for bits in d_additions:
        for cell in bits:
            d_counts[cell] += 1
    representatives = trace_label_greedy(d_counts, leaves, cfg.top_k) if leaves else [None] * cfg.top_k
    complete_representatives = (len(representatives) == cfg.top_k and
                                all(isinstance(s, str) and bool(s)
                                    for s in representatives))
    if protocol_failure is None and not complete_representatives:
        protocol_failure = "symbols_classification_requires_nonempty_representatives"
        events.append({"event": "protocol_unavailable", "stage": "classification",
                       "reason": protocol_failure, "actual_length": L,
                       "nonempty_representatives": sum(
                           isinstance(s, str) and bool(s) for s in representatives)})
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
            "symbols_protocol_version": SYMBOLS_PROTOCOL_VERSION,
            "protocol_valid": protocol_failure is None,
            "protocol_failure": protocol_failure,
            "length": L, "natural_honest_length": natural_L,
            "counterfactual_fixed_length": fixed_length is not None,
            "length_plan": {"target": a_plan.target,
                                        "analytically_feasible": a_plan.feasible,
                                        "strategy": a_plan.strategy,
                                        "reference_length_lower_bound": a_plan.reference_min,
                                        "min_required": a_plan.min_required,
                                        "attempts": a_plan.attempts,
                                        "net_gains": a_plan.net_gains,
                                        "submitted_reports": a_report_details},
            "honest_groups": honest, "fake_total": M,
            "attack_model": cfg.attack_mode if poisoning else "clean",
            "pk_group_caps": pk_caps,
            "pk_replaced_reports": dict(fake_used) if pk else None,
            "pk_unreportable_slots": pk_unreportable,
            "pk_c_layer_replaced_reports": pk_c_layer_replacements if pk else None,
            "fixed_server_population": N,
            "server_report_counts": {
                "A": len(groups["A"]) + (0 if pk else fake_used["A"]),
                "B": sum(sum(v.values()) for v in b_counts.values()),
                "C_per_layer": [state["received_reports"] for state in c_states],
                "D": (len(groups["D"]) + (0 if pk else fake_used["D"]))
                     if leaves else 0},
            "honest_c_per_level": planner.c_per_level,
            "c_allocation_rule": "floor(0.70*post_AB_honest_population/L) per level",
            "planning_knowledge": knowledge.diagnostics(),
            "attack_planning_status": ("insufficient_class_knowledge"
                                       if pk and len(set(knowledge.labels)) < cfg.top_k
                                       else "executed"),
            "missing_knowledge_classes": (sorted(set(range(1, cfg.top_k + 1)) -
                                                set(knowledge.labels)) if pk else []),
            "pk_fixed_full_replacement": pk,
            "planning_search": planner.search_diagnostics(),
            "shape_distance_reference_source": "full_honest_population_posthoc_only",
            "fake_used": fake_used, "fake_remaining": remaining,
            "references": {str(cls): [{"shape": s, "weight": w}
                                       for s, w in planner.references.get(cls, ())]
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
            "complete_representatives": complete_representatives,
            "missing_representative_count": sum(
                not isinstance(s, str) or not s for s in representatives),
            "accuracy": (accuracy(data, representatives)
                         if evaluate_test and protocol_failure is None else None),
            "recall_by_class": (recall_by_class(data, representatives)
                                if evaluate_test and protocol_failure is None else None),
            "test_evaluation_performed": evaluate_test and protocol_failure is None,
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
                "objective": "maximize_sum_of_complete_leaf_weighted_sax_dtw_distances",
                "gamma_d": joint.d_ratio, "d_ratio_reserve": joint.d_ratio_reserve,
                "d_locked_minimum": joint.d_locked,
                "d_predicted_identity_cost": joint.d_predicted,
                "pk_budget_sufficient": joint.pk_budget_sufficient,
                "pk_total_theoretical_shortfall": joint.pk_total_shortfall,
                "pk_stage_theoretical_demands": joint.pk_stage_demands,
                "checked_combinations": joint.checked_combinations,
                "total_combinations": joint.total_combinations,
                "search_complete": joint.search_complete,
                "search_limited": joint.search_limited,
                "search_diagnostics": joint.search_diagnostics,
                "candidate_source": joint.candidate_source}}


CHECKPOINT_FORMAT_VERSION = 1


def _dataset_fingerprint(data: Dataset) -> str:
    digest = sha256()
    for title, sequences, labels in (("train", data.train, data.labels),
                                     ("test", data.test, data.test_labels)):
        digest.update(f"{title}:{len(sequences)}\n".encode("ascii"))
        for shape, label in zip(sequences, labels):
            digest.update(f"{label}:{len(shape)}:{shape}\n".encode("ascii"))
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
    signature_source = {"format_version": CHECKPOINT_FORMAT_VERSION,
                        "symbols_protocol_version": SYMBOLS_PROTOCOL_VERSION,
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
    configure_symbol_size(cfg.symbol_size)
    if runs < 1:
        raise ValueError("runs must be positive")
    if cfg.attack_mode not in ("fk", "pk"):
        raise ValueError("attack_mode must be 'fk' or 'pk'")
    if cfg.attack_mode == "pk" and (cfg.planning_sample != 0 or
                                    cfg.planning_fraction is not None or
                                    not 0 < cfg.beta < 1):
        raise ValueError("PK knowledge is restricted to the fixed compromised set; "
                         "do not additionally set planning_sample/planning_fraction")
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
        
        
        
        knowledge = (sample_pk_knowledge(data, cfg, trial_seed + 707)
                     if cfg.attack_mode == "pk" else
                     sample_planning_knowledge(data, cfg, trial_seed + 707))
        
        
        trial_cfg = replace(cfg, seed_strategy_seed=(
            cfg.seed_strategy_seed + trial_index * 10009))
        length_groups = distribute_groups(len(data.train), trial_cfg,
                                          trial_seed + 1, length=1)
        pk_a_cap = (sum(u in set(knowledge.indices) for u in length_groups["A"])
                    if cfg.attack_mode == "pk" else None)
        known_lengths = {uid: min(cfg.length_max, max(cfg.length_min, len(seq)))
                         for uid, seq in zip(knowledge.indices, knowledge.train)}
        pk_a_slots = (tuple((slot, uid, known_lengths.get(uid))
                            for slot, uid in enumerate(length_groups["A"]))
                      if cfg.attack_mode == "pk" else None)
        candidate_plan = length_plan(
            data, trial_cfg, floor(cfg.beta * len(data.train)),
            len(length_groups["A"]) - pk_a_cap if pk_a_cap is not None
            else len(length_groups["A"]),
            available=pk_a_cap, knowledge=knowledge, pk_a_slots=pk_a_slots)
        a_selection = {
            "status": ("analytically_feasible" if candidate_plan.feasible else
                       candidate_plan.strategy if candidate_plan.target is not None
                       else "no_analytically_feasible_target"),
            "selected_length": candidate_plan.target,
            "analytically_feasible": candidate_plan.feasible,
            "selection_strategy": candidate_plan.strategy,
            "reference_length_lower_bound": candidate_plan.reference_min,
            "malicious_identities": (pk_a_cap if cfg.attack_mode == "pk" else
                                     len(candidate_plan.reports)),
            "required_net_support": candidate_plan.bounds,
            "achieved_net_support": candidate_plan.net_gains,
            "search_attempts": candidate_plan.attempts,
            "knowledge_fingerprint": knowledge.fingerprint,
            "missing_knowledge_classes": (sorted(set(range(1, cfg.top_k + 1)) -
                                                 set(knowledge.labels))
                                          if cfg.attack_mode == "pk" else []),
            "seed_pool_random_seed": trial_cfg.seed_strategy_seed,
            "seed_pool_policy": "not_used_for_symbols_grr",
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
        failed_protocols = {
            name: report["protocol_failure"]
            for name, report in (("clean", clean), ("attack", attack),
                                 ("same_length_clean", same_length_clean))
            if not report["protocol_valid"]
        }
        paired_complete = not failed_protocols
        record = {"trial_index": trial_index, "trial_seed": trial_seed,
                  "status": ("completed" if paired_complete else "failed_protocol"),
                  "failure_reasons": failed_protocols,
                  "clean": clean, "same_length_clean": same_length_clean,
                  "attack": attack, "a_analytical_selection": a_selection,
                  "post_length_accuracy_change": (
                      attack["accuracy"] - same_length_clean["accuracy"]
                      if paired_complete else None),
                  "accuracy_change": (attack["accuracy"] - clean["accuracy"]
                                      if paired_complete else None)}
        
        
        if saved_trial is not None:
            _write_json_atomic(saved_trial, {"signature": signature,
                                             "trial_index": trial_index,
                                             "trial_seed": trial_seed,
                                             "record": record})
        records.append(record)
        if not paired_complete:
            print(f"trial={index + 1}/{runs} global={trial_index} "
                  f"failed_protocol={failed_protocols} "
                  f"A={attack['length_plan']['target']}->{attack['length']} "
                  "checkpoint_saved; continuing", flush=True)
        else:
            print(f"trial={index + 1}/{runs} clean={clean['accuracy']:.4f} "
                  f"attack={attack['accuracy']:.4f} "
                  f"A={attack['length_plan']['target']}->{attack['length']} "
                  f"malicious={sum(attack['fake_used'].values())}/{attack['fake_total']}",
                  flush=True)
    
    
    
    completed = [x for x in records if x["status"] == "completed"]

    def mean_or_none(values: Iterable[float]) -> float | None:
        values = list(values)
        return float(np.mean(values)) if values else None

    def attack_mean_present(key: str) -> float | None:
        values = [x["attack"].get(key) for x in completed]
        measured = [float(value) for value in values if value is not None]
        return mean_or_none(measured)

    summary = {"runs": runs,
               "completed_paired_trials": len(completed),
               "failed_protocol_trials": runs - len(completed),
               "failed_trial_indices": [x["trial_index"] for x in records
                                        if x["status"] != "completed"],
               "failure_counts_by_report": dict(Counter(
                   f"{name}:{reason}" for x in records
                   for name, reason in x.get("failure_reasons", {}).items())),
               "pk_insufficient_class_knowledge_trials": sum(
                   x["attack"].get("attack_planning_status") ==
                   "insufficient_class_knowledge" for x in records),
               "mean_clean_accuracy": mean_or_none(x["clean"]["accuracy"] for x in completed),
               "mean_attack_accuracy": mean_or_none(x["attack"]["accuracy"] for x in completed),
               "mean_same_length_clean_accuracy": mean_or_none(
                   x["same_length_clean"]["accuracy"] for x in completed),
               "mean_accuracy_change": mean_or_none(x["accuracy_change"] for x in completed),
               "mean_post_length_accuracy_change": mean_or_none(
                   x["post_length_accuracy_change"] for x in completed),
               "length_target_hit_rate": float(np.mean([
                   x["attack"]["length_plan"]["target"] is not None and
                   x["attack"]["length_plan"]["target"] == x["attack"]["length"]
                   for x in records])),
               "all_shape_targets_hit_rate": mean_or_none(
                   x["attack"]["all_target_success"] for x in completed),
               
               
               "all_six_classes_hit_rate": mean_or_none(
                   x["attack"]["all_target_success"] for x in completed),
               "trials_with_initial_active_targets": sum(
                   x["attack"].get("initial_active_target_count", 0) > 0
                   for x in completed),
               "trials_with_final_active_targets": sum(
                   x["attack"].get("final_active_target_count", 0) > 0
                   for x in completed),
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
               "attack_no_output_rate": mean_or_none(
                   x["attack"]["no_output"] for x in completed),
               
               
               "attack_no_output_rate_all_trials": mean_or_none(
                   x["attack"]["no_output"] for x in records),
               "attack_incomplete_representatives_rate_all_trials": mean_or_none(
                   not x["attack"].get("complete_representatives", False)
                   for x in records),
               "classification_failure_trials": sum(
                   any(report.get("protocol_failure") ==
                       "symbols_classification_requires_nonempty_representatives"
                       for report in (x["clean"], x["attack"],
                                      x["same_length_clean"]))
                   for x in records)}
    return {"config": asdict(cfg), "summary": summary,
            "symbols_protocol_version": SYMBOLS_PROTOCOL_VERSION,
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
    p = argparse.ArgumentParser(description="PrivShape/Symbols classification four-stage poisoning experiment")
    p.add_argument("--data-dir", type=Path, help="Directory of data.txt, label.txt, test_data.txt, test_label.txt")
    p.add_argument("--demo", action="store_true", help="Run on a synthetic smoke-test dataset")
    p.add_argument("--demo-users", type=int, default=1200)
    p.add_argument("--epsilon", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=.20)
    p.add_argument("--attack-mode", choices=("fk", "pk"), default="fk",
                   help="fk adds fake reports; pk replaces reports of a beta "
                        "fraction of genuine training users")
    p.add_argument("--length-mechanism", choices=("grr",), default="grr",
                   help="Symbols A follows the original GRR protocol")
    p.add_argument("--symbol-size", type=int, default=6,
                   help="SAX alphabet size, from 3 to 10; uses symbols_distance.py")
    p.add_argument("--length-bound", choices=("normal", "cantelli"), default="normal",
                   help="A-stage model-conditional pairwise bound; Cantelli is "
                        "conservative only under the assumed honest-user "
                        "distribution and variance model, not a guarantee "
                        "of real-population attack success")
    p.add_argument("--length-delta", type=float, default=.05)
    p.add_argument("--seed-search", type=int, default=100,
                   help="Compatibility option; unused by Symbols' GRR A stage")
    p.add_argument("--seed-space", type=int, default=2**31,
                   help="Seed values are sampled independently from [0, seed-space)")
    p.add_argument("--reference-coverage", type=float, default=.70,
                   help="Within each class, cover at least this share of honest shapes")
    p.add_argument("--score-z", type=float, default=2.0,
                   help="B/C/D marginal planning margin; not a joint confidence level")
    p.add_argument("--planning-sample", type=int, default=0,
                   help="0 uses all honest training records unless --planning-fraction "
                        "is supplied; positive N is an explicit knowledge sample size")
    p.add_argument("--planning-fraction", type=float, default=None,
                   choices=(.10, .25, .50, 1.0),
                   help="Fraction of honest training users visible to the attacker "
                        "(0.10, 0.25, 0.50, 1.0); overrides default planning-sample=0")
    p.add_argument("--search-timeout-seconds", type=float, default=None,
                   help="Per-search uniform override; omit to choose independent "
                        "stage budgets identical to the 100-percent-knowledge profile; "
                        "explicit 0 disables all search deadlines")
    p.add_argument("--safety-share", type=float, default=.02,
                   help="Optional extra fake-identity buffer; use 0 for no buffer")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=20231025)
    p.add_argument("--seed-strategy-seed", type=int, default=20231025,
                   help="Compatibility field for checkpoint signatures; "
                        "Symbols GRR does not search a seed pool")
    p.add_argument("--trial-start", type=int, default=0,
                   help="First global trial index for parallel workers: assign "
                        "disjoint ranges and distinct --output files")
    p.add_argument("--resume", action="store_true",
                   help="Reuse compatible per-trial checkpoints; unfinished global "
                        "trial indices are recomputed")
    p.add_argument("--no-dynamic-reselect", action="store_true")
    p.add_argument("--output", type=Path, default=Path("symbols_attack_result.json"))
    args = p.parse_args()
    if not args.demo and args.data_dir is None:
        p.error("Provide --data-dir with your preprocessed Symbols files, or --demo")
    if (args.epsilon <= 0 or not 0 <= args.beta <= 1 or args.seed_search < 1
            or args.seed_space < args.seed_search
            or not 0 < args.reference_coverage <= 1
            or (args.search_timeout_seconds is not None and
                args.search_timeout_seconds < 0)
            or not 0 <= args.safety_share <= 1
            or not 0 < args.length_delta < 1 or args.score_z < 0
            or args.planning_sample < 0 or args.trial_start < 0
            or not 3 <= args.symbol_size <= 10
            or args.seed < 0 or args.seed_strategy_seed < 0):
        p.error("Invalid epsilon/beta/delta/seed-space/coverage/timeout/buffer")
    if args.planning_fraction is not None and args.planning_sample != 0:
        p.error("Use either --planning-fraction or positive --planning-sample, not both")
    if args.attack_mode == "pk" and (not 0 < args.beta < 1 or
                                     args.planning_sample != 0 or
                                     args.planning_fraction is not None):
        p.error("PK requires 0<beta<1; its only planning knowledge is the "
                "compromised users, so omit planning-sample/planning-fraction")
    if args.output.exists() and not args.resume:
        p.error(f"Output already exists: {args.output}; select another --output "
                "or use --resume")
    checkpoint_dir = Path(str(args.output) + ".trials")
    if args.resume and args.output.exists() and not (checkpoint_dir / "manifest.json").exists():
        p.error("Cannot resume a legacy result without trial checkpoints; "
                "select a new --output")
    configure_symbol_size(args.symbol_size)
    data = demo_dataset(args.demo_users, seed=args.seed) if args.demo else load_dataset(args.data_dir)
    cfg = Config(epsilon=args.epsilon, beta=args.beta,
                 attack_mode=args.attack_mode,
                 length_mechanism=args.length_mechanism, symbol_size=args.symbol_size,
                 seed_search=args.seed_search,
                 seed_space=args.seed_space,
                 seed_strategy_seed=args.seed_strategy_seed,
                 reference_coverage=args.reference_coverage,
                 probability_bound=args.length_bound,
                 length_success_delta=args.length_delta,
                 score_z=args.score_z, planning_sample=args.planning_sample,
                 planning_fraction=args.planning_fraction,
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
