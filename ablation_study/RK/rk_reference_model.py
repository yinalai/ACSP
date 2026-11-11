"""Reference-only knowledge supplied to the PrivShape RK attack planner.

This module deliberately contains no API for querying the original population.
The experiment owner constructs or loads a public shape/weight summary once;
planning consumes only that summary and the public population size.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isclose
from pathlib import Path
from typing import Any, Iterable
import json


@dataclass(frozen=True)
class PublicPopulation:
    """Only the public size is available in planning, not genuine records."""

    size: int

    @property
    def train(self) -> range:
        
        return range(self.size)


@dataclass(frozen=True)
class RKKnowledge:
    """Selected class references, normalized *within* class, with uniform priors.

    ``train``/``labels`` exist solely for compatibility with candidate code;
    they contain one entry per *published reference*, not user records.
    They MUST NEVER be used as an empirical-uniform population sample.
    """

    references: dict[int, tuple[tuple[str, float], ...]]
    population_size: int
    coverage: float
    origin: str = "reference_summary"
    indices: tuple[int, ...] = ()
    seed: None = None

    @property
    def train(self) -> tuple[str, ...]:
        return tuple(s for items in self.references.values() for s, _ in items)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(k for k, items in self.references.items() for _ in items)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.public_summary(), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False).encode("utf8")
        return sha256(encoded).hexdigest()[:16]

    def weighted_references(self) -> Iterable[tuple[int, str, float]]:
        prior = 1.0 / len(self.references)
        for cls, items in self.references.items():
            for shape, weight in items:
                yield cls, shape, prior * weight

    def mixture(self) -> Counter[str]:
        weights: Counter[str] = Counter()
        for _, shape, weight in self.weighted_references():
            weights[shape] += weight
        return weights

    def length_probabilities(self, lower: int, upper: int) -> dict[int, float]:
        probs = {length: 0.0 for length in range(lower, upper + 1)}
        for _, shape, weight in self.weighted_references():
            probs[min(upper, max(lower, len(shape)))] += weight
        return probs

    def reference_lengths(self) -> dict[int, int]:
        return {cls: ceil(sum(weight * len(shape) for shape, weight in items))
                for cls, items in self.references.items()}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": "reference_only_shape_weight_summary",
            "origin": self.origin,
            "knowledge_model": "RK",
            "class_prior": "uniform_1_over_K_not_true_class_proportions",
            "known_genuine_records": 0,
            "sample_size": 0,
            "honest_population_size": self.population_size,
            "reference_coverage_threshold": self.coverage,
            "reference_shapes_per_class": {
                str(cls): len(items) for cls, items in self.references.items()},
            "indices_fingerprint": None,
            "reference_fingerprint": self.fingerprint,
            "shared_by": ["A_length", "B_position_pairs", "C_prefix_expected_votes",
                          "D_label_expected_votes"],
            "caution": "The censored weighted-reference model is not the real population",
        }

    def public_summary(self) -> dict[str, Any]:
        return {
            "schema": "privshape_rk_references_v1",
            "knowledge_model": "RK",
            "coverage_threshold": self.coverage,
            "num_classes": len(self.references),
            "classes": {
                str(cls): [{"shape": shape, "weight": weight}
                           for shape, weight in items]
                for cls, items in self.references.items()},
            "note": "Weights are conditional on retained references, not class shares; "
                    "no original frequencies or class counts are included.",
        }


def validate_rk(known: RKKnowledge, num_classes: int, alphabet: str) -> RKKnowledge:
    if set(known.references) != set(range(1, num_classes + 1)):
        raise ValueError("RK summary must include every class exactly once")
    if not 0 < known.coverage <= 1:
        raise ValueError("RK reference coverage threshold must be in (0,1]")
    for cls, items in known.references.items():
        if not items or not all(s and set(s) <= set(alphabet) and
                                all(a != b for a, b in zip(s, s[1:]))
                                for s, _ in items):
            raise ValueError(f"Invalid RK references for class {cls}")
        if len({shape for shape, _ in items}) != len(items):
            raise ValueError(f"Duplicate RK reference shape within class {cls}")
        if any(not (0.0 < w <= 1.0) for _, w in items):
            raise ValueError(f"Invalid RK reference weight for class {cls}")
        if not isclose(sum(w for _, w in items), 1.0, abs_tol=1e-8):
            raise ValueError(f"RK weights for class {cls} must sum to one")
    return known


def release_from_training(train: Iterable[str], labels: Iterable[int],
                          population_size: int, classes: int,
                          coverage: float, alphabet: str) -> RKKnowledge:
    """EXPERIMENT OWNER ONLY: project complete training data to public RK summary.

    Original counts never appear in ``RKKnowledge`` or the exported JSON.
    This routine must not be invoked by a planner on its own.
    """
    counts = {cls: Counter() for cls in range(1, classes + 1)}
    for shape, cls in zip(train, labels):
        counts[cls][shape] += 1
    refs: dict[int, tuple[tuple[str, float], ...]] = {}
    for cls, counter in counts.items():
        total = sum(counter.values())
        if total <= 0:
            raise ValueError(f"No training examples for RK class {cls}")
        retained: list[tuple[str, int]] = []
        covered = 0
        for shape, frequency in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            retained.append((shape, frequency))
            covered += frequency
            if covered / total >= coverage - 1e-12:
                break
        refs[cls] = tuple((shape, frequency / covered)
                          for shape, frequency in retained)
    return validate_rk(RKKnowledge(refs, population_size, coverage,
                                   origin="experiment_owner_release"), classes, alphabet)


def load_public_references(path: Path, population_size: int, classes: int,
                           coverage: float, alphabet: str) -> RKKnowledge:
    payload = json.loads(path.read_text(encoding="utf8"))
    if payload.get("schema") != "privshape_rk_references_v1":
        raise ValueError("Unsupported RK reference summary schema")
    if payload.get("num_classes") != classes or not isclose(
            float(payload.get("coverage_threshold", -1)), coverage, abs_tol=1e-9):
        raise ValueError("RK summary must match the dataset classes and coverage threshold")
    refs = {int(cls): tuple((str(item["shape"]), float(item["weight"]))
                            for item in items)
            for cls, items in payload["classes"].items()}
    return validate_rk(RKKnowledge(refs, population_size, coverage,
                                   origin="external_reference_file"), classes, alphabet)
