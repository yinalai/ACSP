from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from math import exp, floor
from pathlib import Path
from typing import Any

import numpy as np


def _largest_remainder(total: int, weights: list[int]) -> list[int]:
    """Allocate all identities to disjoint protocol slots; stable tie breaking."""
    if total < 0 or any(weight < 0 for weight in weights):
        raise ValueError("Negative RRA identity budget or sampling weight")
    if not weights:
        if total:
            raise ValueError("No destination for positive RRA budget")
        return []
    weight_sum = sum(weights)
    if not weight_sum:
        if total:
            raise ValueError("No protocol stage has a sampling weight")
        return [0] * len(weights)
    raw = [total * weight / weight_sum for weight in weights]
    chosen = [floor(x) for x in raw]
    left = total - sum(chosen)
    priority = sorted(range(len(weights)), key=lambda i: (-(raw[i] - chosen[i]), i))
    for i in priority[:left]:
        chosen[i] += 1
    assert sum(chosen) == total
    return chosen


def malicious_quotas(M: int, N: int, before: dict[str, list[int]],
                     after: dict[str, list[int]], c_groups: list[list[int]]
                     ) -> dict[str, Any]:
    """Freeze A/B before L; distribute remaining users by post-L honest ratios.

    A and B honest group sizes do not depend on L. Deterministic nearest-
    integer rounding is applied there; the residual goes to each C layer and
    D using the largest-remainder method, so no fake identity is lost.
    """
    if N < 1 or M < 0:
        raise ValueError("Invalid RRA genuine or injected population")
    if len(before["A"]) != len(after["A"]) or len(before["B"]) != len(after["B"]):
        raise AssertionError("A/B genuine group sizes changed after the public L")
    ma = min(M, floor(M * len(before["A"]) / N + .5))
    mb = min(M - ma, floor(M * len(before["B"]) / N + .5))
    tail = _largest_remainder(M - ma - mb,
                              [len(layer) for layer in c_groups] + [len(after["D"])])
    result = {"A": ma, "B": mb, "C": tail[:-1], "D": tail[-1]}
    assert ma + mb + sum(result["C"]) + result["D"] == M
    return result


def rra_report(engine: Any, data: Any, cfg: Any, run_seed: int,
               *, symbols: bool, evaluate_test: bool = True) -> dict[str, Any]:
    """Run genuine A/B/C/D plus disjoint, uniformly random legal fake reports.

    No attacker target, planning sample, support estimator or future-cost
    oracle is used. Genuine report RNG streams remain coupled to clean runs.
    """
    if cfg.epsilon <= 0 or cfg.beta < 0 or run_seed < 0:
        raise ValueError("RRA requires epsilon>0, beta>=0, and nonnegative seed")
    if cfg.attack_mode != "rra":
        raise ValueError("Set cfg.attack_mode='rra' for this baseline")
    if symbols:
        engine.configure_symbol_size(cfg.symbol_size)
    data.validate()
    N = len(data.train)
    M = floor(cfg.beta * N)
    domain = tuple(range(cfg.length_min, cfg.length_max + 1))
    before = engine.distribute_groups(N, cfg, run_seed + 1, length=1)
    
    ma = min(M, floor(M * len(before["A"]) / N + .5))
    mb = min(M - ma, floor(M * len(before["B"]) / N + .5))

    
    
    honest_a_rng = np.random.default_rng(run_seed + 101)
    fake_a_rng = np.random.default_rng(run_seed + 1101)
    a_counts: Counter = Counter()
    a_fake_reports: list[Any] = []
    if cfg.length_mechanism == "olh":
        bins = floor(exp(cfg.epsilon)) + 1
        for slot, user in enumerate(before["A"]):
            value = max(cfg.length_min, min(cfg.length_max, len(data.train[user])))
            msg = engine.olh_report(value, slot, cfg.epsilon, domain, honest_a_rng)
            a_counts.update(engine.supported_lengths(msg, domain, cfg.epsilon))
        for _ in range(ma):
            seed = int(fake_a_rng.integers(cfg.seed_space))
            bucket = int(fake_a_rng.integers(bins))
            msg = (seed, bucket)
            a_fake_reports.append(msg)
            a_counts.update(engine.supported_lengths(msg, domain, cfg.epsilon))
    elif cfg.length_mechanism == "grr":
        for user in before["A"]:
            value = max(cfg.length_min, min(cfg.length_max, len(data.train[user])))
            a_counts[engine.randomized_response(value, domain,
                                                cfg.epsilon, honest_a_rng)] += 1
        for _ in range(ma):
            msg = domain[int(fake_a_rng.integers(len(domain)))]
            a_fake_reports.append(msg)
            a_counts[msg] += 1
    else:
        raise ValueError("Unsupported length reporting mechanism")
    L = engine.select_length_output(a_counts, domain)
    after = engine.distribute_groups(N, cfg, run_seed + 1, length=L)
    c_groups = engine.split_users(after["C"], L)
    quota = malicious_quotas(M, N, before, after, c_groups)
    assert (quota["A"], quota["B"]) == (ma, mb)
    fake_used = {name: 0 for name in "ABCD"}
    fake_used["A"] = ma
    unreportable = {"B_no_pairs": 0, "C_no_frontier": 0, "D_no_leaves": 0}
    events: list[dict[str, Any]] = [
        {"event": "rra_length_selection", "length": L, "fake_A": ma},
        {"event": "rra_group_allocation", "malicious_quotas": quota},
    ]
    protocol_failure = ("symbols_B_requires_L_at_least_2"
                        if symbols and L < 2 else None)
    if protocol_failure:
        events.append({"event": "protocol_unavailable", "stage": "B",
                       "reason": protocol_failure})

    
    
    honest_b_rng = np.random.default_rng(run_seed + 202)
    fake_b_rng = np.random.default_rng(run_seed + 1202)
    b_counts = {pos: Counter() for pos in range(max(0, L - 1))}
    if L >= 2 and protocol_failure is None:
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
            msg = engine.randomized_response(sequence[pos:pos + 2],
                                             engine.PAIRS, cfg.epsilon, honest_b_rng)
            b_counts[pos][msg] += 1
        for _ in range(mb):
            pos = int(fake_b_rng.integers(L - 1))
            pair = engine.PAIRS[int(fake_b_rng.integers(len(engine.PAIRS)))]
            b_counts[pos][pair] += 1
        fake_used["B"] = mb
    else:
        unreportable["B_no_pairs"] = mb
    keep = cfg.branch_factor * cfg.top_k if symbols else 2 * cfg.top_k
    b_graph = {pos: tuple(engine.top_names(dict(b_counts[pos]), keep))
               for pos in range(max(0, L - 1))}

    
    
    honest_c_rng = np.random.default_rng(run_seed + 303)
    fake_c_rng = np.random.default_rng(run_seed + 1303)
    parents: list[str] = [""]
    c_states: list[dict[str, Any]] = []
    for depth in range(L):
        nodes = engine.expand_paths(parents, b_graph, depth)
        if not nodes:
            unreportable["C_no_frontier"] = sum(quota["C"][depth:])
            events.append({"event": "rra_c_no_frontier", "depth": depth,
                           "unreportable_fake_slots": unreportable["C_no_frontier"]})
            parents = []
            break
        candidates = tuple(nodes)
        votes = Counter()
        for user in c_groups[depth]:
            probs = engine.em_probs(data.train[user], candidates, cfg.epsilon)
            chosen = candidates[int(honest_c_rng.choice(len(candidates), p=probs))]
            votes[chosen] += 1
        for _ in range(quota["C"][depth]):
            votes[candidates[int(fake_c_rng.integers(len(candidates)))]] += 1
        fake_used["C"] += quota["C"][depth]
        if sum(votes.values()) != len(c_groups[depth]) + quota["C"][depth]:
            raise AssertionError("RRA must append, not replace, C reports")
        survivors = engine.top_nodes_in_creation_order(
            nodes, {candidate: votes[candidate] for candidate in nodes}, keep)
        c_states.append({"depth": depth, "candidates": list(candidates),
                         "survivors": survivors,
                         "honest_reports": len(c_groups[depth]),
                         "malicious_reports": quota["C"][depth],
                         "received_reports": sum(votes.values())})
        parents = survivors

    
    
    
    leaves = tuple(parents) if c_states and c_states[-1]["depth"] == L - 1 else ()
    honest_d_rng = np.random.default_rng(run_seed + 404)
    fake_d_rng = np.random.default_rng(run_seed + 1404)
    d_counts = engine.honest_oue_counts(data, after["D"], leaves, cfg, honest_d_rng)
    d_vectors: list[list[tuple[str, int]]] = []
    if leaves:
        cells = tuple((shape, cls) for shape in leaves
                      for cls in range(1, cfg.top_k + 1))
        max_ones = min(len(cells), engine.oue_ones_cap(
            cfg.top_k, len(leaves), cfg.epsilon))
        for _ in range(quota["D"]):
            ones = int(fake_d_rng.integers(1, max_ones + 1))
            indices = fake_d_rng.choice(len(cells), size=ones, replace=False)
            bits = [cells[int(i)] for i in indices]
            d_vectors.append(bits)
            for cell in bits:
                d_counts[cell] += 1
        fake_used["D"] = quota["D"]
    else:
        unreportable["D_no_leaves"] = quota["D"]
    representatives = (engine.trace_label_greedy(d_counts, leaves, cfg.top_k)
                       if leaves else [None] * cfg.top_k)
    complete_representatives = (len(representatives) == cfg.top_k and
                                all(isinstance(s, str) and bool(s)
                                    for s in representatives))
    if symbols:
        if protocol_failure is None and not complete_representatives:
            protocol_failure = "symbols_classification_requires_nonempty_representatives"
    elif not any(isinstance(s, str) and bool(s) for s in representatives):
        protocol_failure = "trace_classification_requires_nonempty_representatives"
    if protocol_failure is not None:
        events.append({"event": "protocol_unavailable", "stage": "classification",
                       "reason": protocol_failure})

    
    
    references, _ = engine.length_reference(data, cfg.reference_coverage)
    shape_distance = {
        str(cls): (engine.ref_dist(representatives[cls - 1], references[cls])
                   if representatives[cls - 1] is not None else None)
        for cls in range(1, cfg.top_k + 1)
    }
    normalized_distance = {
        str(cls): (engine.ref_dist(representatives[cls - 1], references[cls], True)
                   if representatives[cls - 1] is not None else None)
        for cls in range(1, cfg.top_k + 1)
    }
    unused = M - sum(fake_used.values())
    if unused != sum(unreportable.values()):
        raise AssertionError("All protocol-reportable RRA identities must report")
    if fake_used["A"] != quota["A"] or fake_used["B"] + unreportable["B_no_pairs"] != quota["B"]:
        raise AssertionError("RRA A/B injected-stage accounting mismatch")
    if fake_used["C"] + unreportable["C_no_frontier"] != sum(quota["C"]):
        raise AssertionError("RRA C per-layer accounting mismatch")
    if fake_used["D"] + unreportable["D_no_leaves"] != quota["D"]:
        raise AssertionError("RRA D injected-stage accounting mismatch")
    server_counts = {"A": len(after["A"]) + fake_used["A"],
                     "B": sum(sum(v.values()) for v in b_counts.values()),
                     "C_per_layer": [state["received_reports"] for state in c_states],
                     "D": len(after["D"]) + fake_used["D"] if leaves else 0}
    report = {
        "seed": run_seed, "attacked": True, "attack_model": "rra_injected_users",
        "epsilon": cfg.epsilon, "beta": cfg.beta, "length": L,
        "protocol_valid": protocol_failure is None,
        "protocol_failure": protocol_failure,
        "fixed_server_population": N + M,
        "genuine_population": N, "fake_total": M,
        "honest_groups": {name: len(users) for name, users in after.items()},
        "malicious_group_quotas": quota,
        "fake_used": fake_used, "fake_remaining": unused,
        "rra_unreportable_slots": unreportable,
        "rra_length_reports": ([(list(msg) if isinstance(msg, tuple) else msg)
                                for msg in a_fake_reports]),
        "server_report_counts": server_counts,
        "b_kept": {str(pos): list(edges) for pos, edges in b_graph.items()},
        "c_levels": c_states,
        "d_ones_cap": (engine.oue_ones_cap(cfg.top_k, len(leaves), cfg.epsilon)
                       if leaves else 0),
        "d_actual_ones_per_vector": [len(bits) for bits in d_vectors],
        "d_malicious_vectors": [[f"{shape}:{cls}" for shape, cls in bits]
                                for bits in d_vectors],
        "representatives": representatives,
        "complete_representatives": complete_representatives,
        "accuracy": (engine.accuracy(data, representatives)
                     if evaluate_test and protocol_failure is None else None),
        "recall_by_class": (engine.recall_by_class(data, representatives)
                            if evaluate_test and protocol_failure is None else None),
        "shape_distance": shape_distance,
        "normalized_shape_distance": normalized_distance,
        "test_evaluation_performed": evaluate_test and protocol_failure is None,
        "events": events,
        "random_report_policy": {
            "A": "uniform_legal_olh_seed_and_bucket_or_grr_length",
            "B": "uniform_current_position_and_protocol_pair",
            "C": "uniform_current_public_candidate",
            "D": "uniform_sparsity_1_to_cap_then_uniform_distinct_oue_cells"},
    }
    if symbols:
        report["symbols_protocol_version"] = engine.SYMBOLS_PROTOCOL_VERSION
        report["honest_c_per_level"] = len(c_groups[0]) if c_groups else 0
        report["c_allocation_rule"] = "floor(0.70*post_AB_honest_population/L) per level"
    return report


def paired_rra_experiment(engine: Any, data: Any, cfg: Any, *, symbols: bool,
                          runs: int, seed: int, trial_start: int = 0,
                          checkpoint_dir: Path | None = None,
                          resume: bool = False) -> dict[str, Any]:
    """Evaluate clean versus RRA; the same-L diagnostic is never a gate.

    Each branch contributes to its own accuracy mean when it has a valid
    classification. Accuracy differences require only the *two* branches
    being compared, not a valid result from the unrelated third branch.
    """

    def has_accuracy(report: dict[str, Any]) -> bool:
        value = report.get("accuracy")
        return (report.get("protocol_valid") is True and
                isinstance(value, (int, float, np.integer, np.floating)) and
                bool(np.isfinite(value)))

    def finalize_record(record: dict[str, Any]) -> dict[str, Any]:
        """Recompute even for old checkpoints saved with the three-way gate."""
        names = ("clean", "attack", "same_length_clean")
        valid = {name: has_accuracy(record[name]) for name in names}
        main_pair = valid["clean"] and valid["attack"]
        post_length_pair = valid["attack"] and valid["same_length_clean"]
        record.update({
            "status": ("completed" if main_pair else
                       "partial_result" if any(valid.values()) else
                       "failed_protocol"),
            "valid_by_report": valid,
            "main_pair_valid": main_pair,
            "post_length_pair_valid": post_length_pair,
            "failure_reasons": {
                name: (record[name].get("protocol_failure") or "accuracy_unavailable")
                for name in names if not valid[name]
            },
            "accuracy_change": (record["attack"]["accuracy"] -
                                record["clean"]["accuracy"]
                                if main_pair else None),
            "post_length_accuracy_change": (
                record["attack"]["accuracy"] -
                record["same_length_clean"]["accuracy"]
                if post_length_pair else None),
        })
        return record

    if runs < 1 or trial_start < 0:
        raise ValueError("runs must be positive and trial_start nonnegative")
    if resume and checkpoint_dir is None:
        raise ValueError("resume requires checkpoint_dir")
    signature = (engine._prepare_trial_checkpoints(
        checkpoint_dir, data, cfg, seed, resume) if checkpoint_dir else None)
    records: list[dict[str, Any]] = []
    for offset in range(runs):
        index = trial_start + offset
        trial_seed = seed + index * 10007
        saved = (checkpoint_dir / f"trial_{index:08d}.json"
                 if checkpoint_dir is not None else None)
        if saved is not None and saved.exists():
            if not resume:
                raise ValueError(f"Existing trial checkpoint: {saved}; use resume")
            payload = json.loads(saved.read_text(encoding="utf8"))
            if (payload.get("signature") != signature or
                    payload.get("trial_index") != index or
                    payload.get("trial_seed") != trial_seed):
                raise ValueError(f"Incompatible RRA checkpoint: {saved}")
            records.append(finalize_record(payload["record"]))
            print(f"trial={offset + 1}/{runs} global={index} resumed", flush=True)
            continue
        clean = engine.trace_report(data, cfg, trial_seed, False)
        attack = rra_report(engine, data, cfg, trial_seed, symbols=symbols)
        same_length_clean = engine.trace_report(
            data, cfg, trial_seed, False, fixed_length=attack["length"])
        record = finalize_record({
            "trial_index": index, "trial_seed": trial_seed,
            "clean": clean, "same_length_clean": same_length_clean,
            "attack": attack})
        if saved is not None:
            engine._write_json_atomic(saved, {
                "signature": signature, "trial_index": index,
                "trial_seed": trial_seed, "record": record})
        records.append(record)
        print(f"trial={offset + 1}/{runs} global={index} "
              f"length={attack['length']} "
              f"injected={sum(attack['fake_used'].values())}/{attack['fake_total']} "
              f"status={record['status']}", flush=True)
    completed = [record for record in records if record["main_pair_valid"]]

    def average(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    summary = {
        "runs": runs, "completed_trials": len(completed),
        "incomplete_main_pair_trials": runs - len(completed),
        "partial_result_trials": sum(r["status"] == "partial_result"
                                     for r in records),
        "no_valid_branch_trials": sum(r["status"] == "failed_protocol"
                                      for r in records),
        
        
        "failed_protocol_trials": runs - len(completed),
        "valid_trials_by_report": {name: sum(r["valid_by_report"][name]
                                              for r in records)
                                   for name in ("clean", "attack", "same_length_clean")},
        "main_pair_valid_trials": len(completed),
        "post_length_pair_valid_trials": sum(r["post_length_pair_valid"]
                                              for r in records),
        "mean_clean_accuracy": average([r["clean"]["accuracy"] for r in records
                                        if r["valid_by_report"]["clean"]]),
        "mean_attack_accuracy": average([r["attack"]["accuracy"] for r in records
                                         if r["valid_by_report"]["attack"]]),
        "mean_same_length_clean_accuracy": average([
            r["same_length_clean"]["accuracy"] for r in records
            if r["valid_by_report"]["same_length_clean"]]),
        "mean_accuracy_change": average([r["accuracy_change"] for r in completed]),
        "mean_post_length_accuracy_change": average(
            [r["post_length_accuracy_change"] for r in records
             if r["post_length_pair_valid"]]),
        "mean_injected_reports": average([
            sum(r["attack"]["fake_used"].values()) for r in records]),
        "mean_protocol_unreportable_injected_users": average([
            r["attack"]["fake_remaining"] for r in records]),
    }
    return {"attack_model": "rra_injected_users", "config": asdict(cfg),
            "trial_start": trial_start, "summary": summary, "trials": records}


def main(engine: Any, *, symbols: bool) -> None:
    parser = argparse.ArgumentParser(
        description="PrivShape Random Report Attack: disjoint injected fake users")
    parser.add_argument("--data-dir", type=Path,
                        help="Directory containing data.txt, label.txt, test_data.txt, test_label.txt")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-users", type=int, default=1200)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=.10,
                        help="Injected user count floor(beta * number of honest users)")
    if symbols:
        parser.add_argument("--symbol-size", type=int, default=6)
    else:
        parser.add_argument("--length-mechanism", choices=("olh", "grr"), default="olh")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20231025)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(
        "rra_symbols_result.json" if symbols else "rra_trace_result.json"))
    args = parser.parse_args()
    if (not args.demo and args.data_dir is None) or (args.demo and args.data_dir is not None):
        parser.error("Specify exactly one of --demo or --data-dir")
    if args.demo_users < 1 or args.epsilon <= 0 or args.beta < 0 or args.seed < 0:
        parser.error("Invalid dataset size, epsilon, beta, or seed")
    if args.runs < 1 or args.trial_start < 0:
        parser.error("runs must be positive and trial-start nonnegative")
    if symbols and not 3 <= args.symbol_size <= 10:
        parser.error("symbol-size must be in [3,10]")
    if args.output.exists() and not args.resume:
        parser.error("Output exists; choose another path or --resume")
    directory = Path(str(args.output) + ".trials")
    if args.resume and args.output.exists() and not (directory / "manifest.json").exists():
        parser.error("Cannot resume a legacy result without a checkpoint manifest")
    cfg_kwargs = {"epsilon": args.epsilon, "beta": args.beta, "attack_mode": "rra"}
    if symbols:
        cfg_kwargs["symbol_size"] = args.symbol_size
    else:
        cfg_kwargs["length_mechanism"] = args.length_mechanism
    cfg = engine.Config(**cfg_kwargs)
    data = (engine.demo_dataset(args.demo_users, seed=args.seed) if args.demo
            else engine.load_dataset(args.data_dir))
    result = paired_rra_experiment(
        engine, data, cfg, symbols=symbols, runs=args.runs, seed=args.seed,
        trial_start=args.trial_start, checkpoint_dir=directory, resume=args.resume)
    engine._write_json_atomic(args.output, result)
    print(f"Wrote {args.output}; valid clean/RRA pairs="
          f"{result['summary']['main_pair_valid_trials']}/{args.runs}; "
          f"valid RRA accuracies="
          f"{result['summary']['valid_trials_by_report']['attack']}/{args.runs}",
          flush=True)
