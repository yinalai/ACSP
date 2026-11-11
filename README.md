# ACSP

## Folders

| Folder | Contents |
| --- | --- |
| `full_knowledge/` | ACSP for Trace and Symbols with full knowledge. |
| `attacks_with_limited_knowledge/` | PK and RK implementations for both datasets. |
| `ablation_study/` | Trace RK and Symbols PK with dynamic reselection disabled by the run command. |
| `defenses/` | Normalization, Diffstats, and Rand(k) defense experiments for both datasets. |
| `baselines/` | RRA and GP comparison baselines for both datasets. |

Each folder's README lists its Python files and run examples.

## Run one example

Use Python 3.10 or newer. Run all README commands from this `ACSP` directory.

```bash
python -m pip install -r requirements.txt
python full_knowledge/Trace/attack_trace.py --data-dir /path/to/Trace_SAX --epsilon 1 --beta 0.15 --runs 1 --output results/trace_fk_once.json
```

`--runs 1` runs one trial. The output JSON records the class representatives, Accuracy, and attack results.
