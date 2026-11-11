# RK

## Python files

| File | Purpose |
| --- | --- |
| `rk_reference_model.py` | Builds or loads the reference-shape and weight summary used by the RK planner. |
| `Trace/attack_trace_rk.py` | Runs the Trace RK attack, clean controls, target reselection, classification, and result saving. |
| `Symbols/attack_symbols_rk.py` | Runs the Symbols RK attack, clean controls, target reselection, classification, and result saving. |
| `Symbols/symbols_distance.py` | Provides the Symbols symbol-distance and matching functions. |

## Run one trial

From the top-level `ACSP` directory:

```bash
python attacks_with_limited_knowledge/RK/Trace/attack_trace_rk.py --data-dir /path/to/Trace_SAX --epsilon 1 --beta 0.10 --runs 1 --output results/trace_rk_once.json
python attacks_with_limited_knowledge/RK/Symbols/attack_symbols_rk.py --data-dir /path/to/Symbols_SAX --epsilon 1 --beta 0.10 --runs 1 --output results/symbols_rk_once.json
```

Each command runs one trial. Keep `rk_reference_model.py` above the dataset directories.
