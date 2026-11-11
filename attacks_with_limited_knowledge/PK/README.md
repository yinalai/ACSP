# PK

## Python files

| File | Purpose |
| --- | --- |
| `Trace/pk_trace.py` | Starts the Trace experiment in PK mode. |
| `Trace/attack_trace.py` | Implements Trace simulation, controlled-user knowledge, report replacement, target reselection, and evaluation. |
| `Symbols/pk_symbols.py` | Starts the Symbols experiment in PK mode. |
| `Symbols/attack_symbols.py` | Implements Symbols simulation, controlled-user knowledge, report replacement, target reselection, and evaluation. |
| `Symbols/symbols_distance.py` | Provides the Symbols symbol-distance and matching functions. |

## Run one trial

From the top-level `ACSP` directory:

```bash
python attacks_with_limited_knowledge/PK/Trace/pk_trace.py --data-dir /path/to/Trace_SAX --epsilon 1 --beta 0.10 --runs 1 --output results/trace_pk_once.json
python attacks_with_limited_knowledge/PK/Symbols/pk_symbols.py --data-dir /path/to/Symbols_SAX --epsilon 1 --beta 0.10 --runs 1 --output results/symbols_pk_once.json
```

Each command runs one trial. `--beta` is the fraction of controlled existing users.
