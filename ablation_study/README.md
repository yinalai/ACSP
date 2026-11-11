# Ablation study

| Folder | Contents |
| --- | --- |
| `RK/` | Trace RK with dynamic reselection disabled by the command below. |
| `PK/` | Symbols PK with dynamic reselection disabled by the command below. |

The subfolder READMEs describe the Python files.

## Run one trial

From the top-level `ACSP` directory:

```bash
python ablation_study/RK/Trace/attack_trace_rk.py --data-dir /path/to/Trace_SAX --epsilon 2 --beta 0.10 --no-dynamic-reselect --runs 1 --output results/trace_rk_no_reselection_once.json
python ablation_study/PK/Symbols/pk_symbols.py --data-dir /path/to/Symbols_SAX --epsilon 1 --beta 0.10 --no-dynamic-reselect --runs 1 --output results/symbols_pk_no_reselection_once.json
```

Each command runs one trial. `--no-dynamic-reselect` disables dynamic target reselection.
