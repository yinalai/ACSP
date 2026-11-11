# Symbols PK ablation

## Python files

| File | Purpose |
| --- | --- |
| `Symbols/pk_symbols.py` | Starts the Symbols experiment in PK mode. |
| `Symbols/attack_symbols.py` | Implements the Symbols PK attack and evaluation, including the switch for disabling target reselection. |
| `Symbols/symbols_distance.py` | Provides the Symbols symbol-distance and matching functions. |

## Run one trial

From the top-level `ACSP` directory:

```bash
python ablation_study/PK/Symbols/pk_symbols.py --data-dir /path/to/Symbols_SAX --epsilon 1 --beta 0.10 --no-dynamic-reselect --runs 1 --output results/symbols_pk_no_reselection_once.json
```
