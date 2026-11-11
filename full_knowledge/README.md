# Full-knowledge ACSP

## Python files

| File | Purpose |
| --- | --- |
| `Trace/attack_trace.py` | Runs the Trace ACSP attack, clean controls, target reselection, classification, and result saving. |
| `Trace/prepare_trace.py` | Converts existing augmented numeric Trace data and labels from `.npy` files into the four SAX input files. |
| `Symbols/attack_symbols.py` | Runs the Symbols ACSP attack, clean controls, target reselection, classification, and result saving. |
| `Symbols/symbols_distance.py` | Provides the symbol-distance tables and matching functions used by Symbols. |

## Run one trial

Run from the top-level `ACSP` directory:

```bash
python full_knowledge/Trace/attack_trace.py --data-dir /path/to/Trace_SAX --epsilon 1 --beta 0.15 --runs 1 --output results/trace_fk_once.json
python full_knowledge/Symbols/attack_symbols.py --data-dir /path/to/Symbols_SAX --epsilon 1 --beta 0.15 --planning-fraction 1.0 --runs 1 --output results/symbols_fk_once.json
```

Each command runs one trial for its dataset. Both examples use full knowledge.
