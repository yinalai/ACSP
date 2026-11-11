# RRA: Random Report Attack

## Python files

| File | Purpose |
| --- | --- |
| `rra_shared.py` | Implements random malicious reports, identity allocation, paired experiments, and result saving for both datasets. |
| `Trace/rra_trace.py` | Starts the Trace RRA experiment. |
| `Trace/attack_trace.py` | Provides the Trace protocol simulator and classification functions used by RRA. |
| `Symbols/rra_symbols.py` | Starts the Symbols RRA experiment. |
| `Symbols/attack_symbols.py` | Provides the Symbols protocol simulator and classification functions used by RRA. |
| `Symbols/symbols_distance.py` | Provides the Symbols symbol-distance and matching functions. |

## Run one trial

From the top-level `ACSP` directory:

```bash
python baselines/RRA/Trace/rra_trace.py --data-dir /path/to/Trace_SAX --epsilon 1 --beta 0.15 --runs 1 --output results/trace_rra_once.json
python baselines/RRA/Symbols/rra_symbols.py --data-dir /path/to/Symbols_SAX --epsilon 1 --beta 0.15 --runs 1 --output results/symbols_rra_once.json
```

Each command runs one trial and saves the class representatives, Accuracy, and attack results. Use the same prepared data format as the root README.
