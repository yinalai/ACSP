# GP: Greedy Poisoning

## Python files

| File | Purpose |
| --- | --- |
| `gp_shared.py` | Implements stage-local greedy report selection, paired experiments, and result saving for both datasets. |
| `rra_shared.py` | Supplies the identity-allocation helper imported by GP. |
| `Trace/gp_trace.py` | Starts the Trace GP experiment. |
| `Trace/attack_trace.py` | Provides the Trace protocol simulator and classification functions used by GP. |
| `Symbols/gp_symbols.py` | Starts the Symbols GP experiment. |
| `Symbols/attack_symbols.py` | Provides the Symbols protocol simulator and classification functions used by GP. |
| `Symbols/symbols_distance.py` | Provides the Symbols symbol-distance and matching functions. |

## Run one trial

From the top-level `ACSP` directory:

```bash
python baselines/GP/Trace/gp_trace.py --data-dir /path/to/Trace_SAX --epsilon 1 --beta 0.15 --knowledge-size 0 --runs 1 --output results/trace_gp_once.json
python baselines/GP/Symbols/gp_symbols.py --data-dir /path/to/Symbols_SAX --epsilon 1 --beta 0.15 --knowledge-size 0 --runs 1 --output results/symbols_gp_once.json
```

Each command runs one trial. `--knowledge-size 0` uses all training records for planning. Outputs include the class representatives, Accuracy, and attack results.
