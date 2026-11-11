# Defense experiments

## Python files

| File | Purpose |
| --- | --- |
| `defense_shared.py` | Implements shared Normalization, Diffstats, Rand(k), and defense-evaluation functions. |
| `Trace/attack_trace_surface.py` | Runs the Trace attack and clean controls with the selected defense, then saves classification and attack results. |
| `Trace/input_surface_shared.py` | Provides report-handling and paired-result utilities required by the Trace entrypoint. |
| `Symbols/attack_symbols_surface.py` | Runs the Symbols attack and clean controls with the selected defense, then saves classification and attack results. |
| `Symbols/input_surface_shared.py` | Provides report-handling and paired-result utilities required by the Symbols entrypoint. |
| `Symbols/symbols_distance.py` | Provides the Symbols symbol-distance and matching functions. |

## Run one trial

From the top-level `ACSP` directory:

```bash
python defenses/Trace/attack_trace_surface.py --data-dir /path/to/Trace_SAX --report-surface output --epsilon 1 --beta 0.15 --defense-mode normalization --runs 1 --output results/trace_normalization_once.json
python defenses/Symbols/attack_symbols_surface.py --data-dir /path/to/Symbols_SAX --report-surface output --epsilon 1 --beta 0.15 --defense-mode diffstats --runs 1 --output results/symbols_diffstats_once.json
```

Each command runs one trial. Choose `normalization`, `diffstats`, or `randk` with `--defense-mode`, and use a separate output filename for each experiment.
