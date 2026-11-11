# Trace RK ablation

## Python files

| File | Purpose |
| --- | --- |
| `rk_reference_model.py` | Provides the reference-shape and weight knowledge model for RK. |
| `Trace/attack_trace_rk.py` | Implements the Trace RK attack and evaluation, including the switch for disabling target reselection. |

## Run one trial

From the top-level `ACSP` directory:

```bash
python ablation_study/RK/Trace/attack_trace_rk.py --data-dir /path/to/Trace_SAX --epsilon 2 --beta 0.10 --no-dynamic-reselect --runs 1 --output results/trace_rk_no_reselection_once.json
```
