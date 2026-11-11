# Comparison baselines

| Folder | Contents |
| --- | --- |
| `RRA/` | Random Report Attack for Trace and Symbols. |
| `GP/` | Greedy Poisoning for Trace and Symbols. |

The RRA and GP READMEs list every Python file and give one-trial examples.

## Run one example

From the top-level `ACSP` directory:

```bash
python baselines/RRA/Trace/rra_trace.py --data-dir /path/to/Trace_SAX --epsilon 1 --beta 0.15 --runs 1 --output results/trace_rra_once.json
```
