# Attacks with limited knowledge

| Folder | Contents |
| --- | --- |
| `PK/` | Trace and Symbols attacks using controlled existing users' records and report replacement. |
| `RK/` | Trace and Symbols attacks using reference shapes and weights as planning knowledge. |

The PK and RK READMEs describe every Python file.

## Run one example

From the top-level `ACSP` directory:

```bash
python attacks_with_limited_knowledge/PK/Trace/pk_trace.py --data-dir /path/to/Trace_SAX --epsilon 1 --beta 0.10 --runs 1 --output results/trace_pk_once.json
```
