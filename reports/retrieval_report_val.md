# Retrieval report

**SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE**

Entities evaluated: 1200; true pairs: 1207

| channel | recall | unique-only | marginal | zero-cand | P50 | P75 | P90 | P95 | P99 | max | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 0.449 | 0 | 0.0 | 623 | 0.0 | 1.0 | 13.0 | 18.0 | 24.0 | 29 | 0.011 |
| char | 0.8583 | 34 | 0.0282 | 0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20 | 1.662 |
| word | 0.6819 | 1 | 0.0008 | 157 | 3.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20 | 0.603 |
| addr | 0.855 | 138 | 0.1143 | 0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20 | 0.243 |
| UNION | 0.9809 | - | - | 0 | 37.0 | 41.0 | 46.0 | 52.0 | 58.0 | 59 | 6.509 |
