# Retrieval report

**SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE**

Entities evaluated: 6000; true pairs: 6021

| channel | recall | unique-only | marginal | zero-cand | P50 | P75 | P90 | P95 | P99 | max | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 0.4288 | 0 | 0.0 | 3232 | 0.0 | 1.0 | 12.0 | 18.0 | 24.0 | 29 | 0.011 |
| char | 0.8524 | 163 | 0.0271 | 9 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20 | 1.662 |
| word | 0.6768 | 8 | 0.0013 | 805 | 3.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20 | 0.603 |
| addr | 0.8525 | 722 | 0.1199 | 11 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20 | 0.243 |
| UNION | 0.9806 | - | - | 0 | 38.0 | 41.0 | 46.0 | 51.0 | 58.0 | 59 | 6.509 |
