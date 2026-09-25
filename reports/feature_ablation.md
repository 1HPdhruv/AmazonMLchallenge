# Candidate-relative / FS feature ablation

**SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE**

A = current best feature set; B = A + group. Reference decision (multi, tuned on calib). VAL split.

## +fs: dF05 = -0.0011; paired bootstrap {'diff': -0.0011, 'ci90': [-0.0058, 0.0037], 'p_improve': 0.352}; stratum regressions > 0.02: ['cand_bucket=5-20: 0.9871->0.9606']

| stratum | n | A F0.5 | B F0.5 | A no-match FP | B no-match FP |
|---|---|---|---|---|---|
| overall | 1200 | 0.9487 | 0.9476 | 0.0256 | 0.0186 |
| cand_bucket=20+ | 1160 | 0.9474 | 0.9471 | 0.0262 | 0.019 |
| cand_bucket=5-20 | 40 | 0.9871 | 0.9606 | 0.0 | 0.0 |
| gt_type=multi | 245 | 0.946 | 0.9446 | None | None |
| gt_type=no_match | 430 | 0.9744 | 0.9814 | 0.0256 | 0.0186 |
| gt_type=single | 525 | 0.9289 | 0.9213 | None | None |
| country=BR | 122 | 0.9502 | 0.957 | 0.0263 | 0.0263 |
| country=DE | 153 | 0.9549 | 0.9393 | 0.0588 | 0.0588 |
| country=FR | 122 | 0.9498 | 0.9548 | 0.0 | 0.0 |
| country=IN | 189 | 0.9459 | 0.9435 | 0.0154 | 0.0154 |
| country=JP | 105 | 0.9227 | 0.9555 | 0.0488 | 0.0 |
| country=US | 509 | 0.9527 | 0.9459 | 0.0205 | 0.0154 |
| true_sources=none | 430 | 0.9744 | 0.9814 | 0.0256 | 0.0186 |
| true_sources=source2 | 349 | 0.9536 | 0.9524 | None | None |
| true_sources=source2+source3 | 166 | 0.9462 | 0.9437 | None | None |
| true_sources=source3 | 255 | 0.9002 | 0.8865 | None | None |

## +relative_entity: dF05 = +0.0005; paired bootstrap {'diff': 0.0005, 'ci90': [-0.007, 0.0078], 'p_improve': 0.565}; stratum regressions > 0.02: ['gt_type=no_match: 0.9744->0.9465', 'true_sources=none: 0.9744->0.9465']

| stratum | n | A F0.5 | B F0.5 | A no-match FP | B no-match FP |
|---|---|---|---|---|---|
| overall | 1200 | 0.9487 | 0.9492 | 0.0256 | 0.0535 |
| cand_bucket=20+ | 1160 | 0.9474 | 0.9482 | 0.0262 | 0.0548 |
| cand_bucket=5-20 | 40 | 0.9871 | 0.9776 | 0.0 | 0.0 |
| gt_type=multi | 245 | 0.946 | 0.9462 | None | None |
| gt_type=no_match | 430 | 0.9744 | 0.9465 | 0.0256 | 0.0535 |
| gt_type=single | 525 | 0.9289 | 0.9528 | None | None |
| country=BR | 122 | 0.9502 | 0.9391 | 0.0263 | 0.1053 |
| country=DE | 153 | 0.9549 | 0.9429 | 0.0588 | 0.0784 |
| country=FR | 122 | 0.9498 | 0.9455 | 0.0 | 0.025 |
| country=IN | 189 | 0.9459 | 0.9452 | 0.0154 | 0.0615 |
| country=JP | 105 | 0.9227 | 0.9456 | 0.0488 | 0.0488 |
| country=US | 509 | 0.9527 | 0.9566 | 0.0205 | 0.041 |
| true_sources=none | 430 | 0.9744 | 0.9465 | 0.0256 | 0.0535 |
| true_sources=source2 | 349 | 0.9536 | 0.9657 | None | None |
| true_sources=source2+source3 | 166 | 0.9462 | 0.9442 | None | None |
| true_sources=source3 | 255 | 0.9002 | 0.9344 | None | None |

## +relative_record: dF05 = +0.0029; paired bootstrap {'diff': 0.0029, 'ci90': [-0.0037, 0.01], 'p_improve': 0.78}; stratum regressions > 0.02: none

| stratum | n | A F0.5 | B F0.5 | A no-match FP | B no-match FP |
|---|---|---|---|---|---|
| overall | 1200 | 0.9487 | 0.9516 | 0.0256 | 0.0395 |
| cand_bucket=20+ | 1160 | 0.9474 | 0.9507 | 0.0262 | 0.0405 |
| cand_bucket=5-20 | 40 | 0.9871 | 0.9776 | 0.0 | 0.0 |
| gt_type=multi | 245 | 0.946 | 0.9484 | None | None |
| gt_type=no_match | 430 | 0.9744 | 0.9605 | 0.0256 | 0.0395 |
| gt_type=single | 525 | 0.9289 | 0.9458 | None | None |
| country=BR | 122 | 0.9502 | 0.9413 | 0.0263 | 0.0526 |
| country=DE | 153 | 0.9549 | 0.9495 | 0.0588 | 0.0588 |
| country=FR | 122 | 0.9498 | 0.9494 | 0.0 | 0.0 |
| country=IN | 189 | 0.9459 | 0.9555 | 0.0154 | 0.0308 |
| country=JP | 105 | 0.9227 | 0.9329 | 0.0488 | 0.0732 |
| country=US | 509 | 0.9527 | 0.9576 | 0.0205 | 0.0359 |
| true_sources=none | 430 | 0.9744 | 0.9605 | 0.0256 | 0.0395 |
| true_sources=source2 | 349 | 0.9536 | 0.9636 | None | None |
| true_sources=source2+source3 | 166 | 0.9462 | 0.9455 | None | None |
| true_sources=source3 | 255 | 0.9002 | 0.9242 | None | None |

