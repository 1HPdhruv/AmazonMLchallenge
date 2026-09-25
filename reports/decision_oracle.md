# Decision-layer oracle

**SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE**

Frozen candidates + scores; only the mechanism varies. Parameters tuned on the calib fold (carved from train); metrics on VAL. `cal:` = isotonic-calibrated scores (fit on calib). `+excl` = record-exclusive post-filter (EXPERIMENTAL: assumes a record belongs to <=1 entity).

**Selected: `cal:multi`**. Rule: take the best VAL macro F0.5 that does not collapse (>0.05 below the best) in the 0-1 / 2-4 candidate buckets or the no-match stratum; it replaces the default `cal:multi` only if a paired bootstrap over VAL entities gives P(improve) >= 0.9. Selection detail: {'top_by_val': 'raw:multi+excl', 'default': 'cal:multi', 'bootstrap_top_vs_default': {'diff': 0.0015, 'ci90': [-0.004, 0.0074], 'p_improve': 0.663}, 'winner': 'cal:multi'}

## Overall

| mechanism | calib_f05 | val_macro_f05 | val_pair_precision | val_pair_recall | val_no_match_fp_rate | val_multi_precision | val_multi_recall | val_fp_per_entity | val_fn_per_entity | val_accepted_per_entity | params | collapses | paired bootstrap vs default |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw:multi+excl | 0.9271 | 0.9535 | 0.9654 | 0.9254 | 0.0372 | 0.9871 | 0.8988 | 0.0333 | 0.075 | 0.9642 | {'t': np.float64(0.75), 'delta': 0.05} |  | {'diff': 0.0015, 'ci90': [-0.004, 0.0074], 'p_improve': 0.663} |
| cal:multi+excl | 0.9283 | 0.952 | 0.98 | 0.8931 | 0.0256 | 0.9881 | 0.8548 | 0.0183 | 0.1075 | 0.9167 | {'t': np.float64(0.4), 'delta': 0.2} |  | {'diff': 0.0, 'ci90': [0.0, 0.0], 'p_improve': 0.0} |
| cal:multi | 0.9278 | 0.952 | 0.98 | 0.8931 | 0.0256 | 0.9881 | 0.8548 | 0.0183 | 0.1075 | 0.9167 | {'t': np.float64(0.4), 'delta': 0.2} |  | {'diff': 0.0, 'ci90': [0.0, 0.0], 'p_improve': 0.0} |
| cal:threshold+excl | 0.927 | 0.9505 | 0.971 | 0.9155 | 0.0256 | 0.9855 | 0.8944 | 0.0275 | 0.085 | 0.9483 | {'t': np.float64(0.4)} |  | {'diff': -0.0016, 'ci90': [-0.0037, 0.0004], 'p_improve': 0.102} |
| cal:multi_t2+excl | 0.927 | 0.9505 | 0.971 | 0.9155 | 0.0256 | 0.9855 | 0.8944 | 0.0275 | 0.085 | 0.9483 | {'t': np.float64(0.4), 't2': np.float64(0.4)} |  | {'diff': -0.0016, 'ci90': [-0.0037, 0.0004], 'p_improve': 0.102} |
| cal:gmm+excl | 0.9261 | 0.9505 | 0.971 | 0.9155 | 0.0256 | 0.9855 | 0.8944 | 0.0275 | 0.085 | 0.9483 | {'t': np.float64(0.4)} |  | {'diff': -0.0016, 'ci90': [-0.0037, 0.0004], 'p_improve': 0.102} |
| cal:multi_t2 | 0.9261 | 0.9501 | 0.9701 | 0.9155 | 0.0256 | 0.9855 | 0.8944 | 0.0283 | 0.085 | 0.9492 | {'t': np.float64(0.4), 't2': np.float64(0.4)} |  | {'diff': -0.0019, 'ci90': [-0.0042, 0.0002], 'p_improve': 0.064} |
| cal:gmm | 0.9251 | 0.9501 | 0.9701 | 0.9155 | 0.0256 | 0.9855 | 0.8944 | 0.0283 | 0.085 | 0.9492 | {'t': np.float64(0.4)} |  | {'diff': -0.0019, 'ci90': [-0.0042, 0.0002], 'p_improve': 0.064} |
| cal:threshold | 0.9261 | 0.9501 | 0.9701 | 0.9155 | 0.0256 | 0.9855 | 0.8944 | 0.0283 | 0.085 | 0.9492 | {'t': np.float64(0.4)} |  | {'diff': -0.0019, 'ci90': [-0.0042, 0.0002], 'p_improve': 0.064} |
| raw:threshold+excl | 0.926 | 0.9493 | 0.9726 | 0.913 | 0.0256 | 0.987 | 0.893 | 0.0258 | 0.0875 | 0.9442 | {'t': np.float64(0.965)} |  | {'diff': -0.0028, 'ci90': [-0.0058, -0.0002], 'p_improve': 0.037} |
| raw:multi_t2+excl | 0.9247 | 0.9493 | 0.9694 | 0.918 | 0.0279 | 0.9871 | 0.8988 | 0.0292 | 0.0825 | 0.9525 | {'t': np.float64(0.95), 't2': np.float64(0.95)} |  | {'diff': -0.0027, 'ci90': [-0.0055, -0.0002], 'p_improve': 0.039} |
| raw:multi | 0.9244 | 0.9487 | 0.9709 | 0.913 | 0.0256 | 0.9854 | 0.893 | 0.0275 | 0.0875 | 0.9458 | {'t': np.float64(0.965), 'delta': 0.05} |  | {'diff': -0.0033, 'ci90': [-0.0063, -0.0007], 'p_improve': 0.015} |
| raw:threshold | 0.9244 | 0.9487 | 0.9709 | 0.913 | 0.0256 | 0.9854 | 0.893 | 0.0275 | 0.0875 | 0.9458 | {'t': np.float64(0.965)} |  | {'diff': -0.0033, 'ci90': [-0.0063, -0.0007], 'p_improve': 0.015} |
| cal:hybrid_by_bucket | nan | 0.9485 | 0.9764 | 0.8923 | 0.0349 | 0.9881 | 0.8534 | 0.0217 | 0.1083 | 0.9192 | {'5-20': {'mechanism': 'multi', 'params': {'t': np.float64(0.05), 'delta': 0.15}, 'calib_f05': 0.9231}, '20+': {'mechanism': 'multi', 'params': {'t': np.float64(0.4), 'delta': 0.2}, 'calib_f05': 0.928}} |  | {'diff': -0.0035, 'ci90': [-0.0064, -0.001], 'p_improve': 0.0} |
| raw:multi_t2 | 0.9231 | 0.948 | 0.9668 | 0.918 | 0.0302 | 0.9855 | 0.8988 | 0.0317 | 0.0825 | 0.955 | {'t': np.float64(0.95), 't2': np.float64(0.95)} |  | {'diff': -0.004, 'ci90': [-0.0072, -0.0012], 'p_improve': 0.008} |
| raw:hybrid_by_bucket | nan | 0.947 | 0.9692 | 0.913 | 0.0302 | 0.9854 | 0.893 | 0.0292 | 0.0875 | 0.9475 | {'5-20': {'mechanism': 'multi', 'params': {'t': np.float64(0.05), 'delta': 0.05}, 'calib_f05': 0.9231}, '20+': {'mechanism': 'multi', 'params': {'t': np.float64(0.965), 'delta': 0.05}, 'calib_f05': 0.9245}} |  | {'diff': -0.005, 'ci90': [-0.0086, -0.0018], 'p_improve': 0.004} |
| raw:gmm+excl | 0.9177 | 0.9431 | 0.9466 | 0.9395 | 0.0419 | 0.9798 | 0.9252 | 0.0533 | 0.0608 | 0.9983 | {'t': np.float64(0.7)} |  | {'diff': -0.009, 'ci90': [-0.0154, -0.0015], 'p_improve': 0.02} |
| raw:gmm | 0.9117 | 0.9414 | 0.9419 | 0.9395 | 0.0442 | 0.9768 | 0.9252 | 0.0583 | 0.0608 | 1.0033 | {'t': np.float64(0.7)} |  | {'diff': -0.0106, 'ci90': [-0.0174, -0.0028], 'p_improve': 0.008} |
| raw:top1_t+excl | 0.8875 | 0.915 | 0.9751 | 0.6172 | 0.0372 | 0.9959 | 0.3534 | 0.0158 | 0.385 | 0.6367 | {'t': np.float64(0.75)} |  | {'diff': -0.037, 'ci90': [-0.0441, -0.0299], 'p_improve': 0.0} |
| raw:top1_margin+excl | 0.8875 | 0.915 | 0.9751 | 0.6172 | 0.0372 | 0.9959 | 0.3534 | 0.0158 | 0.385 | 0.6367 | {'t': np.float64(0.75), 'm': 0.0} |  | {'diff': -0.037, 'ci90': [-0.0441, -0.0299], 'p_improve': 0.0} |
| cal:top1_t | 0.8886 | 0.9117 | 0.9826 | 0.6098 | 0.0256 | 0.9959 | 0.3534 | 0.0108 | 0.3925 | 0.6242 | {'t': np.float64(0.4)} |  | {'diff': -0.0403, 'ci90': [-0.0454, -0.0355], 'p_improve': 0.0} |
| cal:top1_t+excl | 0.8886 | 0.9117 | 0.9826 | 0.6098 | 0.0256 | 0.9959 | 0.3534 | 0.0108 | 0.3925 | 0.6242 | {'t': np.float64(0.4)} |  | {'diff': -0.0403, 'ci90': [-0.0454, -0.0355], 'p_improve': 0.0} |
| cal:top1_margin+excl | 0.8886 | 0.9117 | 0.9826 | 0.6098 | 0.0256 | 0.9959 | 0.3534 | 0.0108 | 0.3925 | 0.6242 | {'t': np.float64(0.4), 'm': 0.0} |  | {'diff': -0.0403, 'ci90': [-0.0454, -0.0355], 'p_improve': 0.0} |
| cal:top1_margin | 0.8886 | 0.9117 | 0.9826 | 0.6098 | 0.0256 | 0.9959 | 0.3534 | 0.0108 | 0.3925 | 0.6242 | {'t': np.float64(0.4), 'm': 0.0} |  | {'diff': -0.0403, 'ci90': [-0.0454, -0.0355], 'p_improve': 0.0} |
| raw:top1_margin | 0.8864 | 0.91 | 0.98 | 0.6098 | 0.0302 | 0.9959 | 0.3534 | 0.0125 | 0.3925 | 0.6258 | {'t': np.float64(0.95), 'm': 0.0} |  | {'diff': -0.042, 'ci90': [-0.0474, -0.037], 'p_improve': 0.0} |
| raw:top1_t | 0.8875 | 0.91 | 0.9826 | 0.6081 | 0.0256 | 0.9959 | 0.3534 | 0.0108 | 0.3942 | 0.6225 | {'t': np.float64(0.965)} |  | {'diff': -0.042, 'ci90': [-0.0475, -0.0369], 'p_improve': 0.0} |
| cal:top1+excl | 0.6918 | 0.7108 | 0.7303 | 0.628 | 0.6372 | 0.9918 | 0.3534 | 0.2333 | 0.3742 | 0.865 | {} | ['f05[gt_type=no_match]'] | {'diff': -0.2412, 'ci90': [-0.2624, -0.2207], 'p_improve': 0.0} |
| raw:top1+excl | 0.6885 | 0.71 | 0.7288 | 0.628 | 0.6395 | 0.9877 | 0.3534 | 0.235 | 0.3742 | 0.8667 | {} | ['f05[gt_type=no_match]'] | {'diff': -0.242, 'ci90': [-0.2632, -0.2218], 'p_improve': 0.0} |
| raw:top1 | 0.5673 | 0.5808 | 0.6317 | 0.628 | 1.0 | 0.9837 | 0.3534 | 0.3683 | 0.3742 | 1.0 | {} | ['f05[gt_type=no_match]'] | {'diff': -0.3712, 'ci90': [-0.3932, -0.3483], 'p_improve': 0.0} |
| cal:top1 | 0.5685 | 0.5808 | 0.6317 | 0.628 | 1.0 | 0.9837 | 0.3534 | 0.3683 | 0.3742 | 1.0 | {} | ['f05[gt_type=no_match]'] | {'diff': -0.3712, 'ci90': [-0.3932, -0.3483], 'p_improve': 0.0} |

## Stratified macro F0.5 (VAL)

| mechanism | f05[cand_bucket=20+] | f05[cand_bucket=5-20] | f05[gt_type=multi] | f05[gt_type=no_match] | f05[gt_type=single] |
|---|---|---|---|---|---|
| raw:multi+excl | 0.9524 | 0.9871 | 0.9487 | 0.9628 | 0.9481 |
| cal:multi+excl | 0.951 | 0.9814 | 0.9378 | 0.9744 | 0.9403 |
| cal:multi | 0.951 | 0.9814 | 0.9378 | 0.9744 | 0.9403 |
| cal:threshold+excl | 0.9492 | 0.9871 | 0.9465 | 0.9744 | 0.9327 |
| cal:multi_t2+excl | 0.9492 | 0.9871 | 0.9465 | 0.9744 | 0.9327 |
| cal:gmm+excl | 0.9492 | 0.9871 | 0.9465 | 0.9744 | 0.9327 |
| cal:multi_t2 | 0.9488 | 0.9871 | 0.9465 | 0.9744 | 0.9319 |
| cal:gmm | 0.9488 | 0.9871 | 0.9465 | 0.9744 | 0.9319 |
| cal:threshold | 0.9488 | 0.9871 | 0.9465 | 0.9744 | 0.9319 |
| raw:threshold+excl | 0.9479 | 0.9871 | 0.9469 | 0.9744 | 0.9297 |
| raw:multi_t2+excl | 0.948 | 0.9871 | 0.9487 | 0.9721 | 0.931 |
| raw:multi | 0.9474 | 0.9871 | 0.946 | 0.9744 | 0.9289 |
| raw:threshold | 0.9474 | 0.9871 | 0.946 | 0.9744 | 0.9289 |
| cal:hybrid_by_bucket | 0.951 | 0.876 | 0.9369 | 0.9651 | 0.9403 |
| raw:multi_t2 | 0.9466 | 0.9871 | 0.9479 | 0.9698 | 0.9302 |
| raw:hybrid_by_bucket | 0.9474 | 0.9371 | 0.946 | 0.9698 | 0.9289 |
| raw:gmm+excl | 0.9415 | 0.9887 | 0.9406 | 0.9581 | 0.9319 |
| raw:gmm | 0.9398 | 0.9887 | 0.9386 | 0.9558 | 0.931 |
| raw:top1_t+excl | 0.9148 | 0.9208 | 0.7347 | 0.9628 | 0.96 |
| raw:top1_margin+excl | 0.9148 | 0.9208 | 0.7347 | 0.9628 | 0.96 |
| cal:top1_t | 0.9114 | 0.9208 | 0.7347 | 0.9744 | 0.9429 |
| cal:top1_t+excl | 0.9114 | 0.9208 | 0.7347 | 0.9744 | 0.9429 |
| cal:top1_margin+excl | 0.9114 | 0.9208 | 0.7347 | 0.9744 | 0.9429 |
| cal:top1_margin | 0.9114 | 0.9208 | 0.7347 | 0.9744 | 0.9429 |
| raw:top1_margin | 0.9096 | 0.9208 | 0.7347 | 0.9698 | 0.9429 |
| raw:top1_t | 0.9096 | 0.9208 | 0.7347 | 0.9744 | 0.939 |
| cal:top1+excl | 0.7105 | 0.7208 | 0.7347 | 0.3628 | 0.9848 |
| raw:top1+excl | 0.7096 | 0.7208 | 0.7347 | 0.3605 | 0.9848 |
| raw:top1 | 0.5777 | 0.6708 | 0.7347 | 0.0 | 0.9848 |
| cal:top1 | 0.5777 | 0.6708 | 0.7347 | 0.0 | 0.9848 |

## Country / true-source strata for the selected mechanism

| stratum | n | macro F0.5 | no-match FP | FP/entity | FN/entity |
|---|---|---|---|---|---|
| overall | 1200 | 0.952 | 0.0256 | 0.0183 | 0.1075 |
| cand_bucket=20+ | 1160 | 0.951 | 0.0262 | 0.0181 | 0.1069 |
| cand_bucket=5-20 | 40 | 0.9814 | 0.0 | 0.025 | 0.125 |
| gt_type=multi | 245 | 0.9378 | None | 0.0286 | 0.4041 |
| gt_type=no_match | 430 | 0.9744 | 0.0256 | 0.0256 | 0.0 |
| gt_type=single | 525 | 0.9403 | None | 0.0076 | 0.0571 |
| country=BR | 122 | 0.9562 | 0.0263 | 0.0164 | 0.0902 |
| country=DE | 153 | 0.955 | 0.0588 | 0.0458 | 0.1046 |
| country=FR | 122 | 0.9462 | 0.0 | 0.0082 | 0.1721 |
| country=IN | 189 | 0.9535 | 0.0154 | 0.0053 | 0.1164 |
| country=JP | 105 | 0.9345 | 0.0488 | 0.019 | 0.1143 |
| country=US | 509 | 0.9546 | 0.0205 | 0.0177 | 0.0923 |
| true_sources=none | 430 | 0.9744 | 0.0256 | 0.0256 | 0.0 |
| true_sources=source2 | 349 | 0.9638 | None | 0.0057 | 0.0659 |
| true_sources=source2+source3 | 166 | 0.9354 | None | 0.0301 | 0.4518 |
| true_sources=source3 | 255 | 0.909 | None | 0.0157 | 0.1216 |

## Low-candidate-count simulation (calibrated scores; lists truncated to top-n)

The synthetic retriever never yields <5 candidates, so the natural 1 and 2-4 buckets are empty. Here every entity keeps only its top-n candidates; parameters tuned on calib, scored on VAL.

| n | mechanism | val F0.5 | no-match F0.5 | single F0.5 | no-match FP | params |
|---|---|---|---|---|---|---|
| 1 | top1_t | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4)} |
| 1 | top1_margin | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4), 'm': 0.0} |
| 1 | multi | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4), 'delta': 0.05} |
| 1 | gmm | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4)} |
| 2 | top1_t | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4)} |
| 2 | top1_margin | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4), 'm': 0.0} |
| 2 | multi | 0.9421 | 0.9744 | 0.9403 | 0.0256 | {'t': np.float64(0.4), 'delta': 0.15} |
| 2 | gmm | 0.9395 | 0.9744 | 0.942 | 0.0256 | {'t': np.float64(0.4)} |
| 3 | top1_t | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4)} |
| 3 | top1_margin | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4), 'm': 0.0} |
| 3 | multi | 0.9505 | 0.9744 | 0.9403 | 0.0256 | {'t': np.float64(0.4), 'delta': 0.2} |
| 3 | gmm | 0.945 | 0.9744 | 0.9319 | 0.0256 | {'t': np.float64(0.4)} |
| 4 | top1_t | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4)} |
| 4 | top1_margin | 0.9117 | 0.9744 | 0.9429 | 0.0256 | {'t': np.float64(0.4), 'm': 0.0} |
| 4 | multi | 0.952 | 0.9744 | 0.9403 | 0.0256 | {'t': np.float64(0.4), 'delta': 0.2} |
| 4 | gmm | 0.9486 | 0.9744 | 0.9319 | 0.0256 | {'t': np.float64(0.4)} |
