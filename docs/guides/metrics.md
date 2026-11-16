# Metrics and reference results

## Metric definitions

The paper reports **global AUC** and **Logloss** for click, add-to-cart, and order. Both use the same valid candidate population.

| Metric | Definition |
| --- | --- |
| Global AUC | Exact binary ROC-AUC over all evaluated predictions for a task; ties count 0.5. Enable --global-eval. |
| Logloss | Mean binary cross-entropy per task; excludes model-specific auxiliary/listwise losses. |
| Request AUC | Mean within-request positive/negative pair accuracy; only requests containing both classes contribute. |
| HR@K | Fraction of positive-containing requests with a positive in the top K. |
| NDCG@K | Mean binary-gain DCG/IDCG over positive-containing requests. |

Request AUC, HR, and NDCG are supplementary diagnostics, not substitutes for the paper's global AUC. Full and exposed-only scopes evaluate different populations.

## Output fields

Read outputs/eval_metrics.json. requests counts evaluated requests; predictions sums valid task predictions and is not a request count. Inspect auc, auc_mode, losses, request_auc, request_auc_counts, world_size, amp_enabled, negative_mask_fields, valid_candidates, masked_candidates, and empty_requests.

A null AUC can indicate a missing positive/negative class. status=done indicates completion, not numerical validity: check for NaN/Inf. All-invalid evaluation returns no_valid_samples and null loss. Exact global AUC gathers predictions and labels and can require substantial host memory; --global-auc-workers changes computation parallelism, not prediction storage.

## Paper results

The following values are transcribed from manuscript **Table 3**, at the reported four-decimal precision. They are not fresh evaluations or Toy targets. Read the [configuration coverage](train-and-evaluate.md#configuration-coverage), especially INFNet and OneRank V2 Large, before associating scores with current configurations. Unrounded outputs and numerical tolerances are pending.

### RecAgora-60M

| Model | Click AUC | Click Logloss | ATC AUC | ATC Logloss | Order AUC | Order Logloss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DIN | 0.7425 | 0.2505 | 0.8275 | 0.0492 | 0.8924 | 0.0179 |
| DLRM | 0.7469 | 0.2492 | 0.8318 | 0.0490 | 0.8948 | 0.0179 |
| DCNv2 | 0.7492 | 0.2488 | 0.8341 | 0.0489 | 0.8940 | 0.0179 |
| DIN + DCNv2 | 0.7482 | 0.2490 | 0.8333 | 0.0490 | 0.8929 | 0.0180 |
| Wukong | 0.7495 | 0.2486 | 0.8346 | 0.0489 | 0.8962 | 0.0179 |
| HiFormer | 0.7503 | 0.2484 | 0.8316 | 0.0489 | 0.8965 | 0.0177 |
| RankMixer | 0.7488 | 0.2488 | 0.8342 | 0.0489 | 0.8963 | 0.0179 |
| Zenith | 0.7503 | 0.2482 | 0.8366 | 0.0487 | 0.8976 | 0.0178 |
| TokenMixer | 0.7517 | 0.2480 | 0.8370 | 0.0487 | 0.8987 | 0.0178 |
| HeMix | 0.7496 | 0.2486 | 0.8373 | 0.0487 | 0.8976 | 0.0178 |
| UniMixer | 0.7482 | 0.2489 | 0.8360 | 0.0487 | 0.8958 | 0.0178 |
| SSR-D | 0.7480 | 0.2490 | 0.8337 | 0.0489 | 0.8954 | 0.0179 |
| HSTU | 0.7495 | 0.2487 | 0.8383 | 0.0486 | 0.8970 | 0.0178 |
| LONGER | 0.7506 | 0.2484 | 0.8393 | 0.0485 | 0.8993 | 0.0177 |
| MTGR | 0.7517 | 0.2480 | 0.8404 | 0.0485 | 0.8998 | 0.0177 |
| INFNet | 0.7497 | 0.2484 | 0.8386 | 0.0486 | 0.8983 | 0.0178 |
| OneTrans | 0.7521 | 0.2480 | 0.8416 | 0.0484 | 0.9005 | 0.0177 |
| HyFormer | 0.7520 | 0.2479 | 0.8408 | 0.0485 | 0.8998 | 0.0178 |
| EST | 0.7441 | 0.2501 | 0.8294 | 0.0492 | 0.8908 | 0.0180 |
| MixFormer | 0.7529 | 0.2477 | 0.8412 | 0.0484 | 0.8998 | 0.0178 |
| SORT | 0.7498 | 0.2485 | 0.8392 | 0.0485 | 0.8993 | 0.0177 |
| OneRank V1 | 0.7668 | 0.2435 | 0.8503 | 0.0478 | 0.9068 | 0.0175 |
| OneRank V2 | 0.7702 | 0.2425 | 0.8525 | 0.0475 | 0.9116 | 0.0172 |

### RecAgora-270M

| Model | Click AUC | Click Logloss | ATC AUC | ATC Logloss | Order AUC | Order Logloss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DIN | 0.7509 | 0.2446 | 0.8348 | 0.0482 | 0.8964 | 0.0181 |
| DLRM | 0.7561 | 0.2432 | 0.8402 | 0.0476 | 0.8980 | 0.0181 |
| DCNv2 | 0.7594 | 0.2423 | 0.8423 | 0.0475 | 0.8951 | 0.0182 |
| DIN + DCNv2 | 0.7590 | 0.2424 | 0.8417 | 0.0476 | 0.8940 | 0.0183 |
| Wukong | 0.7603 | 0.2420 | 0.8448 | 0.0473 | 0.9024 | 0.0179 |
| HiFormer | 0.7607 | 0.2419 | 0.8450 | 0.0473 | 0.9017 | 0.0180 |
| RankMixer | 0.7605 | 0.2420 | 0.8445 | 0.0473 | 0.9010 | 0.0180 |
| Zenith | 0.7625 | 0.2414 | 0.8473 | 0.0472 | 0.9020 | 0.0180 |
| TokenMixer | 0.7625 | 0.2413 | 0.8464 | 0.0472 | 0.9016 | 0.0180 |
| HeMix | 0.7609 | 0.2419 | 0.8474 | 0.0472 | 0.9026 | 0.0179 |
| UniMixer | 0.7600 | 0.2422 | 0.8461 | 0.0473 | 0.9009 | 0.0180 |
| SSR-D | 0.7579 | 0.2427 | 0.8421 | 0.0475 | 0.8988 | 0.0181 |
| HSTU | 0.7605 | 0.2420 | 0.8480 | 0.0471 | 0.9017 | 0.0180 |
| LONGER | 0.7610 | 0.2419 | 0.8487 | 0.0471 | 0.9029 | 0.0180 |
| MTGR | 0.7626 | 0.2414 | 0.8507 | 0.0470 | 0.9048 | 0.0178 |
| INFNet | 0.7598 | 0.2423 | 0.8469 | 0.0473 | 0.9011 | 0.0182 |
| OneTrans | 0.7634 | 0.2412 | 0.8507 | 0.0470 | 0.9040 | 0.0179 |
| HyFormer | 0.7636 | 0.2411 | 0.8507 | 0.0470 | 0.9029 | 0.0180 |
| EST | 0.7525 | 0.2443 | 0.8377 | 0.0479 | 0.8936 | 0.0183 |
| MixFormer | 0.7645 | 0.2408 | 0.8515 | 0.0470 | 0.9040 | 0.0180 |
| SORT | 0.7621 | 0.2416 | 0.8495 | 0.0471 | 0.9029 | 0.0180 |
| OneRank V1 | 0.7788 | 0.2366 | 0.8582 | 0.0464 | 0.9081 | 0.0178 |
| OneRank V2 | 0.7821 | 0.2358 | 0.8634 | 0.0459 | 0.9189 | 0.0171 |

Cross-scale model-rank Spearman correlations are 0.96/0.97/0.94 for click/ATC/order (Section 4.3). These are correlations of model rankings, not task AUC values.

## Repeated runs and efficiency

For variance estimates, train separate seeds by changing train.seed and preserving separate outputs. Report mean/std and the exact evaluation scope. Re-evaluating one checkpoint does not measure training variance.

For throughput and latency comparisons, hold hardware, batch/process count, precision, candidate scope, and history length fixed. Specify warm-up, measurement interval, whether loading/communication/checkpoint time is included, and the units (requests or candidates). Separate dense model parameters from vocabulary-dependent embeddings. Do not infer end-to-end efficiency from architecture size alone.
