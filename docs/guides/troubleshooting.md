# Troubleshooting

| Symptom | Check |
| --- | --- |
| TorchRec/FBGEMM import fails | Activate the intended environment, run python -m pip check, and match PyTorch/CUDA wheels to the driver. |
| Model cannot be imported | Run from the project root; use module:ClassName and install dependencies in the same environment. |
| Lookup is not bound | Use the framework factory/train/eval path; direct construction only creates the network. |
| Missing data or slot | Verify the data profile, feature selection, quoted glob, and immediate Parquet directory. Public partitions must be dt=DAYxx. |
| History length/time mismatch | Use the 45041 length anchor; align categorical/time/mask tensors. Never infer length from time != 0. |
| Invalid output | Return structured_ranking_output with task-ordered [B,L] logits and supported aux containers. |
| Checkpoint load fails | Preserve the complete DCP directory and matching model/data contracts. last/ is model-only; step_N/ is for full resume. |
| CUDA out of memory | Reduce per-process batch; inspect sparse-table sizes and duplicated histories. Toy data still uses the configured vocabulary. |
| Global AUC runs out of memory | Exact AUC stores/gathers predictions. More AUC workers do not reduce storage. Subset checks are not full benchmark results. |
| Fewer requests than expected | Check shard/global-batch handling, max_batches, filters, and effective batch/process count. |
| NaN/Inf despite status=done | Completion is not numerical validation. Inspect logits, losses, attention masks, denominators, and auxiliary terms. |
| AUC is null | The evaluated population may lack one class; check task counts separately from NaN/Inf. |

For resume, preserve data identity and batch settings. Elastic world-size recovery requires explicit opt-in and does not permit arbitrary input-contract changes. Completed runs may clean up step checkpoints.

See [data rules](../dataset/usage.md), [history alignment](../dataset/features.md#history-alignment), [API contracts](../public-api.md), and [checkpoint usage](train-and-evaluate.md#checkpoints).
