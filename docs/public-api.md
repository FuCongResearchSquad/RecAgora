# Public Python API

```python
from recagora import (
    RecAgoraRankingModel, RecAgoraInputSpec, RecAgoraSparseBatch,
    RecAgoraModelBatch, RankingOutput, structured_ranking_output,
)
from training import RecAgoraConfig, load_recagora_config
```

The framework constructs specs, batches, views, and lookup modules. Models consume these objects rather than creating their own data pipeline. Underscored members are internal. See the [integration guide](guides/add-a-model.md) and [working adapter](../scripts/recagora_demo/onetrans_lite/onetrans_recagora/model.py).

B denotes requests, L candidates (24), D embedding width, and H valid history events.

## RecAgoraRankingModel

Subclass torch.nn.Module through RecAgoraRankingModel. Call super().__init__(config, input_spec), then implement:

```text
forward(batch: RecAgoraSparseBatch) -> RankingOutput
compute_loss(output: RankingOutput, batch: RecAgoraSparseBatch)
    -> tuple[Tensor, dict[str, Tensor]]
```

The loss is a scalar for backward; the dictionary contains named scalar losses/diagnostics. Evaluation computes metrics from logits without calling the model's training loss.

Call self.get_slots(batch) once per forward to obtain RecAgoraModelBatch. self.slot_dim(id) and self.history_spec(name) expose dimensions and history settings. The framework binds RecAgoraSlotLookup before forward; constructing a model directly does not bind lookup.

## Specs and views

| Type | Main members |
| --- | --- |
| RecAgoraInputSpec | list_len, task_names, query/candidate FixedFeatureSpec, histories, slot(id), total_dim(ids) |
| RecAgoraSlotSpec | slot_id, view, role, representation, encoding, reduction, output_dim, mapped_ids_exposed |
| FixedFeatureSpec | slot_ids, total_dim |
| HistoryGroupSpec | categorical/direct slot IDs, event width, max_length, event_order, truncate_side, padding settings, length anchor |
| RecAgoraSparseBatch | labels, candidate_metadata, candidate_valid_mask, candidate_loss_context, batch_size, list_len, to(device) |
| RecAgoraModelBatch | ids, query, candidate, histories, labels, candidate_metadata, candidate_valid_mask, flatten_candidates() |
| FixedFeatureView | Mapping[int, Tensor], slot_ids, concat(slot_ids=None) |
| HistoryView | lengths, categorical, direct, mapped_ids, max_length |
| JaggedFeatureView | values [sum(H),D], lengths [B], to_padded(max_length=..., padding_side=...) |
| JaggedIDView | Mapped int64 IDs [sum(H)], lengths [B], to_padded(...) |
| NamedTensorView | Mapping[str, Tensor], attribute access, flatten_candidates() |
| BatchIDs | request_id [B], optional user_id [B], item_id [B,L] |
| RecAgoraPointwiseBatch | Flattened candidate views/labels, request_index, candidate_index, shared histories |

Query embeddings have shape [B,D], candidate embeddings [B,L,D]. History categorical fields preserve events; query/candidate multi-ID values are reduced according to the feature contract. concat() follows the view's default slot order, or the explicit requested order.

Jagged to_padded() keeps the most recent tail and supports left/right padding. Direct time is initially right-padded; align it with categorical padding and validity masks. Mapped IDs require model.features.mapped_id_slots and are vocabulary IDs, not raw business IDs.

flatten_candidates() produces [B×L] candidate positions but does not copy histories L times. Use request_index for candidate-conditioned history access, then restore logits to [B,L]. Views expose read-only mappings; clone tensors before mutating shared values.

In public no-ID contracts, user_id is absent and item_id contains candidate positions only. It is not a real item identifier.

## Output contract

```python
return structured_ranking_output(
    {task: task_logits[task] for task in self.task_names},
    aux={"regularization": regularization_loss},
)
```

RankingOutput is a built-in dictionary with logits and aux keys. Logits are task-ordered tensors [B,L]. aux uses tensors and nested built-in dict/list/tuple containers with string dictionary keys; use tensor scalars rather than Python numbers. Empty auxiliary output is {}. Shape validation does not ensure finite values.

## Candidate masking

CandidateLossContext supplies local/global valid counts and world_size. It is runtime state, not a model parameter. Models should not initiate their own count collectives.

```python
from recagora import masked_binary_cross_entropy_with_logits

loss = masked_binary_cross_entropy_with_logits(
    logits, labels.to(device=logits.device, dtype=logits.dtype),
    valid_mask=batch.candidate_valid_mask,
    context=batch.candidate_loss_context,
    reduction="mean",
)
```

The helper selects valid values before BCE, avoiding NaN×0 contamination. With distributed context, mean reduction returns world_size × local_sum / global_valid_count for averaged gradients. Empty valid sets return a graph-connected zero; the training loop separately handles globally empty batches.

The helper does not mask attention, listwise softmax, or auxiliary losses. Supporting models must implement the [mask contract](guides/add-a-model.md#candidate-mask-contract).

## Configuration and compatibility

load_recagora_config resolves model/data/training includes into RecAgoraConfig. The compiler builds feature plans and injects RecAgoraInputSpec before construction. Include paths are relative to their YAML; CLI paths follow the calling directory.

There are no old Python-name aliases. API renaming preserves parameter keys and checkpoint format; persisted metadata keys remain stable. Runtime lookup timing/profiling methods are framework internals, not required model-extension hooks.
