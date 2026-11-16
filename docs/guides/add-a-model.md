# Add a model

Start from the [OneTrans-Lite adapter](../../scripts/recagora_demo/onetrans_lite/onetrans_recagora/model.py) and its [configuration](../../scripts/recagora_demo/onetrans_lite/onetrans_recagora/onetrans_lite.yaml). It demonstrates a native PyTorch model behind a RecAgora adapter; the [API reference](../public-api.md) defines available views.

## Implement the model

1. Subclass RecAgoraRankingModel and accept input_spec plus explicit architecture parameters.
2. Call super().__init__(config, input_spec); derive widths and history slots from input_spec.
3. Call self.get_slots(batch) once in forward. Preserve candidate alignment and history lengths.
4. Return structured_ranking_output with task-ordered logits [B,L].
5. Implement compute_loss(output, batch), returning a scalar tensor and named scalar diagnostics.

Use input_spec.task_names as the task contract. Pointwise networks must restore the request/candidate axes before returning. Sparse embeddings and lookup are owned by the framework; do not duplicate tables or map raw IDs inside the network. If changing history padding, move time values and masks with the same events.

## Configure

Copy the demo YAML and update model.class_path to python.module:ClassName. Keep data_config/training_config includes, select features the model actually uses, and declare embedding widths and private model.params. Configurations are passed as keyword arguments to the constructor; reject unknown parameters.

For a file at configs/models/my_ranker/my_ranker.yaml, includes are relative to that file:

```yaml
data_config: ../../data/small.yaml
training_config: ../../train/adamw.yaml
```

The full model/embedding/storage configuration is shown in the working demo. Prefer a copied runnable configuration over a partially specified schema.

## Candidate mask contract

If exposing a *_negmask.yaml configuration, explicitly declare supports_candidate_mask, candidate_mask_all_empty_policy, and candidate_mask_loss_reduction. Use the [mask-aware BCE helper](../public-api.md#candidate-masking), and mask attention, listwise objectives, and auxiliary terms consistently where applicable.

Do not treat a graph-connected zero loss as a valid optimizer update for a globally empty batch. The framework coordinates empty-batch handling. Compare masked behavior with the unmasked path, including all-valid, partially valid, and all-invalid requests.

## Validate

Check constructor and dynamic class-path loading, selected-slot dimensions, empty histories, padding alignment, task names, output shape, finite loss, backward, and checkpoint save/load. Test mask semantics if supported. Run a short training/evaluation job on Toy data before full-data experiments.

Training and evaluation commands are shared across models; see the [run guide](train-and-evaluate.md). Architecture changes require a new experiment identity; do not attach old manuscript scores to a changed configuration.
