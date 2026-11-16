# OneTrans-Lite integration example

This small example separates a native PyTorch ranker from its RecAgora adapter. It demonstrates integration, not a benchmark result or a capacity-matched OneTrans implementation.

| File | Role |
| --- | --- |
| [onetrans_torch/model.py](onetrans_torch/model.py) | Native model: query/candidate/history projections, history Transformer, cross-attention, task heads |
| [onetrans_torch/train.py](onetrans_torch/train.py) | Standalone synthetic-data training |
| [onetrans_recagora/model.py](onetrans_recagora/model.py) | RecAgora views, output/loss adapter |
| [onetrans_recagora/onetrans_lite.yaml](onetrans_recagora/onetrans_lite.yaml) | Public-data configuration |

## Native model

Inputs are query [B,Dq], candidates [B,L,Dc], history [B,H,Dh], and history_valid [B,H]. Output logits are [B,L,num_tasks]. The native model has no slot mapping or sparse embedding tables.

```bash
python scripts/recagora_demo/onetrans_lite/onetrans_torch/train.py
```

Synthetic training uses its own settings; do not interpret it as the public-data experiment.

## RecAgora adapter

OneTransLiteRecAgoraModel subclasses RecAgoraRankingModel, derives dimensions from RecAgoraInputSpec, and translates looked-up views into the four native tensors. It returns task logits [B,L] and implements candidate-aware masking/loss handling.

The YAML selects 158 fields, including action but excluding direct time. Ordinary embeddings are 8-dimensional; action width is 256. Dense AdamW settings come from the shared training configuration; sparse embeddings use TorchRec ExactAdagrad.

With the distributed environment set as in [Quick Start](../../../README.md#quick-start):

```bash
bash run.sh --dataset small --exp onetrans_lite_toy \
  --config scripts/recagora_demo/onetrans_lite/onetrans_recagora/onetrans_lite.yaml \
  --data-root ./datasets/toy-small-public \
  --train-days 01-01 --eval-days 02-02 --batch 512
```

See [model integration](../../../docs/guides/add-a-model.md) and [the public API](../../../docs/public-api.md). Even this small dense network retains the data profile's full sparse vocabulary.
