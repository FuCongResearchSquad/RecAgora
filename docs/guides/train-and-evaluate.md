# Training and evaluation

Install dependencies and run the [Toy example](../../README.md#quick-start) first. Select a model from the [model list](../rank-model-list.md). Run examples from the repository root with your own data paths.

## Paper reproduction

Manuscript Sections 3.1, 4.1–4.4 and Appendix B.2 define the protocol:

- Small: DAY01–DAY07 training, DAY08 evaluation. Large: DAY01–DAY34 training, DAY35 evaluation.
- HotUser: only `train/` and `valid/`, used for training and evaluation respectively.
- Main results use all 24 candidates. *_negmask.yaml uses the same exposed-only mask during training and evaluation, preserving requests and candidate positions.
- Seed 2026; dropout 0; gradient clipping 1.0; eight workers per process; no AMP; maximum history 200.
- Dense AdamW: lr 1e-4, weight decay 0.01, betas (0.9, 0.999), eps 1e-8. Sparse ExactAdagrad: lr 0.01.
- One training-data pass per stage; evaluate the checkpoint saved at stage completion.

Batch sizes count **requests**, not candidate positions. Each process uses one GPU. The following are historical main-run settings; configuration limitations are listed below. Shared YAML defaults (128) and run.sh defaults (512) do not replace this table.

| Model | Current configuration | RecAgora-60M batch × processes | RecAgora-270M batch × processes |
| --- | --- | ---: | ---: |
| DIN | [din.yaml](../../configs/models/din/din.yaml) | 512 × 1 | 512 × 2 |
| DLRM | [dlrm.yaml](../../configs/models/dlrm/dlrm.yaml) | 512 × 1 | 512 × 2 |
| DCNv2 | [dcnv2.yaml](../../configs/models/dcn/dcnv2.yaml) | 512 × 1 | 512 × 2 |
| DIN + DCNv2 | [din_dcnv2.yaml](../../configs/models/din_dcnv2/din_dcnv2.yaml) | 512 × 2 | 512 × 2 |
| Wukong | [wukong.yaml](../../configs/models/wukong/wukong.yaml) | 512 × 3 | 512 × 2 |
| HiFormer | [hiformer.yaml](../../configs/models/hiformer/hiformer.yaml) | 512 × 2 | 512 × 2 |
| RankMixer | [rankmixer.yaml](../../configs/models/rankmixer/rankmixer.yaml) | 512 × 4 | 512 × 2 |
| Zenith | [zenith.yaml](../../configs/models/zenith/zenith.yaml) | 512 × 3 | 512 × 2 |
| TokenMixer | [tokenmixer.yaml](../../configs/models/tokenmixer/tokenmixer.yaml) | 512 × 2 | 192 × 2 |
| HeMix | [hemix.yaml](../../configs/models/hemix/hemix.yaml) | 512 × 2 | 512 × 2 |
| UniMixer | [unimixer.yaml](../../configs/models/unimixer/unimixer.yaml) | 512 × 3 | 512 × 2 |
| SSR-D | [ssr_d.yaml](../../configs/models/ssr_d/ssr_d.yaml) | 512 × 2 | 512 × 2 |
| HSTU | [hstu.yaml](../../configs/models/hstu/hstu.yaml) | 512 × 2 | 512 × 2 |
| LONGER | [longer.yaml](../../configs/models/longer/longer.yaml) | 512 × 2 | 512 × 2 |
| MTGR | [mtgr.yaml](../../configs/models/mtgr/mtgr.yaml) | 512 × 2 | 512 × 2 |
| INFNet | [infnet.yaml](../../configs/models/infnet/infnet.yaml) | 512 × 3 | 512 × 2 |
| OneTrans | [onetrans.yaml](../../configs/models/onetrans/onetrans.yaml) | 512 × 2 | 512 × 2 |
| HyFormer | [hyformer.yaml](../../configs/models/hyformer/hyformer.yaml) | 512 × 2 | 512 × 2 |
| EST | [est.yaml](../../configs/models/est/est.yaml) | 512 × 2 | 512 × 2 |
| MixFormer | [mixformer.yaml](../../configs/models/mixformer/mixformer.yaml) | 256 × 2 | 192 × 2 |
| SORT | [sort.yaml](../../configs/models/sort/sort.yaml) | 512 × 3 | 512 × 2 |
| OneRank V1 | [onerank_v1.yaml](../../configs/models/onerank/onerank_v1.yaml) | 128 × 4 | 128 × 4 |
| OneRank V2 | [onerank_v2.yaml](../../configs/models/onerank/onerank_v2.yaml) | 512 × 2 | 512 × 2 |

### Configuration coverage

| Item | Coverage of this checkout |
| --- | --- |
| INFNet | Keeps the Large/exposed-only structure (3 layers, 4 feature hubs). Historical Small/HotUser used 4 layers and 16 hubs; their scores do not correspond to the current default. |
| OneRank V2 Large | Appendix B.2 describes next-event training, but the audited final log and checkpoint record posttrain ranking. onerank_v2.yaml is ranking; onerank_v2_pretrain.yaml is pretraining. Resolve this manuscript/run discrepancy before claiming exact reproduction. |
| RankMixer | Keeps the Large main structure: width 192, 2 layers, masked-mean history. Old 100M variants are excluded. |
| History ablations | Figure 3/Table B.1 use lengths 10/20/50/100/200. Dedicated ablation YAMLs are not included; history/time alignment and layer keep schedules must remain consistent when changing length. |
| Release artifacts | See [Hugging Face](https://huggingface.co/datasets/PIIR/RecAgora) for available data. Checkpoints, unrounded metrics, and acceptance tolerances remain pending. Toy metrics cannot verify manuscript scores. |

HSTU is a ranking adaptation and TokenMixer a dense adaptation. Model-specific auxiliary/listwise objectives do not enter evaluation Logloss. See [paper scores](metrics.md#paper-results).

## Launch

```bash
export WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
bash run.sh --dataset small --exp rankmixer_small \
  --config configs/models/rankmixer/rankmixer.yaml \
  --data-root /path/to/small --batch 512
```

run.sh starts one GPU process per node; WORLD_SIZE/RANK map to torchrun's nnodes/node_rank. For single-node multi-GPU jobs, use the Python CLI:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/models/rankmixer/rankmixer.yaml \
  --data-config configs/data/large.yaml \
  --data-path '/path/to/large/dt=DAY01/*.parquet' \
  --output-dir ./results/rankmixer --device cuda --batch-size 512 --no-amp
```

Repeat --data-path to select the full training split; this example reads only DAY01. For multi-node launches, configure nnodes, node_rank, master_addr, and master_port. See [data usage](../dataset/usage.md).

## Evaluate

```bash
torchrun --standalone --nproc_per_node=2 scripts/eval.py \
  --config configs/models/rankmixer/rankmixer.yaml \
  --data-config configs/data/large.yaml \
  --data-path '/path/to/large/dt=DAY35/*.parquet' \
  --checkpoint ./results/rankmixer/checkpoints/last \
  --output-dir ./results/rankmixer/eval \
  --device cuda --batch-size 512 --no-amp --global-eval
```

Results are written to outputs/eval_metrics.json. --global-eval enables exact global AUC; request AUC is a different metric. Check actual requests, valid_candidates, masked_candidates, and finite losses before comparing scores. Sharding, complete-global-batch handling, and --max-batches can affect coverage.

## Checkpoints

- **last/**: complete model-only DCP directory for evaluation and --init-from; retain all files and metadata.
- **step_N/**: full resume state, including optimizer and data position. Python CLI uses --resume-from; run.sh uses --resume.
- --init-from and resume are mutually exclusive. Initialization loads weights without optimizer state.
- Strict resume requires compatible input semantics and world size. --resume-world-size-policy elastic is an explicit opt-in; it does not make arbitrary data or architecture changes valid.
- Completed training may remove retained step directories, leaving last/. Do not expect a model-only checkpoint to resume optimizer state.

### OneRank V2

For a two-stage workflow, train onerank_v2_pretrain.yaml, then initialize onerank_v2.yaml with --init-from /path/to/pretrain/checkpoints/last. The stage must match the target experiment; this generic workflow is not a resolution of the Large manuscript discrepancy above.

## Configuration and records

Model YAML selects features, dimensions, architecture parameters, and data/training includes. CLI --data-config overrides the data entry. To change seed or optimizer settings, copy the training YAML and reference it from the model; there is no train --seed flag.

Record the code revision, environment/hardware, complete configurations, data/vocabulary versions and checksums, seed, scope, batch/process count, initialization, stage, and checkpoint. Preserve evaluation counts and unrounded metrics. Independent training seeds are required for variance estimates; repeated evaluation of one checkpoint is not repeated training.

Use `python scripts/train.py --help`, `python scripts/eval.py --help`, and `bash run.sh --help` for complete options. Training writes logs/train.log, outputs/metrics.jsonl, outputs/epoch_metrics.jsonl, and checkpoint directories. Optional timing/profiling settings are defined in the training YAML.
