# Models

The manuscript evaluates 23 models and implementation variants; OneRank V1/V2 count separately. These are benchmark adaptations, including HSTU-style ranking and dense TokenMixer, not exact replicas of upstream production systems. Checkpoints are [pending](README.md#release-resources).

| Category | Models |
| --- | --- |
| Feature and Interest Modeling | DIN, DLRM, DCNv2, DIN + DCNv2, Wukong |
| Stacked Unified Modeling | HiFormer, RankMixer, Zenith, TokenMixer, HeMix, UniMixer, SSR-D |
| Joint Sequence–Feature Modeling | HSTU, LONGER, MTGR, INFNet, OneTrans, HyFormer, EST, MixFormer, SORT |
| Request-level Joint Ranking | OneRank V1, OneRank V2 |

| Model | Configuration | Selected input fields | Reference |
| --- | --- | --- | --- |
| DCNv2 | [`dcnv2.yaml`](../configs/models/dcn/dcnv2.yaml) | 158: action | [DCN V2](https://arxiv.org/abs/2008.13535) |
| DIN | [`din.yaml`](../configs/models/din/din.yaml) | 158: action | [DIN](https://arxiv.org/abs/1706.06978) |
| DIN-DCNv2 | [`din_dcnv2.yaml`](../configs/models/din_dcnv2/din_dcnv2.yaml) | 158: action | [DIN](https://arxiv.org/abs/1706.06978) / [DCN V2](https://arxiv.org/abs/2008.13535) |
| DLRM | [`dlrm.yaml`](../configs/models/dlrm/dlrm.yaml) | 158: action | [DLRM](https://arxiv.org/abs/1906.00091) |
| Wukong | [`wukong.yaml`](../configs/models/wukong/wukong.yaml) | 158: action | [Wukong](https://arxiv.org/abs/2403.02545) |
| RankMixer | [`rankmixer.yaml`](../configs/models/rankmixer/rankmixer.yaml) | 158: action | [RankMixer](https://arxiv.org/abs/2507.15551) |
| OneTrans | [`onetrans.yaml`](../configs/models/onetrans/onetrans.yaml) | 160: action + timestamp + gap | [OneTrans](https://arxiv.org/abs/2510.26104) |
| OneRank V1 | [`onerank_v1.yaml`](../configs/models/onerank/onerank_v1.yaml) | 158: action | [OneRank](https://arxiv.org/abs/2606.16838) |
| SORT | [`sort.yaml`](../configs/models/sort/sort.yaml) | 160: action + timestamp + gap | [SORT](https://arxiv.org/abs/2603.03988) |
| Zenith | [`zenith.yaml`](../configs/models/zenith/zenith.yaml) | 158: action | [Zenith](https://arxiv.org/abs/2601.21285) |
| HSTU-style | [`hstu.yaml`](../configs/models/hstu/hstu.yaml) | 160: action + timestamp + gap | [HSTU](https://arxiv.org/abs/2402.17152) |
| MTGR | [`mtgr.yaml`](../configs/models/mtgr/mtgr.yaml) | 160: action + timestamp + gap | [MTGR](https://arxiv.org/abs/2505.18654) |
| MixFormer | [`mixformer.yaml`](../configs/models/mixformer/mixformer.yaml) | 159: action + timestamp | [MixFormer](https://arxiv.org/abs/2602.14110) |
| SSR-D | [`ssr_d.yaml`](../configs/models/ssr_d/ssr_d.yaml) | 158: action | [SSR](https://arxiv.org/abs/2604.08011) |
| EST-LCA | [`est.yaml`](../configs/models/est/est.yaml) | 158: action | [EST](https://arxiv.org/abs/2602.10811) |
| TokenMixer | [`tokenmixer.yaml`](../configs/models/tokenmixer/tokenmixer.yaml) | 158: action | [TokenMixer-Large](https://arxiv.org/abs/2602.06563) |
| HyFormer | [`hyformer.yaml`](../configs/models/hyformer/hyformer.yaml) | 160: action + timestamp + gap | [HyFormer](https://arxiv.org/abs/2601.12681) |
| HiFormer | [`hiformer.yaml`](../configs/models/hiformer/hiformer.yaml) | 158: action | [Hiformer](https://arxiv.org/abs/2311.05884) |
| HeMix | [`hemix.yaml`](../configs/models/hemix/hemix.yaml) | 158: action | [HeMix](https://arxiv.org/abs/2602.09387) |
| LONGER | [`longer.yaml`](../configs/models/longer/longer.yaml) | 158: action | [LONGER](https://arxiv.org/abs/2505.04421) |
| INFNet | [`infnet.yaml`](../configs/models/infnet/infnet.yaml) | 160: action + timestamp + gap | [Implementation](../models/infnet/model.py) |
| UniMixer | [`unimixer.yaml`](../configs/models/unimixer/unimixer.yaml) | 158: action | [Implementation](../models/unimixer/model.py) |
| OneRank V2 | [`onerank_v2.yaml`](../configs/models/onerank/onerank_v2.yaml) | 159: action + gap | [Implementation](../models/onerank/v2.py); see stage notes |

Each main configuration has a corresponding *_negmask.yaml for exposed-only training/evaluation. Main configurations use configs/train/adamw.yaml; masked variants use adamw_negmask.yaml. The mask applies to candidates, not history padding.

RankMixer retains the Large main structure (width 192, two layers, masked mean). INFNet retains the Large/exposed-only structure (three layers, four feature hubs), not the historical Small/HotUser structure. See [paper coverage and batch settings](guides/train-and-evaluate.md#paper-reproduction).

## OneRank V2

A two-stage workflow uses [pretraining](../configs/models/onerank/onerank_v2_pretrain.yaml), then [ranking](../configs/models/onerank/onerank_v2.yaml) initialized with --init-from. Both select 159 fields, including action and time gap. A matching ranking checkpoint can be evaluated directly. The manuscript's Large objective differs from the audited final checkpoint; resolve this before claiming exact table reproduction.
