# 🗂️ Datasets

RecAgora provides two e-commerce search datasets. Counts below are reported in manuscript Table 2, not measured on the bundled Toy sample.

| Statistic | RecAgora-60M (`small`) | RecAgora-270M (`large`) |
| --- | ---: | ---: |
| Training / evaluation days | 7 / 1 | 34 / 1 |
| Training requests | 52,901,479 | 261,792,919 |
| Evaluation requests | 7,288,725 | 7,485,173 |
| Total requests | 60,190,204 | 269,278,092 |
| Unique users | 7,307,159 | 10,761,415 |
| Unique candidate items | 81,510,502 | 115,406,694 |
| Candidate slots | 1,444,564,896 | 6,462,674,208 |

User/item counts are deduplicated across training and evaluation; history-only items are excluded. The two versions use separate time windows and vocabularies. Visit [PIIR/RecAgora on Hugging Face](https://huggingface.co/datasets/PIIR/RecAgora) for dataset downloads and currently available files. See [release resources](../README.md#release-resources) for outstanding release metadata.

## Construction and labels

The manuscript retains requests with click positives, keeps at most six positives, and samples exposed negatives and unexposed candidates from the same request to fill 24 positions. Excess positives are selected in order > add-to-cart > click priority, randomly within each level. This filtered, sampled dataset is not an unbiased record of all online traffic.

Each candidate has aligned click, add-to-cart, and order labels. The manuscript describes `order ⊆ atc ⊆ click`. Unexposed candidates have zero labels because feedback was not observed, not because negative preference was established. Main results use all 24 positions; exposed-only experiments mask unexposed positions in both training and evaluation.

## Features and schema

The 160 fields comprise 23 query, 46 item, 22 query–item, 39 user–item, and 30 history fields. Physically, these form 23 query, 107 candidate, and 30 history fields. Of these, 158 are discrete IDs in 131 semantic namespaces; two history time fields remain continuous. Histories contain up to 200 aligned events, oldest first.

Public Parquet contracts contain **169 columns**: request_id, three labels, five source flags, and 160 feature columns. Top-level user_id/item_id are absent; item identifiers inside features remain. The reader fills batch.item_id with candidate positions for alignment only. These positions cannot support entity-level statistics.

## Profiles and splits

| Setting | Data entry | Vocabulary contract | Training / evaluation |
| --- | --- | --- | --- |
| RecAgora-60M | [small.yaml](../../configs/data/small.yaml) | dataset_small.yaml | DAY01–DAY07 / DAY08 |
| RecAgora-270M | [large.yaml](../../configs/data/large.yaml) | dataset_large.yaml | DAY01–DAY34 / DAY35 |
| HotUser | [hotuser.yaml](../../configs/data/hotuser.yaml) | dataset_large.yaml | train / valid |

All profiles share [features.yaml](../../configs/data/features.yaml). Small/Large embedding rows are not interchangeable.

**HotUser** is an active-user subset of Large, retaining users active on at least 20 of 34 days. It shares Large's feature schema and ID vocabulary and supports analysis of ranking performance for active users. It contains only two splits: `train` for training and `valid` for evaluation.

| Statistic | HotUser |
| --- | ---: |
| Training requests (`train`) | 56,554,645 |
| Evaluation requests (`valid`) | 7,069,331 |
| Total requests | 63,623,976 |

Counts are recorded from existing split metadata and have not been reverified against the current Hugging Face release.

See [field definitions](features.md), [loading examples](usage.md), and [reproduction notes](../guides/train-and-evaluate.md#paper-reproduction). Follow the final data-use terms when released; do not attempt re-identification. Offline scores should not be interpreted as online business gains.
