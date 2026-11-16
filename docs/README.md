# 📚 Documentation

| Topic | Guide |
| --- | --- |
| Installation and Toy run | [Quick Start](../README.md#quick-start) |
| Dataset scale and sampling | [Dataset overview](dataset/README.md) |
| Fields, IDs, and history alignment | [Feature reference](dataset/features.md) |
| DAY partitions and HotUser | [Data usage](dataset/usage.md) |
| Paper protocol and configuration differences | [Reproduction](guides/train-and-evaluate.md#paper-reproduction) |
| Metrics and manuscript scores | [Evaluation](guides/metrics.md) |
| Model configurations | [Model list](rank-model-list.md) |
| Model integration | [Guide](guides/add-a-model.md) and [Python API](public-api.md) |
| Runtime issues | [Troubleshooting](guides/troubleshooting.md) |

## Release resources

This checkout includes model implementations, 47 model configurations, two training protocols, public data contracts, and a Toy sample. Dataset downloads are hosted on [Hugging Face: PIIR/RecAgora](https://huggingface.co/datasets/PIIR/RecAgora); consult that repository for currently available files. The following release artifacts remain pending:

- Full-data split manifests, verified row counts, vocabulary versions, and checksums.
- Complete checkpoints with matching configurations and code revisions.
- Unrounded reference metrics, observed evaluation counts, and reproduction tolerances.
- A validated software/hardware matrix and memory requirements.
- Stable paper metadata, citation, and separate code/data/checkpoint licenses.

The supplied manuscript is anonymous. Template conference and DOI fields are not publication metadata. Its document license does not grant a software license. Historical configuration differences are listed in the [reproduction notes](guides/train-and-evaluate.md#configuration-coverage).
