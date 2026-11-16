# Feature reference

Each Parquet row is one request: **9 metadata/label columns + 160 feature columns = 169 columns**. Models select subsets through model.features. Full physical types, namespaces, and vocabulary sizes are defined in [features.yaml](../../configs/data/features.yaml), [dataset_small.yaml](../../configs/data/dataset_small.yaml), and [dataset_large.yaml](../../configs/data/dataset_large.yaml).

## Layout

| View | Fields | Per-row Arrow layout |
| --- | ---: | --- |
| Query | 23 | list<int64>; shared across candidates |
| Candidate | 107 | list<list<int64>>; outer length 24 |
| Categorical history | 28 | list<int64>; one aligned event axis |
| Direct history time | 2 | list<double>; physical length 200 |

B denotes requests, L=24 candidates, H valid history events, and D embedding width. Candidate position i aligns across all candidate fields, labels, and source flags. Arrow list types alone do not imply fixed lengths. Slot IDs identify fields, not event positions.

## Base fields

| Field | Arrow type | Meaning |
| --- | --- | --- |
| request_id | int64 | Request identifier |
| click / atc / order | list<int32> | Candidate-aligned binary labels |
| negative_impression | list<int32> | Exposed-negative flag |
| negative_non_impression | list<int32> | Unexposed-negative flag |
| negative_from_rank | list<int32> | Ranking-source negative flag |
| negative_from_recall | list<int32> | Retrieval-source negative flag |
| negative_from_random | list<int32> | Random-source negative flag |

Source flags are not the complements of behavior labels. The manuscript describes order ⊆ atc ⊆ click. Unexposed zero labels indicate unobserved feedback, not established negative preference. Top-level user_id/item_id are absent; batch.item_id contains alignment positions only. Item-ID feature slots remain present.

## History alignment

- 45041 anchors the event count; all categorical fields share that count. Keep the most recent tail when truncating to 200.
- Events are oldest-to-newest. Left padding does not reverse them.
- 45042 is the event timestamp; 45073 is the inter-event gap. Physical arrays contain 200 values: the first H are valid and the rest are zero padding.
- Direct time is read as right-padded float32[B,200]. Models using left padding must shift time and masks with the categorical events.
- Use lengths, not time != 0, to determine validity. Empty histories are valid; valid time values must be finite. Units/origins require release metadata.
- Action slot 45040 has one categorical code per event: 0 padding, 1 OOV, 2 order, 3 click, 4 add-to-cart. It is not a multi-hot vector.

## IDs and shared namespaces

158 discrete fields use 131 namespaces; 24 namespaces are shared by multiple slots. Shared slots use the same mapping and embedding table. For example, namespace 5000 is shared by item slot 5000, clicked-item slots 8115/8127, and history-item slot 45041. Namespace 5052 covers shop slots 5052/3096/45043. The YAML contracts specify every binding.

ID 0 is padding, 1 is OOV, and IDs >=2 are mapped values. Missing values, empty lists, padding, and OOV are distinct. IDs are comparable only within their namespace and data version. Small/Large vocabularies are independent; neither embedding rows nor weights can be swapped directly.

Bucket/category IDs represent discrete bins or states, not raw quantities or ordered bin numbers. Tag IDs encode categories/attributes; entity/token IDs encode identifiers and sequences. Only the two direct time fields retain continuous numeric meaning. Bin boundaries and mapping metadata must come from the matching data release.

## Complete slot dictionary

The table retains all 160 fields. Scalar/list below describes the logical value; physical Arrow layouts follow the view table above. Counts, prices, rates, and intervals marked Bucket/category are discretized, not raw values.

| Slot | ID type | Namespace | Logical layout | Meaning |
| --- | --- | --- | --- | --- |
| `5000` | Entity/token | `5000` | Candidate scalar | Candidate item ID; retained in public features |
| `5052` | Entity/token | `5052` | Candidate scalar | Shop ID |
| `5053` | Entity/token | `5053` | Candidate scalar | Seller ID |
| `5054` | Tag | `5054` | Candidate scalar | Global primary category ID |
| `5055` | Tag | `5055` | Candidate scalar | Global business category ID |
| `5056` | Tag | `5056` | Candidate scalar | Level-1 global business category ID |
| `5057` | Tag | `5057` | Candidate scalar | Level-2 global business category ID |
| `5058` | Tag | `5058` | Candidate scalar | Level-3 global business category ID |
| `5075` | Tag | `5075` | Candidate scalar | Category ID |
| `5077` | Tag | `5077` | Candidate scalar | Brand ID |
| `5078` | Tag | `5078` | Candidate scalar | Level-3 category, falling back to level 2 |
| `5059` | Entity/token | `3002` | Candidate list | Title tokens |
| `5073` | Entity/token | `5073` | Candidate list | Alternative published title-token field |
| `5272` | Tag | `5272` | Candidate list | Global brand/tag sequence |
| `5273` | Tag | `5273` | Candidate list | Color tags |
| `5274` | Tag | `5274` | Candidate list | Product named-entity tags |
| `5275` | Entity/token | `5275` | Candidate list | Shop-name tokens |
| `11000` | Tag | `11000` | Candidate list | Retrieval types associated with the candidate |
| `5004` | Bucket/category | `5004` | Candidate scalar | 7-day platform-wide paying-user count |
| `5005` | Bucket/category | `5005` | Candidate scalar | 14-day platform-wide paying-user count |
| `5006` | Bucket/category | `5006` | Candidate scalar | 30-day platform-wide paying-user count |
| `5021` | Bucket/category | `5021` | Candidate scalar | 30-day search GMV |
| `5022` | Bucket/category | `5022` | Candidate scalar | 7-day search item page views (IPV) |
| `5023` | Bucket/category | `5023` | Candidate scalar | 14-day search IPV |
| `5024` | Bucket/category | `5024` | Candidate scalar | 30-day search IPV |
| `5031` | Bucket/category | `5031` | Candidate scalar | 7-day search CTR |
| `5032` | Bucket/category | `5032` | Candidate scalar | 14-day search CTR |
| `5033` | Bucket/category | `5033` | Candidate scalar | 30-day search CTR |
| `5034` | Bucket/category | `5034` | Candidate scalar | 7-day search CVR |
| `5036` | Bucket/category | `5036` | Candidate scalar | 30-day search CVR |
| `5060` | Bucket/category | `5060` | Candidate scalar | Browsing price (USD) |
| `5061` | Bucket/category | `5061` | Candidate scalar | Average paid price (USD) |
| `5062` | Bucket/category | `5062` | Candidate scalar | Minimum paid price (USD) |
| `5085` | Bucket/category | `5085` | Candidate scalar | Current price (USD) |
| `5086` | Bucket/category | `5086` | Candidate scalar | Average price (USD) |
| `5087` | Bucket/category | `5087` | Candidate scalar | Minimum price (USD) |
| `5088` | Bucket/category | `5088` | Candidate scalar | Maximum price (USD) |
| `5268` | Bucket/category | `5268` | Candidate scalar | Cumulative rating count |
| `5269` | Bucket/category | `5269` | Candidate scalar | 7-day conversion rate |
| `5270` | Bucket/category | `5270` | Candidate scalar | 3-day click rate |
| `5271` | Bucket/category | `5271` | Candidate scalar | 3-day clicking-user count |
| `6270` | Bucket/category | `6270` | Candidate scalar | Combined real-time status and price |
| `6287` | Bucket/category | `6287` | Candidate scalar | Combined real-time status and price difference (USD) |
| `6288` | Bucket/category | `6288` | Candidate scalar | Combined real-time status and average-price difference (USD) |
| `6289` | Bucket/category | `6289` | Candidate scalar | Combined real-time status and minimum-price difference (USD) |
| `6347` | Bucket/category | `6347` | Candidate scalar | Versioned real-time status/price combination (USD) |
| `3000` | Entity/token | `3000` | Query scalar | Primary query identifier |
| `3001` | Entity/token | `3001` | Query scalar | Query hash identifier |
| `3002` | Entity/token | `3002` | Request sequence | Query tokens |
| `3046` | Bucket/category | `3046` | Query scalar | 14-day query unique visitors |
| `3062` | Bucket/category | `3062` | Query scalar | 30-day query orders |
| `3063` | Bucket/category | `3063` | Query scalar | 15-day query orders |
| `3069` | Bucket/category | `3069` | Query scalar | 30-day query clicks |
| `3070` | Bucket/category | `3070` | Query scalar | 15-day query clicks |
| `3071` | Bucket/category | `3071` | Query scalar | 7-day query clicks |
| `3072` | Bucket/category | `3072` | Query scalar | 3-day query clicks |
| `3073` | Bucket/category | `3073` | Query scalar | 30-day query page views |
| `3074` | Bucket/category | `3074` | Query scalar | 15-day query page views |
| `3075` | Bucket/category | `3075` | Query scalar | 7-day query page views |
| `3077` | Bucket/category | `3077` | Query scalar | 30-day query impression depth |
| `3080` | Bucket/category | `3080` | Query scalar | 30-day query click depth |
| `3086` | Tag | `3086` | Request sequence | Predicted query level-3/level-2 category keys |
| `3087` | Entity/token | `3087` | Request sequence | Query slices |
| `3088` | Entity/token | `3088` | Request sequence | Query–item click keys |
| `3089` | Entity/token | `3089` | Request sequence | Query–item order keys |
| `3090` | Entity/token | `3090` | Request sequence | Query–shop click keys |
| `3091` | Entity/token | `3091` | Request sequence | Query–shop order keys |
| `3096` | Entity/token | `5052` | Request sequence | Query–shop click keys in the shop namespace |
| `3100` | Entity/token | `5053` | Request sequence | Query-associated top-K seller IDs |
| `9007` | Bucket/category | `9007` | Candidate scalar | 7-day query–item search CTR |
| `9009` | Bucket/category | `9009` | Candidate scalar | 14-day query–item search page views |
| `9010` | Bucket/category | `9010` | Candidate scalar | 14-day query–item search IPV |
| `9011` | Bucket/category | `9011` | Candidate scalar | 14-day query–item search add-to-cart count |
| `9016` | Bucket/category | `9016` | Candidate scalar | 14-day query–item search CTR |
| `9028` | Bucket/category | `9028` | Candidate scalar | Predicted query-category/candidate-category match |
| `9029` | Bucket/category | `9029` | Candidate scalar | Discrete query–item click relationship |
| `9044` | Entity/token | `9044` | Candidate list | Query-slice/candidate-category cross sequence |
| `9045` | Entity/token | `9045` | Candidate list | Query-slice/candidate level-1 category cross sequence |
| `9046` | Entity/token | `9046` | Candidate list | Query-slice/candidate-brand cross sequence |
| `9048` | Entity/token | `9048` | Candidate list | Query category-click key/candidate cross sequence |
| `9049` | Entity/token | `9049` | Candidate list | Query category-click key/candidate-category cross sequence |
| `9072` | Entity/token | `9072` | Candidate list | Query-slice/candidate-title-token cross sequence |
| `9073` | Bucket/category | `9073` | Candidate scalar | Query-slice/candidate-title association count |
| `43051` | Bucket/category | `43051` | Candidate scalar | 30-day query–item search IPV |
| `43052` | Bucket/category | `43052` | Candidate scalar | 30-day query–item search CTR |
| `43053` | Bucket/category | `43053` | Candidate scalar | Maximum item price under the query (USD) |
| `43054` | Bucket/category | `43054` | Candidate scalar | 30-day query–item search page views |
| `43055` | Bucket/category | `43055` | Candidate scalar | 30-day query–item paying-user count |
| `43056` | Bucket/category | `43056` | Candidate scalar | Query–item average paid price (USD) |
| `43057` | Bucket/category | `43057` | Candidate scalar | 90-day query–item completed orders |
| `43058` | Bucket/category | `43058` | Candidate scalar | 90-day query–item canceled orders |
| `7090` | Bucket/category | `7090` | Candidate scalar | User age/candidate-price combination |
| `7091` | Bucket/category | `7091` | Candidate scalar | User paid GMV/candidate-price combination |
| `7092` | Bucket/category | `7092` | Candidate scalar | User paid-order count/candidate-price combination |
| `7097` | Bucket/category | `7097` | Candidate scalar | Long-term user/level-1-category GMV relationship |
| `7098` | Tag | `7098` | Candidate scalar | User/level-1-category count and current-category combination |
| `7099` | Tag | `7099` | Candidate scalar | User–item categorical interaction feature |
| `7100` | Tag | `7100` | Candidate scalar | User–item categorical interaction feature |
| `7102` | Tag | `7102` | Candidate scalar | User–item categorical interaction feature |
| `8000` | Bucket/category | `8000` | Candidate scalar | Time since user's last click on this item |
| `8001` | Bucket/category | `8001` | Candidate scalar | Time since user's last click on this shop |
| `8002` | Bucket/category | `8002` | Candidate scalar | Time since user's last click on this level-1 category |
| `8003` | Bucket/category | `8003` | Candidate scalar | Time since user's last click on this category |
| `8004` | Bucket/category | `8004` | Candidate scalar | Time since user's last cart action on an item in this shop |
| `8005` | Bucket/category | `8005` | Candidate scalar | Time since user's last cart action in this level-1 category |
| `8007` | Bucket/category | `8007` | Candidate scalar | Time since user's last order from this shop |
| `8008` | Bucket/category | `8008` | Candidate scalar | Time since user's last order in this level-1 category |
| `8009` | Bucket/category | `8009` | Candidate scalar | Time since user's last order in this category |
| `8010` | Tag | `8010` | Candidate scalar | Last level-1-category click interval/current-category combination |
| `8023` | Bucket/category | `8023` | Candidate scalar | 1-day user/candidate-category click count |
| `8034` | Bucket/category | `8034` | Candidate scalar | 3-hour user/candidate click count |
| `8035` | Bucket/category | `8035` | Candidate scalar | 1-day user/candidate click count |
| `8037` | Bucket/category | `8037` | Candidate scalar | 1-day user/candidate add-to-cart count |
| `8039` | Bucket/category | `8039` | Candidate scalar | 1-day user/candidate order count |
| `8056` | Bucket/category | `8056` | Candidate scalar | User impressions of the candidate's level-1 category |
| `8057` | Bucket/category | `8057` | Candidate scalar | User impressions of the candidate's level-2 category |
| `8058` | Bucket/category | `8058` | Candidate scalar | User impressions of the candidate's level-3 category |
| `8059` | Bucket/category | `8059` | Candidate scalar | Time since last impression of the candidate's level-1 category |
| `8060` | Bucket/category | `8060` | Candidate scalar | Time since last impression of the candidate's level-2 category |
| `8061` | Bucket/category | `8061` | Candidate scalar | Time since last impression of the candidate's level-3 category |
| `8063` | Entity/token | `8063` | Candidate list | Intersection of clicked-item history and candidate i2i retrieval |
| `8085` | Bucket/category | `8085` | Candidate scalar | Size of clicked-history/candidate i2i intersection |
| `8115` | Entity/token | `5000` | Candidate list | Clicked items in the candidate's level-1 category |
| `8127` | Entity/token | `5000` | Candidate list | Clicked items in the candidate's level-2 category |
| `8131` | Bucket/category | `8131` | Candidate scalar | Cart item-hit count |
| `8132` | Bucket/category | `8132` | Candidate scalar | Clicked-item/i2i retrieval intersection count |
| `8192` | Bucket/category | `8192` | Candidate scalar | Long-term user/candidate click count |
| `8193` | Bucket/category | `8193` | Candidate scalar | Long-term user/candidate add-to-cart count |
| `8195` | Bucket/category | `8195` | Candidate scalar | 7-day user/candidate impression count |
| `8196` | Bucket/category | `8196` | Candidate scalar | Time since user's last impression of this candidate |
| `45040` | Bucket/category | `45040` | Request sequence | Action: 0 padding, 1 OOV, 2 order, 3 click, 4 add-to-cart |
| `45041` | Entity/token | `5000` | Request sequence | History item ID and history-length anchor |
| `45042` | Continuous | — | History time | Event timestamp; units defined by release metadata |
| `45043` | Entity/token | `5052` | Request sequence | History item shop ID |
| `45044` | Tag | `5077` | Request sequence | History item brand ID |
| `45045` | Bucket/category | `5090` | Request sequence | History item review count |
| `45046` | Bucket/category | `5240` | Request sequence | History item sales count |
| `45047` | Tag | `5055` | Request sequence | History item global business category ID |
| `45048` | Tag | `5056` | Request sequence | History item level-1 global business category ID |
| `45049` | Tag | `5057` | Request sequence | History item level-2 global business category ID |
| `45051` | Bucket/category | `5060` | Request sequence | History item browsing price (USD) |
| `45054` | Bucket/category | `5005` | Request sequence | History item 14-day platform-wide paying-user count |
| `45055` | Bucket/category | `5006` | Request sequence | History item 30-day platform-wide paying-user count |
| `45056` | Bucket/category | `5022` | Request sequence | History item 7-day search item page views (IPV) |
| `45057` | Bucket/category | `5031` | Request sequence | History item 7-day search CTR |
| `45058` | Bucket/category | `5034` | Request sequence | History item 7-day search CVR |
| `45059` | Bucket/category | `5023` | Request sequence | History item 14-day search IPV |
| `45060` | Bucket/category | `5032` | Request sequence | History item 14-day search CTR |
| `45061` | Bucket/category | `5030` | Request sequence | History item 30-day search page views |
| `45062` | Bucket/category | `5024` | Request sequence | History item 30-day search IPV |
| `45063` | Bucket/category | `5033` | Request sequence | History item 30-day search CTR |
| `45064` | Bucket/category | `5036` | Request sequence | History item 30-day search CVR |
| `45066` | Bucket/category | `5358` | Request sequence | History item like count |
| `45067` | Bucket/category | `5268` | Request sequence | History item cumulative rating count |
| `45068` | Bucket/category | `5271` | Request sequence | History item 3-day clicking-user count |
| `45069` | Bucket/category | `5269` | Request sequence | History item 7-day conversion rate |
| `45070` | Bucket/category | `5270` | Request sequence | History item 3-day click rate |
| `45072` | Bucket/category | `21055` | Request sequence | History item price (USD) |
| `45073` | Continuous | — | History time | Inter-event time gap; units defined by release metadata |
| `45075` | Bucket/category | `5004` | Request sequence | History item 7-day platform-wide paying-user count |
