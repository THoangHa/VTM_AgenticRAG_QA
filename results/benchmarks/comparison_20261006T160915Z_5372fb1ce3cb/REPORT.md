# Dense validation comparison

| Model | nDCG@10 | Recall@10 | MRR@10 | Recall@100 | p50 latency (ms) |
|---|---:|---:|---:|---:|---:|
| bge_m3 | 0.717628 | 0.826840 | 0.682174 | 0.913420 | 31.43 |
| vi_bi_encoder | 0.623070 | 0.733045 | 0.587552 | 0.841270 | 25.85 |

| Model | Index build (s) | Batched queries/s | Peak allocated CUDA (MiB) | Truncated passages | Truncated queries |
|---|---:|---:|---:|---:|---:|
| bge_m3 | 60.99 | 97.83 | 2654.22 | 41.00 | 0.00 |
| vi_bi_encoder | 37.31 | 116.19 | 618.60 | 343.00 | 0.00 |

Selected baseline: **bge_m3**. Higher unrounded NDCG@10; uncertainty is reported separately.

BGE-M3 minus Vietnamese encoder nDCG@10: 0.094558; paired 95% CI [0.07143379824843538, 0.1178180407159929]. Conclusion: a_higher.

Selection uses validation and fixed 512/256 token limits. Labels credit the original source passage only.
Bootstrap resamples questions; questions sharing source passages may be correlated. No test questions were evaluated.
