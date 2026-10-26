**List diagnostics and utility** (3,867 users; HR/P/NDCG@10 on the 695 users with test items)

| Model | Prompt | List version | Match rate | Mean length | <5 items | Already watched (raw) | Distinct items | Top-50 share | HR@10 | NDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-4-9B | neutral | raw | 0.90 | 9.0 | 0.0% | 29.7% | 512 | 87.8% | 0.378 | 0.084 |
| GLM-4-9B | neutral | unseen | 0.90 | 6.3 | 23.2% | 29.7% | 347 | 89.2% | 0.380 | 0.096 |
| GLM-4-9B | sensitive | raw | 0.90 | 9.0 | 0.1% | 30.9% | 986 | 80.4% | 0.378 | 0.078 |
| GLM-4-9B | sensitive | unseen | 0.90 | 6.3 | 24.4% | 30.9% | 460 | 83.5% | 0.384 | 0.095 |
| Llama-3.1-8B | neutral | raw | 0.82 | 8.4 | 2.5% | 25.3% | 909 | 49.5% | 0.332 | 0.066 |
| Llama-3.1-8B | neutral | unseen | 0.82 | 6.3 | 23.4% | 25.3% | 859 | 49.1% | 0.332 | 0.073 |
| Llama-3.1-8B | sensitive | raw | 0.82 | 8.4 | 3.1% | 27.0% | 895 | 50.7% | 0.354 | 0.072 |
| Llama-3.1-8B | sensitive | unseen | 0.82 | 6.1 | 26.4% | 27.0% | 832 | 50.5% | 0.354 | 0.081 |

**Gender AUC, unseen lists (train items removed, comparable with EXP-003/004)** — LR balanced (best of the 9 classifier variants in brackets); 500 fixed test users, 3,367-user train pool

| Tier | Input | GLM-4-9B neutral | GLM-4-9B sensitive | Llama-3.1-8B neutral | Llama-3.1-8B sensitive |
| --- | --- | ---: | ---: | ---: | ---: |
| TH | Full train history (reference) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) |
| TH-prompt | Items shown in the LLMRec prompt (reference) | 0.716 (0.726) | 0.716 (0.726) | 0.716 (0.726) | 0.716 (0.726) |
| T2 | Multi-hot over list items | 0.638 (0.639) | 0.916 (0.925) | 0.647 (0.699) | 0.810 (0.817) |
| T2 | TF-IDF multi-hot | 0.646 (0.651) | 0.921 (0.923) | 0.680 (0.705) | 0.814 (0.816) |
| T3 | Genre histogram (18-d) | 0.578 (0.608) | 0.836 (0.838) | 0.688 (0.689) | 0.748 (0.764) |
| T3 | Title text embedding (MiniLM) | 0.640 (0.640) | 0.910 (0.914) | 0.630 (0.690) | 0.801 (0.813) |
| T4 | GLM-4-9B attacker on the titles (zero-shot / 8-shot staab) | 0.544 / 0.570 | 0.820 / 0.810 | 0.670 / 0.676 | 0.787 / 0.737 |

**Gender AUC, raw lists (Stairway's output as is)** — LR balanced (best of the 9 classifier variants in brackets); 500 fixed test users, 3,367-user train pool

| Tier | Input | GLM-4-9B neutral | GLM-4-9B sensitive | Llama-3.1-8B neutral | Llama-3.1-8B sensitive |
| --- | --- | ---: | ---: | ---: | ---: |
| TH | Full train history (reference) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) |
| TH-prompt | Items shown in the LLMRec prompt (reference) | 0.716 (0.726) | 0.716 (0.726) | 0.716 (0.726) | 0.716 (0.726) |
| T2 | Multi-hot over list items | 0.606 (0.626) | 0.927 (0.939) | 0.637 (0.709) | 0.840 (0.849) |
| T2 | TF-IDF multi-hot | 0.618 (0.632) | 0.934 (0.938) | 0.678 (0.711) | 0.849 (0.851) |
| T3 | Genre histogram (18-d) | 0.616 (0.620) | 0.854 (0.860) | 0.703 (0.704) | 0.796 (0.813) |
| T3 | Title text embedding (MiniLM) | 0.606 (0.622) | 0.894 (0.935) | 0.602 (0.693) | 0.824 (0.838) |

**Neutral → sensitive, paired on the same 500 users** (LR balanced; ΔAUC with 95% paired-bootstrap CI) and how much the list content moves (Jaccard between a user's two lists, all 3,867 users)

| Model | List version | Multi-hot ΔAUC | TF-IDF ΔAUC | Genre ΔAUC | MiniLM ΔAUC | Jaccard male | Jaccard female | Romance share Δ (F / M) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-4-9B | unseen | 0.638→0.916 (+0.278 [+0.227, +0.332]) | 0.646→0.921 (+0.275 [+0.225, +0.326]) | 0.578→0.836 (+0.258 [+0.203, +0.317]) | 0.640→0.910 (+0.270 [+0.217, +0.326]) | 0.42 | 0.27 | +0.081 / -0.010 |
| GLM-4-9B | raw | 0.606→0.927 (+0.321 [+0.272, +0.372]) | 0.618→0.934 (+0.316 [+0.268, +0.367]) | 0.616→0.854 (+0.237 [+0.183, +0.293]) | 0.606→0.894 (+0.287 [+0.225, +0.345]) | 0.44 | 0.29 | +0.077 / -0.008 |
| Llama-3.1-8B | unseen | 0.647→0.810 (+0.163 [+0.114, +0.217]) | 0.680→0.814 (+0.135 [+0.087, +0.185]) | 0.688→0.748 (+0.060 [+0.017, +0.103]) | 0.630→0.801 (+0.171 [+0.117, +0.229]) | 0.35 | 0.35 | +0.029 / -0.013 |
| Llama-3.1-8B | raw | 0.637→0.840 (+0.203 [+0.152, +0.261]) | 0.678→0.849 (+0.170 [+0.128, +0.217]) | 0.703→0.796 (+0.093 [+0.055, +0.133]) | 0.602→0.824 (+0.222 [+0.166, +0.282]) | 0.37 | 0.36 | +0.029 / -0.013 |
