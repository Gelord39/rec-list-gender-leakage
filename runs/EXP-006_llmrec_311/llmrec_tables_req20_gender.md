**List diagnostics and utility** (3,867 users; HR/P/NDCG@10 on the 695 users with test items)

| Model | Prompt | List version | Match rate | Mean length | <5 items | Already watched (raw) | Distinct items | Top-50 share | HR@10 | NDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-4-9B | neutral | raw | 0.88 | 10.0 | 0.0% | 26.3% | 537 | 88.1% | 0.390 | 0.085 |
| GLM-4-9B | neutral | unseen | 0.88 | 9.1 | 4.8% | 26.3% | 438 | 86.3% | 0.446 | 0.117 |
| GLM-4-9B | gender | raw | 0.88 | 10.0 | 0.0% | 26.8% | 594 | 87.2% | 0.376 | 0.087 |
| GLM-4-9B | gender | unseen | 0.88 | 9.1 | 5.4% | 26.8% | 466 | 85.7% | 0.435 | 0.115 |
| Llama-3.1-8B | neutral | raw | 0.81 | 9.9 | 0.4% | 23.3% | 980 | 48.6% | 0.357 | 0.072 |
| Llama-3.1-8B | neutral | unseen | 0.81 | 9.2 | 4.1% | 23.3% | 990 | 46.5% | 0.413 | 0.095 |
| Llama-3.1-8B | gender | raw | 0.80 | 9.9 | 0.5% | 23.4% | 993 | 47.8% | 0.355 | 0.070 |
| Llama-3.1-8B | gender | unseen | 0.80 | 9.2 | 3.5% | 23.4% | 1002 | 45.7% | 0.417 | 0.093 |

**Gender AUC, unseen lists (train items removed, comparable with EXP-003/004)** — LR balanced (best of the 9 classifier variants in brackets); 500 fixed test users, 3,367-user train pool

| Tier | Input | GLM-4-9B neutral | GLM-4-9B gender | Llama-3.1-8B neutral | Llama-3.1-8B gender |
| --- | --- | ---: | ---: | ---: | ---: |
| TH | Full train history (reference) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) |
| TH-prompt | Items shown in the LLMRec prompt (reference) | 0.716 (0.726) | 0.716 (0.726) | 0.716 (0.726) | 0.716 (0.726) |
| T2 | Multi-hot over list items | 0.636 (0.665) | 0.809 (0.813) | 0.667 (0.691) | 0.765 (0.795) |
| T2 | TF-IDF multi-hot | 0.645 (0.660) | 0.809 (0.816) | 0.693 (0.695) | 0.793 (0.797) |
| T3 | Genre histogram (18-d) | 0.602 (0.619) | 0.726 (0.731) | 0.667 (0.677) | 0.739 (0.740) |
| T3 | Title text embedding (MiniLM) | 0.621 (0.642) | 0.793 (0.807) | 0.615 (0.691) | 0.764 (0.789) |
| T4 | GLM-4-9B attacker on the titles (zero-shot / 8-shot staab) | 0.582 / 0.590 | 0.710 / 0.660 | 0.655 / 0.669 | 0.753 / 0.740 |
| T4 | Llama-3.1-8B attacker on the titles (zero-shot / 8-shot staab) | 0.573 / 0.598 | 0.697 / 0.717 | 0.661 / 0.651 | 0.752 / 0.748 |

**Gender AUC, raw lists (Stairway's output as is)** — LR balanced (best of the 9 classifier variants in brackets); 500 fixed test users, 3,367-user train pool

| Tier | Input | GLM-4-9B neutral | GLM-4-9B gender | Llama-3.1-8B neutral | Llama-3.1-8B gender |
| --- | --- | ---: | ---: | ---: | ---: |
| TH | Full train history (reference) | — | — | — | — |
| TH-prompt | Items shown in the LLMRec prompt (reference) | — | — | — | — |
| T2 | Multi-hot over list items | — | — | — | — |
| T2 | TF-IDF multi-hot | — | — | — | — |
| T3 | Genre histogram (18-d) | — | — | — | — |
| T3 | Title text embedding (MiniLM) | — | — | — | — |

**Neutral → gender, paired on the same 500 users** (LR balanced; ΔAUC with 95% paired-bootstrap CI) and how much the list content moves (Jaccard between a user's two lists, all 3,867 users)

| Model | List version | Multi-hot ΔAUC | TF-IDF ΔAUC | Genre ΔAUC | MiniLM ΔAUC | Jaccard male | Jaccard female | Romance share Δ (F / M) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-4-9B | unseen | 0.636→0.809 (+0.174 [+0.128, +0.223]) | 0.645→0.809 (+0.163 [+0.118, +0.214]) | 0.602→0.726 (+0.125 [+0.077, +0.176]) | 0.621→0.793 (+0.172 [+0.118, +0.230]) | 0.57 | 0.48 | +0.025 / -0.004 |
| Llama-3.1-8B | unseen | 0.667→0.765 (+0.098 [+0.049, +0.146]) | 0.693→0.793 (+0.101 [+0.057, +0.143]) | 0.667→0.739 (+0.073 [+0.033, +0.111]) | 0.615→0.764 (+0.149 [+0.092, +0.200]) | 0.49 | 0.46 | +0.018 / -0.007 |
