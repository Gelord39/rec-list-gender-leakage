**List diagnostics and utility** (3,867 users; HR/P/NDCG@10 on the 695 users with test items)

| Model | Prompt | List version | Match rate | Mean length | <5 items | Already watched (raw) | Distinct items | Top-50 share | HR@10 | NDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-4-9B | neutral | raw | 0.91 | 9.0 | 0.2% | 28.9% | 888 | 88.3% | 0.360 | 0.078 |
| GLM-4-9B | neutral | unseen | 0.91 | 6.5 | 22.7% | 28.9% | 250 | 93.6% | 0.361 | 0.090 |
| GLM-4-9B | sensitive | raw | 0.93 | 9.2 | 0.3% | 38.6% | 1411 | 75.0% | 0.317 | 0.072 |
| GLM-4-9B | sensitive | unseen | 0.93 | 6.0 | 30.5% | 38.6% | 403 | 87.0% | 0.324 | 0.086 |
| Llama-3.1-8B | neutral | raw | 0.85 | 8.8 | 1.6% | 18.8% | 726 | 64.8% | 0.353 | 0.071 |
| Llama-3.1-8B | neutral | unseen | 0.85 | 7.2 | 17.5% | 18.8% | 654 | 65.8% | 0.354 | 0.077 |
| Llama-3.1-8B | sensitive | raw | 0.86 | 8.7 | 2.2% | 19.5% | 754 | 64.7% | 0.345 | 0.073 |
| Llama-3.1-8B | sensitive | unseen | 0.86 | 7.1 | 18.4% | 19.5% | 668 | 66.0% | 0.347 | 0.081 |

**Gender AUC, unseen lists (train items removed, comparable with EXP-003/004)** — LR balanced (best of the 9 classifier variants in brackets); 500 fixed test users, 3,367-user train pool

| Tier | Input | GLM-4-9B neutral | GLM-4-9B sensitive | Llama-3.1-8B neutral | Llama-3.1-8B sensitive |
| --- | --- | ---: | ---: | ---: | ---: |
| TH | Full train history (reference) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) |
| TH-prompt | Items shown in the LLMRec prompt (reference) | 0.823 (0.823) | 0.823 (0.823) | 0.823 (0.823) | 0.823 (0.823) |
| T2 | Multi-hot over list items | 0.675 (0.698) | 0.894 (0.895) | 0.699 (0.709) | 0.825 (0.833) |
| T2 | TF-IDF multi-hot | 0.680 (0.685) | 0.895 (0.896) | 0.717 (0.720) | 0.835 (0.839) |
| T3 | Genre histogram (18-d) | 0.569 (0.591) | 0.812 (0.833) | 0.697 (0.697) | 0.788 (0.790) |
| T3 | Title text embedding (MiniLM) | 0.659 (0.678) | 0.865 (0.878) | 0.687 (0.701) | 0.793 (0.826) |
| T4 | GLM-4-9B attacker on the titles (zero-shot / 8-shot staab) | 0.588 / 0.613 | 0.812 / 0.753 | 0.671 / 0.665 | 0.801 / 0.794 |

**Gender AUC, raw lists (Stairway's output as is)** — LR balanced (best of the 9 classifier variants in brackets); 500 fixed test users, 3,367-user train pool

| Tier | Input | GLM-4-9B neutral | GLM-4-9B sensitive | Llama-3.1-8B neutral | Llama-3.1-8B sensitive |
| --- | --- | ---: | ---: | ---: | ---: |
| TH | Full train history (reference) | — | — | — | — |
| TH-prompt | Items shown in the LLMRec prompt (reference) | — | — | — | — |
| T2 | Multi-hot over list items | — | — | — | — |
| T2 | TF-IDF multi-hot | — | — | — | — |
| T3 | Genre histogram (18-d) | — | — | — | — |
| T3 | Title text embedding (MiniLM) | — | — | — | — |

**Neutral → sensitive, paired on the same 500 users** (LR balanced; ΔAUC with 95% paired-bootstrap CI) and how much the list content moves (Jaccard between a user's two lists, all 3,867 users)

| Model | List version | Multi-hot ΔAUC | TF-IDF ΔAUC | Genre ΔAUC | MiniLM ΔAUC | Jaccard male | Jaccard female | Romance share Δ (F / M) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-4-9B | unseen | 0.675→0.894 (+0.219 [+0.168, +0.271]) | 0.680→0.895 (+0.216 [+0.167, +0.266]) | 0.569→0.812 (+0.243 [+0.178, +0.306]) | 0.659→0.865 (+0.206 [+0.154, +0.262]) | 0.41 | 0.30 | +0.046 / -0.008 |
| Llama-3.1-8B | unseen | 0.699→0.825 (+0.126 [+0.082, +0.174]) | 0.717→0.835 (+0.118 [+0.081, +0.160]) | 0.697→0.788 (+0.091 [+0.053, +0.134]) | 0.687→0.793 (+0.105 [+0.055, +0.158]) | 0.38 | 0.36 | +0.022 / -0.010 |
