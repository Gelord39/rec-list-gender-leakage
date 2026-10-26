**List diagnostics and utility** (3,867 users; HR/P/NDCG@10 on the 695 users with test items)

| Model | Prompt | List version | Match rate | Mean length | <5 items | Already watched (raw) | Distinct items | Top-50 share | HR@10 | NDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-4-9B | neutral | raw | 0.91 | 10.0 | 0.0% | 26.3% | 905 | 88.6% | 0.358 | 0.080 |
| GLM-4-9B | neutral | unseen | 0.91 | 8.9 | 8.6% | 26.3% | 373 | 90.5% | 0.414 | 0.106 |
| GLM-4-9B | gender | raw | 0.91 | 10.0 | 0.1% | 27.0% | 907 | 87.6% | 0.373 | 0.083 |
| GLM-4-9B | gender | unseen | 0.91 | 8.9 | 9.2% | 27.0% | 379 | 89.9% | 0.422 | 0.111 |
| Llama-3.1-8B | neutral | raw | 0.85 | 9.9 | 0.6% | 17.3% | 827 | 63.3% | 0.344 | 0.076 |
| Llama-3.1-8B | neutral | unseen | 0.85 | 9.3 | 3.8% | 17.3% | 805 | 62.0% | 0.393 | 0.093 |
| Llama-3.1-8B | gender | raw | 0.85 | 9.9 | 0.6% | 17.5% | 855 | 62.5% | 0.344 | 0.074 |
| Llama-3.1-8B | gender | unseen | 0.85 | 9.3 | 4.1% | 17.5% | 837 | 61.3% | 0.383 | 0.091 |

**Gender AUC, unseen lists (train items removed, comparable with EXP-003/004)** — LR balanced (best of the 9 classifier variants in brackets); 500 fixed test users, 3,367-user train pool

| Tier | Input | GLM-4-9B neutral | GLM-4-9B gender | Llama-3.1-8B neutral | Llama-3.1-8B gender |
| --- | --- | ---: | ---: | ---: | ---: |
| TH | Full train history (reference) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) | 0.870 (0.870) |
| TH-prompt | Items shown in the LLMRec prompt (reference) | 0.823 (0.823) | 0.823 (0.823) | 0.823 (0.823) | 0.823 (0.823) |
| T2 | Multi-hot over list items | 0.618 (0.623) | 0.771 (0.779) | 0.678 (0.712) | 0.762 (0.786) |
| T2 | TF-IDF multi-hot | 0.634 (0.634) | 0.781 (0.781) | 0.690 (0.707) | 0.776 (0.783) |
| T3 | Genre histogram (18-d) | 0.581 (0.600) | 0.721 (0.722) | 0.698 (0.700) | 0.746 (0.757) |
| T3 | Title text embedding (MiniLM) | 0.608 (0.623) | 0.757 (0.767) | 0.655 (0.698) | 0.753 (0.796) |
| T4 | GLM-4-9B attacker on the titles (zero-shot / 8-shot staab) | 0.556 / 0.598 | 0.633 / 0.625 | 0.665 / 0.648 | 0.726 / 0.694 |
| T4 | Llama-3.1-8B attacker on the titles (zero-shot / 8-shot staab) | 0.579 / 0.556 | 0.682 / 0.647 | 0.665 / 0.637 | 0.722 / 0.728 |

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
| GLM-4-9B | unseen | 0.618→0.771 (+0.153 [+0.100, +0.208]) | 0.634→0.781 (+0.147 [+0.095, +0.201]) | 0.581→0.721 (+0.140 [+0.087, +0.199]) | 0.608→0.757 (+0.149 [+0.091, +0.209]) | 0.60 | 0.52 | +0.021 / -0.002 |
| Llama-3.1-8B | unseen | 0.678→0.762 (+0.084 [+0.045, +0.129]) | 0.690→0.776 (+0.086 [+0.051, +0.122]) | 0.698→0.746 (+0.048 [+0.012, +0.083]) | 0.655→0.753 (+0.098 [+0.050, +0.148]) | 0.54 | 0.51 | +0.013 / -0.006 |
