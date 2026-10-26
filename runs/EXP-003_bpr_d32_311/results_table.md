# Results — EXP-003_bpr_d32_311

## Recommendation utility (test set)

| Metric | Value |
| --- | --- |
| hit@5 | 0.5540 |
| hit@10 | 0.6734 |
| ndcg@5 | 0.2639 |
| ndcg@10 | 0.2448 |
| recall@5 | 0.0304 |
| recall@10 | 0.0501 |

## Sensitive-attribute leakage — all attackers

| Attacker | Input | BAcc | AUC / Accuracy | n | Majority/chance BAcc |
| --- | --- | --- | --- | --- | --- |
| Embedding — mlp | user embedding | 0.6807 | 0.7635 | 3869 | 0.5000 |
| Embedding — logistic_regression | user embedding | 0.6696 | 0.8127 | 3869 | 0.5000 |
| Embedding — svm | user embedding | 0.6553 | 0.7983 | 3869 | 0.5000 |
| LLM list — zero_shot | rec list only | 0.5733 | 0.7800 | 100 | 0.5000 |
| LLM list — few_shot | rec list only | 0.6267 | 0.4800 | 100 | 0.5000 |
| LLM list — staab | rec list only | 0.6067 | 0.5500 | 100 | 0.5000 |
| LLM list — staab_few_shot | rec list only | 0.5798 | 0.6296 | 81 | 0.5000 |

chance BAcc = 0.5 for balanced binary gender classification; majority BAcc = 0.5 by construction (balanced accuracy of an always-predict-majority classifier). "AUC / Accuracy" column is AUC for embedding attackers, plain accuracy for LLM list attackers (no probability score available from those).
