### Table 1 — Master table: all attribute-inference attackers tried so far

EXP-004 (LightGCN d32, global_311 split, ML-1M). Fixed single train/test split: the same 500 test users for every attacker, 3367-user training pool. Target attribute: gender (binary). All classifiers use balanced class weights (see Table 2). The LLM has no training step to balance; its AUC comes from the answer token's log-probabilities (score = P(male) / (P(male) + P(female))). "Labels" = how many users with a known gender the attacker needs before it can attack anybody.

Tiers — **T0** model's user embedding · **T1** list + the model's item embedding table · **T2** list item IDs only · **T3** list + public item metadata · **T4** list titles as plain text. n = 500 test users for every classifier row; the LLM rows give their own n (refusals are dropped before scoring).


| Tier | Input representation          | Attacker                          | Labels | Acc   | BAcc  | F1-macro | Recall (F) | AUC [95% CI]         |
| ---- | ----------------------------- | --------------------------------- | ------ | ----- | ----- | -------- | ---------- | -------------------- |
| T0   | User embedding (upper bound)  | MLP                               | 3,367  | 0.726 | 0.748 | 0.710    | 0.803      | 0.828 [0.786, 0.866] |
| T0   | User embedding (upper bound)  | LR                                | 3,367  | 0.740 | 0.758 | 0.722    | 0.803      | 0.827 [0.788, 0.867] |
| T0   | User embedding (upper bound)  | SVM (linear)                      | 3,367  | 0.740 | 0.758 | 0.722    | 0.803      | 0.827 [0.786, 0.866] |
| T0   | User embedding (upper bound)  | SVM (RBF)                         | 3,367  | 0.730 | 0.752 | 0.714    | 0.809      | 0.817 [0.775, 0.857] |
| T1   | Recommender inversion u-hat   | LR                                | 3,367  | 0.624 | 0.632 | 0.603    | 0.651      | 0.704 [0.655, 0.750] |
| T1   | Recommender inversion u-hat   | SVM (linear)                      | 3,367  | 0.626 | 0.633 | 0.605    | 0.651      | 0.704 [0.656, 0.750] |
| T1   | List-derived, mean-pool       | SVM (linear)                      | 3,367  | 0.626 | 0.631 | 0.604    | 0.645      | 0.684 [0.636, 0.733] |
| T1   | List-derived, rank-pool       | SVM (linear)                      | 3,367  | 0.630 | 0.642 | 0.611    | 0.671      | 0.683 [0.632, 0.732] |
| T1   | List-derived, rank-pool       | LR                                | 3,367  | 0.628 | 0.629 | 0.604    | 0.632      | 0.678 [0.625, 0.727] |
| T1   | List-derived, mean-pool       | LR                                | 3,367  | 0.630 | 0.636 | 0.609    | 0.651      | 0.677 [0.626, 0.727] |
| T1   | Recommender inversion u-hat   | SVM (RBF)                         | 3,367  | 0.606 | 0.624 | 0.590    | 0.671      | 0.673 [0.624, 0.720] |
| T1   | List-derived, rank-pool       | MLP                               | 3,367  | 0.626 | 0.620 | 0.599    | 0.605      | 0.672 [0.620, 0.718] |
| T1   | List-derived, mean-pool       | MLP                               | 3,367  | 0.634 | 0.622 | 0.604    | 0.592      | 0.668 [0.614, 0.714] |
| T1   | List-derived, rank-pool       | SVM (RBF)                         | 3,367  | 0.616 | 0.615 | 0.592    | 0.612      | 0.660 [0.611, 0.706] |
| T1   | List-derived, mean-pool       | SVM (RBF)                         | 3,367  | 0.590 | 0.609 | 0.575    | 0.658      | 0.658 [0.607, 0.705] |
| T1   | Recommender inversion u-hat   | MLP                               | 3,367  | 0.618 | 0.590 | 0.580    | 0.520      | 0.634 [0.578, 0.682] |
| T2   | TF-IDF multi-hot              | LR                                | 3,367  | 0.648 | 0.656 | 0.627    | 0.678      | 0.702 [0.652, 0.747] |
| T2   | Multi-hot over list items     | LR                                | 3,367  | 0.642 | 0.661 | 0.625    | 0.711      | 0.700 [0.649, 0.745] |
| T2   | Multi-hot over list items     | SVM (RBF)                         | 3,367  | 0.628 | 0.633 | 0.606    | 0.645      | 0.698 [0.649, 0.744] |
| T2   | TF-IDF multi-hot              | SVM (linear)                      | 3,367  | 0.646 | 0.660 | 0.628    | 0.697      | 0.698 [0.648, 0.744] |
| T2   | Multi-hot over list items     | SVM (linear)                      | 3,367  | 0.636 | 0.657 | 0.620    | 0.711      | 0.693 [0.644, 0.739] |
| T2   | TF-IDF multi-hot              | SVM (RBF)                         | 3,367  | 0.628 | 0.635 | 0.607    | 0.651      | 0.691 [0.641, 0.736] |
| T2   | TF-IDF multi-hot              | MLP                               | 3,367  | 0.652 | 0.609 | 0.604    | 0.500      | 0.672 [0.623, 0.717] |
| T2   | Multi-hot over list items     | MLP                               | 3,367  | 0.660 | 0.606 | 0.604    | 0.467      | 0.671 [0.621, 0.717] |
| T3   | Title text embedding (MiniLM) | LR                                | 3,367  | 0.640 | 0.664 | 0.625    | 0.724      | 0.694 [0.644, 0.739] |
| T3   | Title text embedding (MiniLM) | SVM (RBF)                         | 3,367  | 0.632 | 0.639 | 0.611    | 0.658      | 0.692 [0.642, 0.739] |
| T3   | Title text embedding (MiniLM) | SVM (linear)                      | 3,367  | 0.632 | 0.654 | 0.617    | 0.711      | 0.691 [0.640, 0.737] |
| T3   | Genre histogram (18-d)        | SVM (linear)                      | 3,367  | 0.584 | 0.586 | 0.562    | 0.592      | 0.656 [0.603, 0.707] |
| T3   | Genre histogram (18-d)        | LR                                | 3,367  | 0.580 | 0.582 | 0.558    | 0.586      | 0.656 [0.602, 0.707] |
| T3   | Title text embedding (MiniLM) | MLP                               | 3,367  | 0.640 | 0.582 | 0.581    | 0.434      | 0.646 [0.595, 0.695] |
| T3   | Genre histogram (18-d)        | SVM (RBF)                         | 3,367  | 0.596 | 0.613 | 0.580    | 0.658      | 0.645 [0.591, 0.696] |
| T3   | Genre histogram (18-d)        | MLP                               | 3,367  | 0.598 | 0.591 | 0.571    | 0.572      | 0.637 [0.587, 0.685] |
| T4   | Raw list titles               | llama3.2 / staab_few_shot (n=412) | 2      | 0.604 | 0.535 | 0.535    | 0.336      | 0.533                |
| T4   | Raw list titles               | llama3.2 / staab (n=493)          | 0      | 0.493 | 0.507 | 0.479    | 0.544      | 0.514                |
| T4   | Raw list titles               | llama3.2 / zero_shot (n=500)      | 0      | 0.696 | 0.504 | 0.423    | 0.013      | 0.491                |
| T4   | Raw list titles               | llama3.2 / few_shot (n=500)       | 2      | 0.376 | 0.479 | 0.372    | 0.743      | 0.459                |
| —    | Majority class                | —                                 | 0      | 0.696 | 0.500 | 0.410    | 0.000      | —                    |


LLM rows drop hard safety refusals before scoring: staab 7/500; staab_few_shot 88/500.

### Table 2 — Effect of class balancing (same estimator, nothing else changed)

`default` = library defaults, no class weighting. `balanced` = balanced class weights and nothing else (LR/SVM via `class_weight`, MLP via `sample_weight`, which `MLPClassifier` takes instead — it has no `class_weight` parameter). `lr_balanced_cv` additionally tunes C by 5-fold CV on the train pool and is reported separately because it changes two things at once (this is the variant the 2026-09-14 numbers used).


| Input representation          | Attacker               | BAcc def → bal | ΔBAcc  | AUC def → bal | ΔAUC   | Predicted male def → bal |
| ----------------------------- | ---------------------- | -------------- | ------ | ------------- | ------ | ------------------------ |
| User embedding (upper bound)  | LR                     | 0.673 → 0.758  | +0.085 | 0.827 → 0.827 | -0.000 | 81% → 56%                |
| User embedding (upper bound)  | MLP                    | 0.706 → 0.748  | +0.041 | 0.822 → 0.828 | +0.006 | 75% → 54%                |
| User embedding (upper bound)  | SVM (RBF)              | 0.628 → 0.752  | +0.124 | 0.809 → 0.817 | +0.008 | 86% → 54%                |
| User embedding (upper bound)  | SVM (linear)           | 0.666 → 0.758  | +0.092 | 0.828 → 0.827 | -0.001 | 81% → 56%                |
| User embedding (upper bound)  | LR, balanced + C tuned | — → 0.761      | —      | — → 0.828     | —      | — → 56%                  |
| List-derived, mean-pool       | LR                     | 0.519 → 0.636  | +0.117 | 0.675 → 0.677 | +0.002 | 96% → 54%                |
| List-derived, mean-pool       | MLP                    | 0.523 → 0.622  | +0.099 | 0.677 → 0.668 | -0.009 | 97% → 58%                |
| List-derived, mean-pool       | SVM (RBF)              | 0.500 → 0.609  | +0.109 | 0.635 → 0.658 | +0.023 | 100% → 49%               |
| List-derived, mean-pool       | SVM (linear)           | 0.520 → 0.631  | +0.111 | 0.684 → 0.684 | +0.001 | 96% → 54%                |
| List-derived, mean-pool       | LR, balanced + C tuned | — → 0.630      | —      | — → 0.685     | —      | — → 54%                  |
| List-derived, rank-pool       | LR                     | 0.525 → 0.629  | +0.104 | 0.676 → 0.678 | +0.002 | 96% → 55%                |
| List-derived, rank-pool       | MLP                    | 0.524 → 0.620  | +0.096 | 0.681 → 0.672 | -0.009 | 95% → 56%                |
| List-derived, rank-pool       | SVM (RBF)              | 0.500 → 0.615  | +0.115 | 0.669 → 0.660 | -0.008 | 100% → 55%               |
| List-derived, rank-pool       | SVM (linear)           | 0.521 → 0.642  | +0.120 | 0.682 → 0.683 | +0.001 | 96% → 53%                |
| List-derived, rank-pool       | LR, balanced + C tuned | — → 0.640      | —      | — → 0.683     | —      | — → 53%                  |
| Recommender inversion u-hat   | LR                     | 0.538 → 0.632  | +0.094 | 0.704 → 0.704 | +0.000 | 93% → 53%                |
| Recommender inversion u-hat   | MLP                    | 0.543 → 0.590  | +0.047 | 0.622 → 0.634 | +0.012 | 83% → 61%                |
| Recommender inversion u-hat   | SVM (RBF)              | 0.501 → 0.624  | +0.123 | 0.614 → 0.673 | +0.060 | 99% → 50%                |
| Recommender inversion u-hat   | SVM (linear)           | 0.527 → 0.633  | +0.106 | 0.703 → 0.704 | +0.001 | 95% → 53%                |
| Recommender inversion u-hat   | LR, balanced + C tuned | — → 0.635      | —      | — → 0.702     | —      | — → 51%                  |
| Multi-hot over list items     | LR                     | 0.545 → 0.661  | +0.116 | 0.700 → 0.700 | +0.000 | 92% → 51%                |
| Multi-hot over list items     | MLP                    | 0.632 → 0.606  | -0.027 | 0.671 → 0.671 | +0.000 | 70% → 68%                |
| Multi-hot over list items     | SVM (RBF)              | 0.509 → 0.633  | +0.124 | 0.627 → 0.698 | +0.071 | 99% → 54%                |
| Multi-hot over list items     | SVM (linear)           | 0.527 → 0.657  | +0.130 | 0.689 → 0.693 | +0.004 | 93% → 51%                |
| Multi-hot over list items     | LR, balanced + C tuned | — → 0.658      | —      | — → 0.703     | —      | — → 52%                  |
| TF-IDF multi-hot              | LR                     | 0.522 → 0.656  | +0.135 | 0.702 → 0.702 | +0.001 | 94% → 54%                |
| TF-IDF multi-hot              | MLP                    | 0.594 → 0.609  | +0.015 | 0.663 → 0.672 | +0.009 | 72% → 65%                |
| TF-IDF multi-hot              | SVM (RBF)              | 0.523 → 0.635  | +0.111 | 0.630 → 0.691 | +0.061 | 96% → 54%                |
| TF-IDF multi-hot              | SVM (linear)           | 0.535 → 0.660  | +0.125 | 0.693 → 0.698 | +0.005 | 92% → 53%                |
| TF-IDF multi-hot              | LR, balanced + C tuned | — → 0.656      | —      | — → 0.703     | —      | — → 54%                  |
| Genre histogram (18-d)        | LR                     | 0.503 → 0.582  | +0.079 | 0.656 → 0.656 | -0.000 | 98% → 53%                |
| Genre histogram (18-d)        | MLP                    | 0.546 → 0.591  | +0.045 | 0.631 → 0.637 | +0.007 | 82% → 55%                |
| Genre histogram (18-d)        | SVM (RBF)              | 0.502 → 0.613  | +0.112 | 0.561 → 0.645 | +0.084 | 100% → 50%               |
| Genre histogram (18-d)        | SVM (linear)           | 0.500 → 0.586  | +0.086 | 0.657 → 0.656 | -0.001 | 99% → 53%                |
| Genre histogram (18-d)        | LR, balanced + C tuned | — → 0.589      | —      | — → 0.660     | —      | — → 55%                  |
| Title text embedding (MiniLM) | LR                     | 0.542 → 0.664  | +0.122 | 0.692 → 0.694 | +0.001 | 90% → 50%                |
| Title text embedding (MiniLM) | MLP                    | 0.613 → 0.582  | -0.031 | 0.654 → 0.646 | -0.009 | 70% → 68%                |
| Title text embedding (MiniLM) | SVM (RBF)              | 0.516 → 0.639  | +0.123 | 0.625 → 0.692 | +0.067 | 97% → 54%                |
| Title text embedding (MiniLM) | SVM (linear)           | 0.537 → 0.654  | +0.117 | 0.683 → 0.691 | +0.008 | 93% → 50%                |
| Title text embedding (MiniLM) | LR, balanced + C tuned | — → 0.657      | —      | — → 0.701     | —      | — → 51%                  |




### Table 3 — Does the choice of classifier matter? (balanced variants, AUC)


| Input representation          | LR    | MLP   | SVM (RBF) | SVM (linear) | spread |
| ----------------------------- | ----- | ----- | --------- | ------------ | ------ |
| User embedding (upper bound)  | 0.827 | 0.828 | 0.817     | 0.827        | 0.010  |
| List-derived, mean-pool       | 0.677 | 0.668 | 0.658     | 0.684        | 0.027  |
| List-derived, rank-pool       | 0.678 | 0.672 | 0.660     | 0.683        | 0.023  |
| Recommender inversion u-hat   | 0.704 | 0.634 | 0.673     | 0.704        | 0.070  |
| Multi-hot over list items     | 0.700 | 0.671 | 0.698     | 0.693        | 0.029  |
| TF-IDF multi-hot              | 0.702 | 0.672 | 0.691     | 0.698        | 0.030  |
| Genre histogram (18-d)        | 0.656 | 0.637 | 0.645     | 0.656        | 0.019  |
| Title text embedding (MiniLM) | 0.694 | 0.646 | 0.692     | 0.691        | 0.048  |


Largest AUC spread across classifiers for a fixed input: **0.070**. AUC spread across inputs (balanced attackers, all classifiers): **0.193**. The input representation, not the classifier, is what decides how much gender leaks.