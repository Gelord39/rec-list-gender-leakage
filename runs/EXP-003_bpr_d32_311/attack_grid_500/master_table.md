### Table 1 — Master table: all attribute-inference attackers tried so far

EXP-003 (BPR d32, global_311 split, ML-1M). Fixed single train/test split: the same 500 test users for every attacker, 3367-user training pool. Target attribute: gender (binary). All classifiers use balanced class weights (see Table 2). T4 (LLM on the raw list titles) has not been run for this recommender yet. "Labels" = how many users with a known gender the attacker needs before it can attack anybody.

Tiers — **T0** model's user embedding · **T1** list + the model's item embedding table · **T2** list item IDs only · **T3** list + public item metadata · **T4** list titles as plain text. n = 500 test users for every classifier row; the LLM rows give their own n (refusals are dropped before scoring).

| Tier | Input representation | Attacker | Labels | Acc | BAcc | F1-macro | Recall (F) | AUC [95% CI] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| T0 | User embedding (upper bound) | SVM (RBF) | 3,367 | 0.748 | 0.758 | 0.728 | 0.783 | 0.833 [0.793, 0.870] |
| T0 | User embedding (upper bound) | LR | 3,367 | 0.756 | 0.760 | 0.734 | 0.770 | 0.822 [0.781, 0.859] |
| T0 | User embedding (upper bound) | SVM (linear) | 3,367 | 0.754 | 0.758 | 0.732 | 0.770 | 0.822 [0.781, 0.859] |
| T0 | User embedding (upper bound) | MLP | 3,367 | 0.720 | 0.690 | 0.681 | 0.612 | 0.792 [0.752, 0.833] |
| T1 | Recommender inversion u-hat | LR | 3,367 | 0.672 | 0.690 | 0.655 | 0.737 | 0.760 [0.714, 0.805] |
| T1 | Recommender inversion u-hat | SVM (linear) | 3,367 | 0.672 | 0.690 | 0.655 | 0.737 | 0.758 [0.712, 0.802] |
| T1 | List-derived, mean-pool | LR | 3,367 | 0.682 | 0.701 | 0.665 | 0.750 | 0.752 [0.707, 0.797] |
| T1 | List-derived, mean-pool | SVM (linear) | 3,367 | 0.676 | 0.697 | 0.660 | 0.750 | 0.751 [0.705, 0.797] |
| T1 | List-derived, rank-pool | LR | 3,367 | 0.674 | 0.688 | 0.655 | 0.724 | 0.749 [0.704, 0.795] |
| T1 | List-derived, rank-pool | SVM (linear) | 3,367 | 0.672 | 0.688 | 0.654 | 0.730 | 0.748 [0.702, 0.794] |
| T1 | List-derived, mean-pool | MLP | 3,367 | 0.684 | 0.684 | 0.660 | 0.684 | 0.747 [0.700, 0.790] |
| T1 | List-derived, rank-pool | MLP | 3,367 | 0.672 | 0.672 | 0.647 | 0.671 | 0.744 [0.698, 0.790] |
| T1 | List-derived, mean-pool | SVM (RBF) | 3,367 | 0.654 | 0.657 | 0.631 | 0.664 | 0.737 [0.690, 0.783] |
| T1 | List-derived, rank-pool | SVM (RBF) | 3,367 | 0.650 | 0.650 | 0.626 | 0.651 | 0.736 [0.691, 0.783] |
| T1 | Recommender inversion u-hat | SVM (RBF) | 3,367 | 0.688 | 0.683 | 0.661 | 0.671 | 0.727 [0.682, 0.776] |
| T1 | Recommender inversion u-hat | MLP | 3,367 | 0.670 | 0.613 | 0.612 | 0.467 | 0.673 [0.622, 0.721] |
| T2 | Multi-hot over list items | SVM (RBF) | 3,367 | 0.688 | 0.678 | 0.659 | 0.651 | 0.739 [0.694, 0.786] |
| T2 | TF-IDF multi-hot | LR | 3,367 | 0.676 | 0.684 | 0.655 | 0.704 | 0.736 [0.688, 0.783] |
| T2 | TF-IDF multi-hot | SVM (RBF) | 3,367 | 0.706 | 0.691 | 0.675 | 0.651 | 0.734 [0.692, 0.782] |
| T2 | Multi-hot over list items | LR | 3,367 | 0.664 | 0.677 | 0.645 | 0.711 | 0.718 [0.671, 0.767] |
| T2 | TF-IDF multi-hot | SVM (linear) | 3,367 | 0.656 | 0.668 | 0.636 | 0.697 | 0.716 [0.668, 0.766] |
| T2 | Multi-hot over list items | SVM (linear) | 3,367 | 0.648 | 0.656 | 0.627 | 0.678 | 0.697 [0.647, 0.748] |
| T2 | Multi-hot over list items | MLP | 3,367 | 0.674 | 0.588 | 0.591 | 0.368 | 0.671 [0.623, 0.720] |
| T2 | TF-IDF multi-hot | MLP | 3,367 | 0.684 | 0.627 | 0.627 | 0.480 | 0.669 [0.619, 0.720] |
| T3 | Title text embedding (MiniLM) | SVM (RBF) | 3,367 | 0.674 | 0.664 | 0.645 | 0.638 | 0.731 [0.684, 0.779] |
| T3 | Genre histogram (18-d) | SVM (linear) | 3,367 | 0.644 | 0.655 | 0.624 | 0.684 | 0.713 [0.663, 0.761] |
| T3 | Genre histogram (18-d) | LR | 3,367 | 0.642 | 0.652 | 0.622 | 0.678 | 0.713 [0.662, 0.760] |
| T3 | Genre histogram (18-d) | SVM (RBF) | 3,367 | 0.642 | 0.641 | 0.617 | 0.638 | 0.705 [0.654, 0.755] |
| T3 | Title text embedding (MiniLM) | LR | 3,367 | 0.662 | 0.674 | 0.642 | 0.704 | 0.701 [0.652, 0.750] |
| T3 | Title text embedding (MiniLM) | MLP | 3,367 | 0.674 | 0.603 | 0.605 | 0.421 | 0.690 [0.641, 0.737] |
| T3 | Title text embedding (MiniLM) | SVM (linear) | 3,367 | 0.652 | 0.667 | 0.634 | 0.704 | 0.679 [0.630, 0.730] |
| T3 | Genre histogram (18-d) | MLP | 3,367 | 0.624 | 0.635 | 0.605 | 0.664 | 0.666 [0.612, 0.716] |
| — | Majority class | — | 0 | 0.696 | 0.500 | 0.410 | 0.000 | — |

### Table 2 — Effect of class balancing (same estimator, nothing else changed)

`default` = library defaults, no class weighting. `balanced` = balanced class weights and nothing else (LR/SVM via `class_weight`, MLP via `sample_weight`, which `MLPClassifier` takes instead — it has no `class_weight` parameter). `lr_balanced_cv` additionally tunes C by 5-fold CV on the train pool and is reported separately because it changes two things at once (this is the variant the 2026-09-14 numbers used).

| Input representation | Attacker | BAcc def → bal | ΔBAcc | AUC def → bal | ΔAUC | Predicted male def → bal |
| --- | --- | --- | ---: | --- | ---: | --- |
| User embedding (upper bound) | LR | 0.661 → 0.760 | +0.099 | 0.822 → 0.822 | -0.000 | 82% → 59% |
| User embedding (upper bound) | MLP | 0.698 → 0.690 | -0.008 | 0.791 → 0.792 | +0.001 | 72% → 65% |
| User embedding (upper bound) | SVM (RBF) | 0.668 → 0.758 | +0.089 | 0.811 → 0.833 | +0.022 | 82% → 58% |
| User embedding (upper bound) | SVM (linear) | 0.654 → 0.758 | +0.105 | 0.821 → 0.822 | +0.001 | 83% → 59% |
| User embedding (upper bound) | LR, balanced + C tuned | — → 0.760 | — | — → 0.822 | — | — → 59% |
| List-derived, mean-pool | LR | 0.594 → 0.701 | +0.107 | 0.749 → 0.752 | +0.003 | 89% → 53% |
| List-derived, mean-pool | MLP | 0.632 → 0.684 | +0.052 | 0.745 → 0.747 | +0.002 | 82% → 57% |
| List-derived, mean-pool | SVM (RBF) | 0.556 → 0.657 | +0.101 | 0.693 → 0.737 | +0.044 | 94% → 55% |
| List-derived, mean-pool | SVM (linear) | 0.586 → 0.697 | +0.111 | 0.748 → 0.751 | +0.003 | 90% → 52% |
| List-derived, mean-pool | LR, balanced + C tuned | — → 0.660 | — | — → 0.740 | — | — → 54% |
| List-derived, rank-pool | LR | 0.598 → 0.688 | +0.090 | 0.747 → 0.749 | +0.003 | 89% → 54% |
| List-derived, rank-pool | MLP | 0.627 → 0.672 | +0.045 | 0.747 → 0.744 | -0.004 | 83% → 57% |
| List-derived, rank-pool | SVM (RBF) | 0.559 → 0.650 | +0.091 | 0.700 → 0.736 | +0.036 | 94% → 56% |
| List-derived, rank-pool | SVM (linear) | 0.586 → 0.688 | +0.102 | 0.745 → 0.748 | +0.002 | 90% → 53% |
| List-derived, rank-pool | LR, balanced + C tuned | — → 0.653 | — | — → 0.739 | — | — → 55% |
| Recommender inversion u-hat | LR | 0.614 → 0.690 | +0.076 | 0.760 → 0.760 | -0.000 | 86% → 53% |
| Recommender inversion u-hat | MLP | 0.582 → 0.613 | +0.031 | 0.657 → 0.673 | +0.016 | 73% → 69% |
| Recommender inversion u-hat | SVM (RBF) | 0.587 → 0.683 | +0.096 | 0.712 → 0.727 | +0.014 | 88% → 58% |
| Recommender inversion u-hat | SVM (linear) | 0.600 → 0.690 | +0.091 | 0.758 → 0.758 | -0.000 | 88% → 53% |
| Recommender inversion u-hat | LR, balanced + C tuned | — → 0.683 | — | — → 0.757 | — | — → 52% |
| Multi-hot over list items | LR | 0.608 → 0.677 | +0.069 | 0.720 → 0.718 | -0.001 | 85% → 54% |
| Multi-hot over list items | MLP | 0.610 → 0.588 | -0.022 | 0.652 → 0.671 | +0.019 | 71% → 75% |
| Multi-hot over list items | SVM (RBF) | 0.584 → 0.678 | +0.094 | 0.700 → 0.739 | +0.040 | 90% → 60% |
| Multi-hot over list items | SVM (linear) | 0.588 → 0.656 | +0.068 | 0.697 → 0.697 | +0.000 | 86% → 54% |
| Multi-hot over list items | LR, balanced + C tuned | — → 0.684 | — | — → 0.736 | — | — → 53% |
| TF-IDF multi-hot | LR | 0.607 → 0.684 | +0.077 | 0.737 → 0.736 | -0.001 | 87% → 55% |
| TF-IDF multi-hot | MLP | 0.613 → 0.627 | +0.014 | 0.673 → 0.669 | -0.004 | 72% → 70% |
| TF-IDF multi-hot | SVM (RBF) | 0.583 → 0.691 | +0.108 | 0.701 → 0.734 | +0.033 | 88% → 61% |
| TF-IDF multi-hot | SVM (linear) | 0.596 → 0.668 | +0.072 | 0.717 → 0.716 | -0.001 | 87% → 54% |
| TF-IDF multi-hot | LR, balanced + C tuned | — → 0.687 | — | — → 0.736 | — | — → 55% |
| Genre histogram (18-d) | LR | 0.589 → 0.652 | +0.063 | 0.711 → 0.713 | +0.002 | 89% → 53% |
| Genre histogram (18-d) | MLP | 0.614 → 0.635 | +0.021 | 0.671 → 0.666 | -0.004 | 76% → 52% |
| Genre histogram (18-d) | SVM (RBF) | 0.586 → 0.641 | +0.055 | 0.652 → 0.705 | +0.054 | 90% → 56% |
| Genre histogram (18-d) | SVM (linear) | 0.589 → 0.655 | +0.066 | 0.710 → 0.713 | +0.003 | 90% → 53% |
| Genre histogram (18-d) | LR, balanced + C tuned | — → 0.649 | — | — → 0.712 | — | — → 54% |
| Title text embedding (MiniLM) | LR | 0.607 → 0.674 | +0.067 | 0.702 → 0.701 | -0.001 | 83% → 54% |
| Title text embedding (MiniLM) | MLP | 0.610 → 0.603 | -0.007 | 0.685 → 0.690 | +0.005 | 76% → 72% |
| Title text embedding (MiniLM) | SVM (RBF) | 0.584 → 0.664 | +0.080 | 0.697 → 0.731 | +0.035 | 90% → 59% |
| Title text embedding (MiniLM) | SVM (linear) | 0.587 → 0.667 | +0.080 | 0.684 → 0.679 | -0.005 | 84% → 53% |
| Title text embedding (MiniLM) | LR, balanced + C tuned | — → 0.666 | — | — → 0.717 | — | — → 53% |

### Table 3 — Does the choice of classifier matter? (balanced variants, AUC)

| Input representation | LR | MLP | SVM (RBF) | SVM (linear) | spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| User embedding (upper bound) | 0.822 | 0.792 | 0.833 | 0.822 | 0.041 |
| List-derived, mean-pool | 0.752 | 0.747 | 0.737 | 0.751 | 0.015 |
| List-derived, rank-pool | 0.749 | 0.744 | 0.736 | 0.748 | 0.013 |
| Recommender inversion u-hat | 0.760 | 0.673 | 0.727 | 0.758 | 0.087 |
| Multi-hot over list items | 0.718 | 0.671 | 0.739 | 0.697 | 0.068 |
| TF-IDF multi-hot | 0.736 | 0.669 | 0.734 | 0.716 | 0.067 |
| Genre histogram (18-d) | 0.713 | 0.666 | 0.705 | 0.713 | 0.047 |
| Title text embedding (MiniLM) | 0.701 | 0.690 | 0.731 | 0.679 | 0.052 |

Largest AUC spread across classifiers for a fixed input: **0.087**. AUC spread across inputs (balanced attackers, all classifiers): **0.166**. The input representation, not the classifier, is what decides how much gender leaks.

