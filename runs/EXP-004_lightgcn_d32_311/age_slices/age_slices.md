Gender attack, BAcc / AUC by true age bucket — runs/EXP-004_lightgcn_d32_311 (lr_balanced for classifiers)

| Attacker | all | 18-24 (n=85) | 25-49 (n=343) | >=50 (n=72) | AUC range |
| --- | --- | --- | --- | --- | ---: |
| T0 user embedding | 0.758 / 0.827 | 0.765 / 0.831 | 0.773 / 0.840 | 0.669 / 0.767 | 0.073 |
| T1 list-derived, mean-pool | 0.636 / 0.677 | 0.662 / 0.660 | 0.635 / 0.686 | 0.616 / 0.712 | 0.052 |
| T1 recommender inversion | 0.632 / 0.704 | 0.583 / 0.646 | 0.645 / 0.722 | 0.636 / 0.725 | 0.079 |
| T2 multi-hot | 0.661 / 0.700 | 0.652 / 0.682 | 0.669 / 0.707 | 0.626 / 0.685 | 0.025 |
| T3 genre histogram | 0.582 / 0.656 | 0.474 / 0.546 | 0.609 / 0.683 | 0.616 / 0.727 | 0.181 |
| T3 MiniLM titles | 0.664 / 0.694 | 0.642 / 0.673 | 0.673 / 0.701 | 0.636 / 0.686 | 0.027 |
| T4 LLM zero_shot (lightgcn_glm4_9b_500) | 0.497 / 0.510 | 0.478 / 0.468 | 0.494 / 0.525 | 0.541 / 0.550 | 0.082 |
| T4 LLM staab_few_shot (lightgcn_glm4_9b_500) (labels n=488) | 0.517 / 0.538 | 0.481 / 0.504 | 0.520 / 0.541 | 0.569 / 0.609 | 0.105 |
