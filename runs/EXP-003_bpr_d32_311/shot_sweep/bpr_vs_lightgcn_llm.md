LLM attacker AUC on BPR vs LightGCN top-10 lists, same 500 users (paired bootstrap)

| Style | Attacker tag | n | AUC BPR | AUC LightGCN | Δ BPR − LightGCN [95% CI] |
| --- | --- | ---: | ---: | ---: | --- |
| zero_shot | llama32_500_lp | 500 | 0.630 | 0.491 | +0.139 [+0.082, +0.197] |
| staab_few_shot | llama32_500_lp | 349 | 0.649 | 0.518 | +0.130 [+0.054, +0.212] |
| zero_shot | glm4_9b_500 | 500 | 0.638 | 0.510 | +0.128 [+0.074, +0.181] |
| staab_few_shot | glm4_9b_500 | 500 | 0.630 | 0.538 | +0.092 [+0.035, +0.151] |
| zero_shot | llama31_8b_500 | 500 | 0.650 | 0.489 | +0.160 [+0.110, +0.212] |
| zero_shot | qwen25_14b_500 | 500 | 0.659 | 0.532 | +0.127 [+0.079, +0.176] |
| many_shot | glm4_9b_500_shots8_s2025 | 500 | 0.670 | 0.580 | +0.090 [+0.052, +0.129] |
| many_shot | qwen25_14b_500_shots16_s2025 | 500 | 0.693 | 0.585 | +0.108 [+0.068, +0.148] |
| staab_many_shot | qwen25_14b_500_shots16_s2025 | 500 | 0.679 | 0.595 | +0.084 [+0.042, +0.124] |
