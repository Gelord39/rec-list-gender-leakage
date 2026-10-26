Romance intervention, EXP-004_lightgcn_d32_311, glm4_9b zero-shot, same 500 users

| Arm | Romance share | mean log-pop | AUC | Δ AUC vs original [95% CI] | mean P(male), women | mean P(male), men |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| original | 0.150 | 7.14 | 0.510 | — | 0.714 | 0.720 |
| romance_swap | 0.000 | 7.03 | 0.493 | -0.017 [-0.056, +0.021] | 0.779 | 0.776 |
| control_swap_s2024 | 0.150 | 7.03 | 0.481 | -0.029 [-0.060, +0.001] | 0.719 | 0.717 |
| control_swap_s2025 | 0.150 | 7.03 | 0.504 | -0.006 [-0.034, +0.021] | 0.709 | 0.723 |
| control_swap_s2026 | 0.150 | 7.03 | 0.525 | +0.015 [-0.013, +0.042] | 0.712 | 0.729 |

control arms replace min(#Romance, #non-Romance) films per user; 0 Romance slots per arm had no non-Romance partner
