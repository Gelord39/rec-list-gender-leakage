Romance intervention, EXP-003_bpr_d32_311, glm4_9b zero-shot, same 500 users

| Arm | Romance share | mean log-pop | AUC | Δ AUC vs original [95% CI] | mean P(male), women | mean P(male), men |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| original | 0.163 | 7.04 | 0.638 | — | 0.615 | 0.718 |
| romance_swap | 0.000 | 6.94 | 0.592 | -0.046 [-0.086, -0.006] | 0.727 | 0.768 |
| control_swap_s2024 | 0.163 | 6.93 | 0.628 | -0.010 [-0.036, +0.017] | 0.617 | 0.712 |
| control_swap_s2025 | 0.163 | 6.93 | 0.632 | -0.006 [-0.031, +0.017] | 0.608 | 0.708 |
| control_swap_s2026 | 0.163 | 6.93 | 0.647 | +0.009 [-0.019, +0.039] | 0.606 | 0.719 |

control arms replace min(#Romance, #non-Romance) films per user; 28 Romance slots per arm had no non-Romance partner
