# Sensitive-attribute leakage in recommendation lists

Code and results for the 15 ECTS project report *Evaluating Sensitive Attribute Leakage and
Fairness in LLM-based Recommender Systems* (Yuwang Ji, DIKU, University of Copenhagen, 2026;
supervisors Maria Maistro and Theresia Veronika Rampisela).

The report measures how much a user's **gender** leaks from **top-10 recommendation lists** on
MovieLens 1M, for two collaborative filtering recommenders (BPR, LightGCN) and an LLM used as the
recommender (the LLMRec of *Stairway to Fairness*), to classifiers trained on labelled users and
to LLMs that use no or only a few labelled examples.

## Research questions

| | Question |
| --- | --- |
| RQ1 | How much does a collaborative-filtering top-10 list reveal about a user's gender, compared with the user embedding and the interaction history, and what in the list carries it? |
| RQ2 | Can an LLM infer gender from a list without labelled training data, and when does it succeed? |
| RQ3 | When an LLM recommender is told the user's gender, does it treat women and men differently, and does this make its lists leak gender? |

## Setup

Python 3.12 and [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

LLM attackers and the LLM recommender run locally through [Ollama](https://ollama.com)
(`llama3.2`, `llama3.1:8b`, `glm4:9b`, `qwen2.5:14b`). The larger runs were done on the DIKU
Hendrix cluster with the Slurm scripts in `scripts/`; every script calls the same Python modules
and can be run locally as well.

### Data

MovieLens 1M is not included (its licence does not allow redistribution). Download it from
[GroupLens](https://grouplens.org/datasets/movielens/1m/) and place `users.dat`, `movies.dat`
and `ratings.dat` under `dataset/ml-1m/`. Processed data are written to `data/` (gitignored).

### What this repository contains

The code, configurations and cluster scripts, and the **aggregate** results of every experiment
in the report (AUC and balanced accuracy with CIs, effectiveness, shot curves, ablations, the
Romance swap test), together with the report's figures and tables in `docs/`.

Files derived from MovieLens at the level of individual users are **not** included, because
they would redistribute the dataset: the top-10 lists (`lists*.jsonl`), per-user attacker
predictions (`preds_*.jsonl`, `predictions` fields), the LLM recommender's raw outputs, the
trained checkpoints and embeddings, and the two few-shot example users
(`src/llm/few_shot_examples.json`, recreated by `src.llm.select_few_shot_examples`). All of them
are recreated by the pipeline below once ML-1M is in place; the figures and tables that read
them need this step before they can be redrawn.

## Pipeline

Every module documents its full arguments in its docstring (`uv run python -m <module> --help`).
The commands below use the BPR run (`EXP-003_bpr_d32_311`); the LightGCN run
(`EXP-004_lightgcn_d32_311`) is the same with `--model LightGCN` and `configs/lightgcn_ml1m_311.yaml`.

1. **Preprocess** (filters, 5-core, global temporal 3:1:1 split, post-split filter) →
   `data/ml-1m-311/` and Table 2:
   ```bash
   uv run python -m src.preprocess.ml1m --split global_311
   ```
2. **Train the CF recommenders** (RecBole, d = 32) → `runs/EXP-00x_*/` with checkpoint,
   user/item embeddings and `rec_metrics.json`:
   ```bash
   uv run python -m src.train.train_recbole --model BPR --config configs/bpr_ml1m_311.yaml --exp-id EXP-003_bpr_d32_311
   ```
3. **Export the top-10 lists** for the 500 attack test users (`lists.jsonl`) and the 3,367-user
   training pool (`lists_train_pool.jsonl`, with `--exclude-lists-jsonl`): `src.export.export_lists`.
4. **Classifier attackers** (tiers TH–T3, 4 classifiers × class-balanced or not):
   - `src.attack.list_representation_aia` → `attack_grid_500/` (T0, T1 incl. the reconstructed
     user embedding, T2, T3; also writes `summary.json`)
   - `src.attack.history_aia` → `history_aia/` (TH, the Slokom et al. attacker)
   - `src.attack.popularity_ablation_aia` → `popularity_ablation_500/`
5. **LLM attackers** (tier T4): `src.llm.list_aia` with `--style zero_shot | few_shot | staab |
   staab_few_shot | many_shot | history_zero_shot | ...` → `llm_aia/preds_*.jsonl` and
   `metrics_*.json`. The shot curve uses `scripts/hendrix_shot_sweep.sh` and
   `scripts/hendrix_shot_fill.sh`; the Romance swap test uses `src.llm.romance_intervention`
   (`build`, `scripts/hendrix_romance_intervention.sh`, `report`).
6. **LLM as the recommender** (`runs/EXP-006_llmrec_311/`): `src.rec.llm_retrieval_rec`
   `build-prompts` → `generate` (`scripts/hendrix_llmrec.sh`) → `postprocess`, with
   `--variant neutral | gender --n-request 20`; then `src.attack.llmrec_aia attack` (T2/T3) and
   `scripts/hendrix_llmrec_t4.sh` (T4).
7. **Report figures and tables** (read only from `runs/`):
   ```bash
   uv run python -m src.report.report_figures     # -> docs/report_figs/
   uv run python -m src.report.report_tables      # -> docs/report_tables/
   uv run python -m src.report.appendix_table --latex docs/report_tables/full_auc.tex
   uv run python -m src.report.effectiveness_by_gender
   ```

## Where each result in the report comes from

| Report | Content | Produced by | Reads |
| --- | --- | --- | --- |
| Figure 1 | Threat model | `report_figures --only 03_threat_model` | — |
| Figure 2 | AUC by attacker tier | `report_figures --only 04_ladder` | `attack_grid_500/`, `history_aia/`, `llm_aia/`, `shot_sweep/` |
| Figure 3 | Classifier comparison | `report_figures --only 05_classifiers` | `attack_grid_500/summary.json` |
| Figure 4 | Most-recommended vs other movies | `report_figures --only 06_popularity` | `popularity_ablation_500/` |
| Figure 5 | Already-watched masking | `report_figures --only 13_head_masking` | `lists*.jsonl`, `data/ml-1m-311/splits/train.csv` |
| Figure 6 | Classifier vs LLM, history vs list | `report_figures --only 07_attacker_x_input` | `attack_grid_500/`, `history_aia/`, `llm_aia/` |
| Figure 7 | AUC vs number of in-context examples | `report_figures --only 08_shots` | `shot_sweep/shot_sweep.json` (`src.report.shot_sweep_table`) |
| Figure 8 | BPR vs LightGCN for the LLM attacker; Romance swap | `report_figures --only 09_bpr_vs_lightgcn_llm` | `llm_aia/`, `shot_sweep/`, `romance_intervention/` |
| Figure 9 | Neutral vs gender prompt | `report_figures --only 10_neutral_vs_gender` | `EXP-006_llmrec_311/*_req20_*_unseen/` |
| Figure 10 | How the gender prompt changes the lists | `report_figures --only 11_romance_shift` | `EXP-006_llmrec_311/*_req20_*_unseen/lists.jsonl` |
| Figure 11 | Gender AUC by age group | `report_figures --only 12_age_slices` | `age_slices/age_slices.json` (`src.report.age_slice_table`) |
| Table 1 | ML-1M statistics | `src.preprocess.ml1m` | `data/ml-1m-311/stats.json` |
| Table 2 | Effectiveness and women − men gap | `report_tables` (`effectiveness`) | `runs/effectiveness_by_gender/` |
| Table 3 | List concentration | `report_tables` (`concentration`) | `lists*.jsonl` |
| Table 4 | AUC of every recommender and tier | `src.report.appendix_table --latex ...` | all of the above |
| Table 5 | Effectiveness change, neutral → gender | `report_tables` (`effectiveness_change`) | `runs/effectiveness_by_gender/` |
| Table 6 | LLM non-answers | `report_tables` (`nonanswers`) | `llm_aia/preds_*.jsonl` |
| Section 5.3.3 | Genre gaps between women's and men's lists | `report_tables` (`genre_gaps`) | `lists.jsonl` |

`report_figures` writes each figure as `docs/report_figs/<name>.pdf`; the report includes them
from `figs/`. Some tables were shortened by hand in the report (e.g. Table 2 without the CIs).

## Repository layout

```
configs/            RecBole configs (BPR, LightGCN on the global_311 split)
scripts/            Slurm job scripts for the LLM runs on the Hendrix cluster
src/
  preprocess/       ML-1M -> RecBole atomic files, splits, stats
  train/            RecBole training, embedding export
  export/           top-10 lists (lists.jsonl) for the attackers
  attack/           classifier attackers (TH-T3), popularity ablation, LLMRec attacks
  llm/              LLM attacker: prompts, backends, list_aia, Romance swap
  rec/              LLM as the recommender (Stairway LLMRec)
  report/           figures and tables for the report
runs/
  EXP-003_bpr_d32_311/        BPR: aggregate results of every attack
  EXP-004_lightgcn_d32_311/   LightGCN: same structure
  EXP-006_llmrec_311/         LLM recommender: attack summaries
  effectiveness_by_gender/    HR@10 / NDCG@10 by gender (Tables 2 and 5)
docs/               the report's figures (report_figs/) and tables (report_tables/)
```



## References

- Rampisela et al., *Stairway to Fairness* — preprocessing and the LLMRec setup:
  [github.com/theresiavr/stairway-to-fairness](https://github.com/theresiavr/stairway-to-fairness)
- Slokom et al. (2021), PerBlur — the interaction-history gender classifier:
  [github.com/SlokomManel/PerBlur](https://github.com/SlokomManel/PerBlur)
- Feng et al. (2025), RAID — attacks on user embeddings (no public code)
- Staab et al. (2024) — the guessing-game prompt of the LLM attacker

## Licence

The code is released under the MIT licence (see `LICENSE`). MovieLens 1M is subject to the
GroupLens terms of use.
