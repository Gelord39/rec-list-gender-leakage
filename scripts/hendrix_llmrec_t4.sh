#!/bin/bash -l
# EXP-006 T4: an LLM (GLM-4-9B and/or Llama-3.1-8B) as the attacker on LLM-generated rec lists (unseen version),
# 500 fixed test users. zero_shot + staab_many_shot with 8 exemplars drawn from the
# SAME run's train pool (the fixed few_shot fixture holds LightGCN lists, not LLMRec ones).
# Lists are produced on the laptop:
#   rsync -a --include='*/' --include='*_unseen/lists*.jsonl' --exclude='*' \
#     runs/EXP-006_llmrec_311/ hendrix:~/Fairness_project/runs/EXP-006_llmrec_311/
# Submit from ~/Fairness_project:  sbatch scripts/hendrix_llmrec_t4.sh
#SBATCH -p gpu
#SBATCH --job-name=llmrec_t4
#SBATCH --array=0-7%4
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=24G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs
module load python/3.12.8

# Tasks = RUNS x attacker models x styles. Defaults reproduce the 2026-09-27 GLM-only run.
# SETTING="" (Stairway 10-item history) or SETTING=_hist50:  SETTING=_hist50 sbatch scripts/hendrix_llmrec_t4.sh
# 2026-09-28 extension (neutral vs gender-only, 20 requested -> 10 kept), GLM + Llama attackers:
#   RUNS="neutral_req20_glm4_9b_unseen gender_req20_glm4_9b_unseen ..." \
#   ATTACKERS="glm4:9b,glm4_9b llama3.1:8b,llama31_8b" sbatch --array=0-<4*n_runs-1>%4 scripts/hendrix_llmrec_t4.sh
S=${SETTING:-}
read -r -a RUN_LIST <<< "${RUNS:-neutral${S}_glm4_9b_unseen sensitive${S}_glm4_9b_unseen neutral${S}_llama31_8b_unseen sensitive${S}_llama31_8b_unseen}"
read -r -a ATT_LIST <<< "${ATTACKERS:-glm4:9b,glm4_9b}"
STYLES=(zero_shot staab_many_shot)
TASKS=()
for r in "${RUN_LIST[@]}"; do
  for a in "${ATT_LIST[@]}"; do
    for st in "${STYLES[@]}"; do TASKS+=("$r ${a/,/ } $st"); done
  done
done
if (( SLURM_ARRAY_TASK_ID >= ${#TASKS[@]} )); then
  echo "task $SLURM_ARRAY_TASK_ID out of range (${#TASKS[@]} tasks)"; exit 1
fi
read -r RUN_NAME MODEL ASLUG STYLE <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
DIR=runs/EXP-006_llmrec_311/$RUN_NAME

EXTRA=()
TAG=${ASLUG}_500
if [[ $STYLE == staab_many_shot ]]; then
  EXTRA=(--n-shots 8 --shot-seed 2024 --shot-pool "$DIR/lists_train_pool.jsonl")
  TAG=${ASLUG}_500_shots8
fi
echo "task $SLURM_ARRAY_TASK_ID: run=$RUN_NAME attacker=$MODEL style=$STYLE tag=$TAG"

export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=24h
PORT=$((13000 + RANDOM % 1000))
export OLLAMA_HOST=127.0.0.1:$PORT
"$HOME/ollama/bin/ollama" serve > /dev/null 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null && break; sleep 2; done

.venv/bin/python -m src.llm.list_aia \
  --lists "$DIR/lists.jsonl" --out-dir "$DIR/llm_aia" \
  --style "$STYLE" --tag "$TAG" \
  --model "$MODEL" --host "http://127.0.0.1:$PORT" \
  --num-ctx 16384 \
  ${EXTRA[@]+"${EXTRA[@]}"}
