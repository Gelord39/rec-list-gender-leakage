#!/bin/bash -l
# BPR (EXP-003) catch-up: every LLM list-AIA run that exists for LightGCN (EXP-004)
# on the same 500 fixed test users, so the two recommenders are comparable on T4.
#   0-3    glm4:9b      zero_shot / few_shot / staab / staab_few_shot (2 fixed exemplars)
#   4-7    llama3.1:8b, qwen2.5:14b   zero_shot / staab anchors
#   8-43   3 models x {many_shot, staab_many_shot} x {8, 16} shots x seeds {2024, 2025, 2026}
#   44-47  glm4:9b {many_shot, staab_many_shot} x {32, 128} shots, seed 2024
# Exemplars for many-shot come from BPR's own train pool. llama3.2 runs on the laptop.
# Tags: bpr_{slug}_500[_shots{N}[_s{seed}]] (the 32/128-shot runs have no _s suffix, as for LightGCN).
#
# Lists are produced on the laptop:
#   rsync -a runs/EXP-003_bpr_d32_311/lists.jsonl runs/EXP-003_bpr_d32_311/lists_train_pool.jsonl \
#     hendrix:~/Fairness_project/runs/EXP-003_bpr_d32_311/
# Submit from ~/Fairness_project:  sbatch scripts/hendrix_bpr_llm_aia.sh
# Smoke test:  LIMIT=5 sbatch --array=0,8 --time=0:30:00 scripts/hendrix_bpr_llm_aia.sh
#SBATCH -p gpu
#SBATCH --job-name=bpr_aia
#SBATCH --array=0-47%4
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=24G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs
module load python/3.12.8   # cluster default python3 is 3.6.8

RUN=runs/EXP-003_bpr_d32_311
PREFIX=bpr

# ---- task list: "model slug style shots seed" (shots 0 = no sampled exemplars, seed 0 = no _s suffix)
TASKS=()
for s in zero_shot few_shot staab staab_few_shot; do TASKS+=("glm4:9b glm4_9b $s 0 0"); done
for m in "llama3.1:8b llama31_8b" "qwen2.5:14b qwen25_14b"; do
  for s in zero_shot staab; do TASKS+=("$m $s 0 0"); done
done
for m in "glm4:9b glm4_9b" "llama3.1:8b llama31_8b" "qwen2.5:14b qwen25_14b"; do
  for s in many_shot staab_many_shot; do
    for n in 8 16; do
      for seed in 2024 2025 2026; do TASKS+=("$m $s $n $seed"); done
    done
  done
done
for s in many_shot staab_many_shot; do
  for n in 32 128; do TASKS+=("glm4:9b glm4_9b $s $n 0"); done
done
if (( SLURM_ARRAY_TASK_ID >= ${#TASKS[@]} )); then
  echo "task $SLURM_ARRAY_TASK_ID out of range (${#TASKS[@]} tasks)"; exit 1
fi
read -r MODEL SLUG STYLE N SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

TAG="${PREFIX}_${SLUG}_500"
EXTRA=()
if (( N > 0 )); then
  TAG="${TAG}_shots${N}"
  (( SEED > 0 )) && TAG="${TAG}_s${SEED}"
  EXTRA=(--n-shots "$N" --shot-seed "$(( SEED > 0 ? SEED : 2024 ))" --shot-pool "$RUN/lists_train_pool.jsonl")
fi
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA+=(--limit "$LIMIT")
  TAG="smoke_${TAG}"
fi
echo "task $SLURM_ARRAY_TASK_ID: model=$MODEL style=$STYLE shots=$N seed=$SEED tag=$TAG"

# ---- private Ollama server on a random port
export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=24h
PORT=$((14000 + RANDOM % 1000))
export OLLAMA_HOST=127.0.0.1:$PORT
"$HOME/ollama/bin/ollama" serve > /dev/null 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null && break; sleep 2; done

.venv/bin/python -m src.llm.list_aia \
  --lists "$RUN/lists.jsonl" \
  --out-dir "$RUN/llm_aia" \
  --style "$STYLE" --tag "$TAG" \
  --model "$MODEL" --host "http://127.0.0.1:$PORT" \
  --num-ctx 16384 \
  ${EXTRA[@]+"${EXTRA[@]}"}
