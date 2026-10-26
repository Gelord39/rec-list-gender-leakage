#!/bin/bash -l
# Shot-count x exemplar-draw x model sweep (meeting 2026-09-21 action items:
# "try llama 8b, qwen big model, 8-16 shots" + "different shot, 3 times").
#
# Grid (EXP-004 LightGCN lists, 500 fixed test users):
#   models    glm4:9b, llama3.1:8b, qwen2.5:14b
#   templates many_shot, staab_many_shot
#   shots     8, 16            (same seed => the 16 exemplars start with the 8)
#   seeds     2024 2025 2026   (exemplar draw; decoding is temperature 0)
# + 0-shot anchors (zero_shot, staab) for the two new models.
# glm4:9b / 8 shots / seed 2024 already exists (tag ..._shots8, 2026-09-21) and is skipped.
#
# Submit from ~/Fairness_project:   sbatch scripts/hendrix_shot_sweep.sh
# Smoke test (5 users, first N tasks): LIMIT=5 sbatch --array=0-3 --time=0:30:00 scripts/hendrix_shot_sweep.sh
#SBATCH -p gpu
#SBATCH --job-name=shot_sweep
#SBATCH --array=0-37%4
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=24G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs
module load python/3.12.8   # cluster default python3 is 3.6.8

RUN=runs/EXP-004_lightgcn_d32_311

# ---- build the task list: "model slug style shots seed" (shots/seed = 0 for anchors)
TASKS=()
for m in "llama3.1:8b llama31_8b" "qwen2.5:14b qwen25_14b"; do
  for s in zero_shot staab; do TASKS+=("$m $s 0 0"); done
done
for m in "glm4:9b glm4_9b" "llama3.1:8b llama31_8b" "qwen2.5:14b qwen25_14b"; do
  for s in many_shot staab_many_shot; do
    for n in 8 16; do
      for seed in 2024 2025 2026; do
        [[ "$m" == "glm4:9b glm4_9b" && $n == 8 && $seed == 2024 ]] && continue
        TASKS+=("$m $s $n $seed")
      done
    done
  done
done
if (( SLURM_ARRAY_TASK_ID >= ${#TASKS[@]} )); then
  echo "task $SLURM_ARRAY_TASK_ID out of range (${#TASKS[@]} tasks)"; exit 1
fi
read -r MODEL SLUG STYLE N SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

EXTRA=()
if (( N > 0 )); then
  TAG="lightgcn_${SLUG}_500_shots${N}_s${SEED}"
  EXTRA=(--n-shots "$N" --shot-seed "$SEED" --shot-pool "$RUN/lists_train_pool.jsonl")
else
  TAG="lightgcn_${SLUG}_500"
fi
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA+=(--limit "$LIMIT")
  TAG="smoke_${TAG}"
fi
echo "task $SLURM_ARRAY_TASK_ID: model=$MODEL style=$STYLE shots=$N seed=$SEED tag=$TAG"

# ---- private Ollama server on a random port
export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=24h
PORT=$((11000 + RANDOM % 1000))
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
