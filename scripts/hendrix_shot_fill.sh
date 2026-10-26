#!/bin/bash -l
# Fill the shot curve (slide 8 / report fig. "AUC vs shots") so that every point
# is 3 models x 3 exemplar draws, plain template (many_shot):
#   shots  2, 32, 64, 128   (8 and 16 already exist from hendrix_shot_sweep.sh /
#                            hendrix_bpr_llm_aia.sh)
#   seeds  2024 2025 2026   (random.sample prefix property: within a seed the
#                            2 / 8 / 16 / 32 / 64 / 128 exemplars are nested)
#   models glm4:9b, llama3.1:8b, qwen2.5:14b
# Skipped: glm4:9b 32 / 128 shots seed 2024 (exist without the _s suffix).
# The 2-shot points become random draws from the train pool, replacing the
# fixed `few_shot` fixture in the curve (that run stays on disk, unchanged).
#
# REC=lightgcn (default) or REC=bpr selects the recommender's lists + train pool.
# Submit from ~/Fairness_project:
#   sbatch scripts/hendrix_shot_fill.sh
#   REC=bpr sbatch scripts/hendrix_shot_fill.sh
# Smoke test:  LIMIT=5 sbatch --array=0,9 --time=0:30:00 scripts/hendrix_shot_fill.sh
#SBATCH -p gpu
#SBATCH --job-name=shot_fill
#SBATCH --array=0-33%4
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=24G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs
module load python/3.12.8   # cluster default python3 is 3.6.8

REC=${REC:-lightgcn}
case "$REC" in
  lightgcn) RUN=runs/EXP-004_lightgcn_d32_311 ;;
  bpr)      RUN=runs/EXP-003_bpr_d32_311 ;;
  *) echo "unknown REC=$REC"; exit 1 ;;
esac

# ---- task list: "model slug shots seed"
TASKS=()
for m in "glm4:9b glm4_9b" "llama3.1:8b llama31_8b" "qwen2.5:14b qwen25_14b"; do
  for n in 2 32 64 128; do
    for seed in 2024 2025 2026; do
      [[ "$m" == "glm4:9b glm4_9b" && ( $n == 32 || $n == 128 ) && $seed == 2024 ]] && continue
      TASKS+=("$m $n $seed")
    done
  done
done
if (( SLURM_ARRAY_TASK_ID >= ${#TASKS[@]} )); then
  echo "task $SLURM_ARRAY_TASK_ID out of range (${#TASKS[@]} tasks)"; exit 1
fi
read -r MODEL SLUG N SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

STYLE=many_shot
TAG="${REC}_${SLUG}_500_shots${N}_s${SEED}"
EXTRA=(--n-shots "$N" --shot-seed "$SEED" --shot-pool "$RUN/lists_train_pool.jsonl")
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA+=(--limit "$LIMIT")
  TAG="smoke_${TAG}"
fi
echo "task $SLURM_ARRAY_TASK_ID: rec=$REC model=$MODEL shots=$N seed=$SEED tag=$TAG"

# ---- private Ollama server on a random port
export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=24h
PORT=$((15000 + RANDOM % 1000))
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
