#!/bin/bash -l
# Many-shot in-context learning sweep: does the LLM's near-chance AUC close
# when it sees many labelled exemplars (the classifier sees ~3.4k)?
# One array task per (style x n_shots).  Submit from ~/Fairness_project.
#SBATCH -p gpu
#SBATCH --job-name=glm_shots
#SBATCH --array=0-5%2
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=24G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs
module load python/3.12.8

STYLES=(many_shot many_shot many_shot staab_many_shot staab_many_shot staab_many_shot)
SHOTS=(8 32 128 8 32 128)
STYLE=${STYLES[$SLURM_ARRAY_TASK_ID]}
N=${SHOTS[$SLURM_ARRAY_TASK_ID]}

export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=24h
PORT=$((11000 + RANDOM % 1000))
export OLLAMA_HOST=127.0.0.1:$PORT
"$HOME/ollama/bin/ollama" serve > /dev/null 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null && break; sleep 2; done

.venv/bin/python -m src.llm.list_aia \
  --lists runs/EXP-004_lightgcn_d32_311/lists.jsonl \
  --out-dir runs/EXP-004_lightgcn_d32_311/llm_aia \
  --style "$STYLE" --tag "lightgcn_glm4_9b_500_shots$N" \
  --model glm4:9b --host "http://127.0.0.1:$PORT" \
  --n-shots "$N" --shot-pool runs/EXP-004_lightgcn_d32_311/lists_train_pool.jsonl \
  --num-ctx 16384
