#!/bin/bash -l
# LLM AIA on the user's watched history (100 most recent titles) instead of the
# rec list — the missing "history x LLM" cell of the attacker matrix.
# Threat model: an insider who sees the raw profile, i.e. an upper bound, not the
# deployment-realistic list-only attacker.
#SBATCH -p gpu
#SBATCH --job-name=glm_hist
#SBATCH --array=0-1%2
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=24G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs
module load python/3.12.8

STYLES=(history_zero_shot history_staab)
STYLE=${STYLES[$SLURM_ARRAY_TASK_ID]}

export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=24h
PORT=$((11000 + RANDOM % 1000))
export OLLAMA_HOST=127.0.0.1:$PORT
"$HOME/ollama/bin/ollama" serve > /dev/null 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null && break; sleep 2; done

.venv/bin/python -m src.llm.list_aia \
  --lists runs/EXP-004_lightgcn_d32_311/lists_hist100.jsonl \
  --out-dir runs/EXP-004_lightgcn_d32_311/llm_aia \
  --style "$STYLE" --tag lightgcn_glm4_9b_500_hist100 \
  --model glm4:9b --host "http://127.0.0.1:$PORT" --num-ctx 8192
