#!/bin/bash -l
# LLM list-AIA with glm4:9b on the DIKU Hendrix cluster.
# One array task per prompt style; each runs its own Ollama server on a private port.
# Submit from ~/Fairness_project:  sbatch scripts/hendrix_glm_aia.sh
#SBATCH -p gpu
#SBATCH --job-name=glm_aia
#SBATCH --array=0-3%2
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=4G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs

module load python/3.12.8   # cluster default python3 is 3.6.8

STYLES=(zero_shot few_shot staab staab_few_shot)
STYLE=${STYLES[$SLURM_ARRAY_TASK_ID]}

export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=24h                 # don't unload the model between requests
PORT=$((11000 + RANDOM % 1000))              # avoid clashing with other jobs on the node
export OLLAMA_HOST=127.0.0.1:$PORT

"$HOME/ollama/bin/ollama" serve &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

# wait for the server instead of a fixed sleep
for _ in $(seq 60); do
    curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null && break
    sleep 2
done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python -m src.llm.list_aia \
  --lists runs/EXP-004_lightgcn_d32_311/lists.jsonl \
  --out-dir runs/EXP-004_lightgcn_d32_311/llm_aia \
  --style "$STYLE" --tag lightgcn_glm4_9b_500 \
  --model glm4:9b --host "http://127.0.0.1:$PORT"
