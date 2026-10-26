#!/bin/bash -l
# Romance intervention (src/llm/romance_intervention.py): GLM-4-9B zero-shot on the
# original / Romance-swapped / control-swapped lists of BPR and LightGCN, 500 users each.
# Lists are built on the laptop:
#   uv run python -m src.llm.romance_intervention build --run-dir runs/EXP-00{3,4}_*
#   rsync -a runs/EXP-003_bpr_d32_311/romance_intervention runs/EXP-004_lightgcn_d32_311/romance_intervention ...
# Submit from ~/Fairness_project:  sbatch scripts/hendrix_romance_intervention.sh
#SBATCH -p gpu
#SBATCH --job-name=romance_iv
#SBATCH --array=0-9%5
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs
module load python/3.12.8   # cluster default python3 is 3.6.8

TASKS=()
for rp in "runs/EXP-003_bpr_d32_311 bpr" "runs/EXP-004_lightgcn_d32_311 lightgcn"; do
  for arm in original romance_swap control_swap_s2024 control_swap_s2025 control_swap_s2026; do
    TASKS+=("$rp $arm")
  done
done
read -r RUN PREFIX ARM <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
DIR=$RUN/romance_intervention
echo "task $SLURM_ARRAY_TASK_ID: $RUN $ARM"

export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=24h
PORT=$((15000 + RANDOM % 1000))
export OLLAMA_HOST=127.0.0.1:$PORT
"$HOME/ollama/bin/ollama" serve > /dev/null 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null && break; sleep 2; done

.venv/bin/python -m src.llm.list_aia \
  --lists "$DIR/lists_${ARM}.jsonl" --out-dir "$DIR/llm_aia" \
  --style zero_shot --tag "${PREFIX}_glm4_9b_500_${ARM}" \
  --model glm4:9b --host "http://127.0.0.1:$PORT"
