#!/bin/bash -l
# Validation of the 2026-09-27 finding "exemplar draws whose labels agree with
# their list's gender signal (high separation) make better many-shot LLM attackers".
# 16 shots, all draws with exactly 4 female exemplars; 5 seeds each of low / mid /
# high separation, chosen by src/llm/exemplar_quality.py (manifest:
# runs/EXP-004_lightgcn_d32_311/exemplar_quality/manifest_16shot.json).
# plain many_shot template only (one-word answers, ~5 min per run), 3 models.
#
# Submit from ~/Fairness_project:   sbatch scripts/hendrix_exemplar_quality.sh
#SBATCH -p gpu
#SBATCH --job-name=exq
#SBATCH --array=0-44%4
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs
module load python/3.12.8

# RUN / PREFIX select the recommender; seeds are read from that run's manifest.
#   BPR: RUN=runs/EXP-003_bpr_d32_311 PREFIX=bpr sbatch scripts/hendrix_exemplar_quality.sh
RUN=${RUN:-runs/EXP-004_lightgcn_d32_311}
PREFIX=${PREFIX:-lightgcn}
# low, mid, high separation, 5 each (LightGCN: low 11148 11883 10969 11434 11227,
# mid 10211 11906 10429 10362 10893, high 11614 10555 11044 11800 11812)
read -r -a SEEDS <<< "$(.venv/bin/python -c "import json; g=json.load(open('$RUN/exemplar_quality/manifest_16shot.json'))['groups']; print(' '.join(str(d['seed']) for k in ('low','mid','high') for d in g[k]))")"
MODELS=("glm4:9b glm4_9b" "llama3.1:8b llama31_8b" "qwen2.5:14b qwen25_14b")

M=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
S=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
read -r MODEL SLUG <<< "${MODELS[$M]}"
SEED=${SEEDS[$S]}
TAG="${PREFIX}_${SLUG}_500_shots16_s${SEED}"
echo "task $SLURM_ARRAY_TASK_ID: model=$MODEL seed=$SEED tag=$TAG"

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
  --style many_shot --tag "$TAG" \
  --model "$MODEL" --host "http://127.0.0.1:$PORT" \
  --num-ctx 16384 \
  --n-shots 16 --shot-seed "$SEED" --shot-pool "$RUN/lists_train_pool.jsonl"
