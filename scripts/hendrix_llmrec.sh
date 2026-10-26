#!/bin/bash -l
# EXP-006: LLM as the recommender (Stairway to Fairness setup), 3,867 users
# (500 fixed attack test users + 3,367 train pool) per (model x prompt variant).
# Prompts are built on the laptop (data/ never goes to the cluster):
#   uv run python -m src.rec.llm_retrieval_rec build-prompts --variant {neutral,sensitive}
#   rsync -a src scripts hendrix:~/Fairness_project/
#   rsync -a runs/EXP-006_llmrec_311/prompts_*.jsonl hendrix:~/Fairness_project/runs/EXP-006_llmrec_311/
# Submit from ~/Fairness_project:  sbatch scripts/hendrix_llmrec.sh
# Smoke test:  LIMIT=5 sbatch --array=0-3 --time=0:30:00 scripts/hendrix_llmrec.sh
# Resumable: a re-submitted task skips users already in its responses file.
# History-length extension (last 50 train items instead of Stairway's 10):
#   uv run python -m src.rec.llm_retrieval_rec build-prompts --variant {neutral,sensitive} --history-len 50
#   HIST=50 sbatch scripts/hendrix_llmrec.sh
#SBATCH -p gpu
#SBATCH --job-name=llmrec
#SBATCH --array=0-3%4
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=24G   # 8G OOM-killed llama3.1:8b (2026-09-27)
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$HOME/Fairness_project"
mkdir -p logs
module load python/3.12.8   # cluster default python3 is 3.6.8

RUN=runs/EXP-006_llmrec_311
# SETTINGS = prompt files to generate for (prompts_<setting>.jsonl); tasks = models x settings.
# Default: Stairway's two variants. HIST=50 keeps the old shorthand for the hist50 extension.
# 2026-09-28 extension (gender-only prompt, 20 requested):
#   SETTINGS="neutral_req20 gender_req20 neutral_hist50_req20 gender_hist50_req20" sbatch --array=0-7%4 scripts/hendrix_llmrec.sh
HIST=${HIST:-10}
if [[ -z "${SETTINGS:-}" ]]; then
  SETTINGS="neutral sensitive"; (( HIST != 10 )) && SETTINGS="neutral_hist${HIST} sensitive_hist${HIST}"
fi
TASKS=()
for m in "glm4:9b glm4_9b" "llama3.1:8b llama31_8b"; do
  for st in $SETTINGS; do TASKS+=("$m $st"); done
done
if (( SLURM_ARRAY_TASK_ID >= ${#TASKS[@]} )); then
  echo "task $SLURM_ARRAY_TASK_ID out of range (${#TASKS[@]} tasks)"; exit 1
fi
read -r MODEL SLUG SETTING <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

OUT="$RUN/responses_${SETTING}_${SLUG}.jsonl"
EXTRA=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA=(--limit "$LIMIT")
  OUT="$RUN/smoke_responses_${SETTING}_${SLUG}.jsonl"
fi
echo "task $SLURM_ARRAY_TASK_ID: model=$MODEL setting=$SETTING out=$OUT"

export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=24h
PORT=$((12000 + RANDOM % 1000))
export OLLAMA_HOST=127.0.0.1:$PORT
"$HOME/ollama/bin/ollama" serve > /dev/null 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/api/tags" >/dev/null && break; sleep 2; done

.venv/bin/python -m src.rec.llm_retrieval_rec generate \
  --prompts "$RUN/prompts_${SETTING}.jsonl" --out "$OUT" \
  --model "$MODEL" --host "http://127.0.0.1:$PORT" \
  ${EXTRA[@]+"${EXTRA[@]}"}
