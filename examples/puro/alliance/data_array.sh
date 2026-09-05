#!/bin/bash
#SBATCH --account=def-dorrik_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --job-name=puro-data
#SBATCH --output=/scratch/dorrik/puro/logs/data_%A_%a.out
# Array size is supplied at submit time with --array.
# No --partition: let the scheduler choose from the requested resources.

set -euo pipefail
cd "$SCRATCH/puro"
# Jobs inherit the submitting shell's env; Alliance sets both of these at login
# and they shadow the venv's python stack.
unset PYTHONPATH PIP_CONFIG_FILE
export HF_HOME="$SCRATCH/.hf"
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export PYTHONUNBUFFERED=1   # Slurm buffers output in blocks and drops the tail on cancel

OUT=${OUT_DIR:-$SCRATCH/puro/data_p1}
NTASKS=${NUM_TASKS:-$SLURM_ARRAY_TASK_COUNT}

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID/$NTASKS cpus=$SLURM_CPUS_PER_TASK start=$(date -Is)"
df -h "$SCRATCH" | tail -1

./dataenv/bin/python Puro-Megatron/examples/puro/data/build_puro_data.py run \
  --out-dir "$OUT" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --num-tasks "$NTASKS" \
  --workers "$SLURM_CPUS_PER_TASK"

echo "done=$(date -Is)"
echo "=== shards produced by this task ==="
ls "$OUT/shards" 2>/dev/null | wc -l
du -sh "$OUT/shards" 2>/dev/null || true
# Parquet is deleted as each file is consumed; this should stay small.
du -sh "$OUT/_parquet" 2>/dev/null || echo "  (parquet cache already cleaned)"
