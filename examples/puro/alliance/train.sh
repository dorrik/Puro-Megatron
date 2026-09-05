#!/bin/bash
#SBATCH --account=def-dorrik_gpu
#SBATCH --gpus-per-node=h100:4   # override at submit time on 8-GPU clusters
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --job-name=puro06b
#SBATCH --output=/scratch/dorrik/puro/logs/train_%j.out
# Node count comes from -N at submit time. On Fir (4 H100/node):
#   sbatch -N2  ->  8 GPUs      sbatch -N8  -> 32 GPUs
# On Nibi/Killarney (8 H100/node): sbatch -N1 --gpus-per-node=h100:8 -> 8 GPUs
# No --partition: let the scheduler pick from the requested resources.

set -euo pipefail
PURO_ROOT=$SCRATCH/puro
RUN=${RUN_NAME:-puro06b_p1}
RECIPE=${RECIPE:-phase1-power}
DATA=${DATA_DIR:-$PURO_ROOT/data_p1}
cd "$PURO_ROOT"
mkdir -p logs

# Jobs inherit the submitting shell's environment; Alliance sets both of these
# at login and inside the container they shadow the image's python stack.
unset PYTHONPATH PIP_CONFIG_FILE
module purge 2>/dev/null || true

export TRAIN_DATA_PATH=$DATA/train
export VALID_DATA_PATH=$DATA/valid
export DATA_CACHE_PATH=$PURO_ROOT/cache/$RUN
export SAVE_PATH=$PURO_ROOT/ckpt/$RUN
export LOAD_PATH=$PURO_ROOT/ckpt/$RUN     # resume in place across requeues
export TENSORBOARD_DIR=$PURO_ROOT/tb/$RUN
mkdir -p "$DATA_CACHE_PATH" "$SAVE_PATH" "$TENSORBOARD_DIR"

export PURO_FP8=${PURO_FP8:-1}            # blockwise FP8: Puro's production setting
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}

# Slurm reports this, so the same script works on 4-GPU (Fir, Rorqual,
# Trillium) and 8-GPU (Nibi, Killarney) nodes without editing.
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

echo "job=$SLURM_JOB_ID nodes=$SLURM_NNODES world=$WORLD master=$MASTER_ADDR start=$(date -Is)"
echo "recipe=$RECIPE fp8=$PURO_FP8 mbs=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE"
# GBS must divide evenly across (world x micro-batch) or Megatron rejects it.
if (( GLOBAL_BATCH_SIZE % (WORLD * MICRO_BATCH_SIZE) != 0 )); then
  echo "!! global batch $GLOBAL_BATCH_SIZE not divisible by world*mbs=$((WORLD*MICRO_BATCH_SIZE))" >&2
  exit 1
fi

# \$SLURM_NODEID stays escaped so it resolves inside each srun task, not here.
srun --ntasks-per-node=1 --unbuffered --export=ALL "$PURO_ROOT/ct.sh" '
  # Launch via the container python: the host ~/.local/bin/torchrun is on PATH
  # here (bind-mounted /home) and cannot execute inside the image.
  export LAUNCHER="$CTVENV/bin/python -m torch.distributed.run \
    --nnodes='"$SLURM_NNODES"' --nproc_per_node='"$GPUS_PER_NODE"' \
    --node_rank=$SLURM_NODEID --master_addr='"$MASTER_ADDR"' --master_port='"$MASTER_PORT"'"
  export PYTHON=$CTVENV/bin/python
  exec '"$PURO_ROOT"'/Puro-Megatron/examples/puro/run_puro_0p6b.sh '"$RECIPE"'
'
rc=$?
echo "exit=$rc end=$(date -Is)"

# COMPLETED is not evidence the run worked; count the artifacts it should have left.
echo "=== checkpoints ==="
ls -d "$SAVE_PATH"/iter_* 2>/dev/null | tail -3 || echo "!! no checkpoints written"
exit $rc
