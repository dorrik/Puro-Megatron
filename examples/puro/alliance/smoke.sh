#!/bin/bash
# Puro-0.6B single-node smoke + throughput measurement.
#
#   smoke1.sh [nproc] [steps]
#
# Run through `srun --jobid=<held interactive allocation>`. Everything the
# recipe reads has to be exported INSIDE the container command: ct.sh only
# forwards variables it explicitly re-exports as APPTAINERENV_*.
set -euo pipefail
NPROC=${1:-1}
STEPS=${2:-20}
PURO_ROOT=${PURO_ROOT:-$SCRATCH/puro}
RUN=${RUN_NAME:-smoke${NPROC}}
MBS=${MICRO_BATCH_SIZE:-4}
GBS=${GLOBAL_BATCH_SIZE:-$(( MBS * NPROC * 4 ))}   # 4 grad-accum steps
FP8=${PURO_FP8:-0}

mkdir -p "$PURO_ROOT"/{cache,ckpt,tb}/"$RUN"

exec "$PURO_ROOT/ct.sh" "
set -e
export TRAIN_DATA_PATH=$PURO_ROOT/data_smoke/train
export VALID_DATA_PATH=$PURO_ROOT/data_smoke/valid
export DATA_CACHE_PATH=$PURO_ROOT/cache/$RUN
export SAVE_PATH=$PURO_ROOT/ckpt/$RUN
export TENSORBOARD_DIR=$PURO_ROOT/tb/$RUN
export MICRO_BATCH_SIZE=$MBS
export GLOBAL_BATCH_SIZE=$GBS
export PURO_FP8=$FP8
export PYTHONUNBUFFERED=1
export PYTHON=\$CTVENV/bin/python
export LAUNCHER=\"\$CTVENV/bin/python -m torch.distributed.run --standalone --nproc_per_node=$NPROC\"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2
exec $PURO_ROOT/Puro-Megatron/examples/puro/run_puro_0p6b.sh phase1-power \
  --train-samples $(( GBS * STEPS )) \
  --lr-warmup-samples $(( GBS * 2 )) \
  --log-interval 1 \
  --eval-iters 0 \
  --eval-interval 1000000 \
  --save-interval $(( STEPS / 2 )) \
  --non-persistent-save-interval 1000000 \
  --diag-interval 1000000
"
