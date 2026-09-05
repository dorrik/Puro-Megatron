#!/bin/bash
# Measure throughput across precision / micro-batch on one GPU, so the campaign
# is sized on measurement rather than on assumed peak FLOPs.
set -uo pipefail
cd "$SCRATCH/puro"
RESULT=logs/sweep_results.txt
: > "$RESULT"

# Override with e.g. CONFIGS="1 8|1 16" to re-run a subset.
IFS="|" read -ra CFGS <<< "${CONFIGS:-0 4|0 8|1 4|1 8|1 16}"
for cfg in "${CFGS[@]}"; do
  set -- $cfg
  FP8=$1; MBS=$2
  NAME="fp8${FP8}_mbs${MBS}"
  echo "=== $NAME ===" | tee -a "$RESULT"
  PURO_FP8=$FP8 MICRO_BATCH_SIZE=$MBS RUN_NAME="sw_$NAME" \
    srun -A def-dorrik_gpu -N1 --gpus-per-node=h100:1 -c 12 --mem=100G \
      -t 0:20:0 -J puro-sweep --unbuffered \
      ./smoke.sh 1 12 > "logs/sweep_$NAME.log" 2>&1
  # Median of the steady-state iterations; the first two include warmup.
  tput=$(grep -o "throughput per GPU (TFLOP/s/GPU): *[0-9.]*" "logs/sweep_$NAME.log" \
         | awk '{print $NF}' | tail -8 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  mem=$(grep -o "max allocated: *[0-9.]*" "logs/sweep_$NAME.log" | awk '{print $NF}' | sort -n | tail -1)
  oom=$(grep -c "out of memory" "logs/sweep_$NAME.log")
  echo "  throughput=${tput:-FAILED} TFLOP/s/GPU  max_mem=${mem:-?} MB  oom=$oom" | tee -a "$RESULT"
  rm -rf "ckpt/sw_$NAME" "cache/sw_$NAME"
done

echo "=== summary ===" | tee -a "$RESULT"
cat "$RESULT"
