#!/usr/bin/env bash
set -euo pipefail

recipe=${1:?"usage: $0 {phase1-power|phase2-transition|phase2-constant-continuation} [extra args...]"}
shift

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PURO_MEGATRON_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
python_bin=${PYTHON:-python}
train_data=${TRAIN_DATA_PATH:?set TRAIN_DATA_PATH}
valid_data=${VALID_DATA_PATH:?set VALID_DATA_PATH}
data_cache=${DATA_CACHE_PATH:?set DATA_CACHE_PATH}
load_path=${LOAD_PATH:?set LOAD_PATH}
save_path=${SAVE_PATH:?set SAVE_PATH}
tensorboard_dir=${TENSORBOARD_DIR:-tensorboard_logs/puro_${recipe}}

model_args=(
  --use-mcore-models
  --num-layers 28
  --hidden-size 2048
  --ffn-hidden-size 6144
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 8
  --kv-channels 128
  --seq-length 4096
  --max-position-embeddings 4096
  --position-embedding-type rope
  --rotary-base 10000
  --rotary-percent 1.0
  --attention-dropout 0.0
  --attention-type thd
  --hidden-dropout 0.0
  --swiglu
  --init-method-std 0.018
  --normalization RMSNorm
  --qk-layernorm
  --untie-embeddings-and-output-weights
  --disable-bias-linear
  --attention-softmax-in-fp32
  --accumulate-allreduce-grads-in-fp32
)

optimizer_args=(
  --micro-batch-size 2
  --global-batch-size 1536
  --clip-grad 1.0
  --weight-decay 0.1
  --optimizer muon_hyperball
  --muon-hyperball-scalar-optimizer adam
  --muon-hyperball-eps 1e-12
  --muon-hyperball-lr-mult 10.0
  --muon-qkv-norm-mode separate
  --muon-swiglu-norm-mode separate
  --muon-qkv-ns-mode separate
  --muon-swiglu-ns-mode separate
  --adam-beta1 0.9
  --adam-beta2 0.95
  --bf16
  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl te
  --calculate-per-token-loss
  --manual-gc
  --rerun-mode skip
  --use-distributed-optimizer
  --layerwise-optimizer-memory-balance
  --overlap-grad-reduce
  --fp8-format e4m3
  --fp8-recipe blockwise
  --tensor-model-parallel-size 1
  --context-parallel-size 1
  --sequence-parallel
  --use-tp-pp-dp-mapping
)

if [[ -n ${MUON_HYPERBALL_RMS:-} ]]; then
  optimizer_args+=(--muon-hyperball-rms "$MUON_HYPERBALL_RMS")
fi

data_args=(
  --train-data-path "$train_data"
  --valid-data-path "$valid_data"
  --tokenizer-type NullTokenizer
  --vocab-size 151936
  --data-cache-path "$data_cache"
  --tiktoken-pattern v2
  --eos-token-id 151645
  --no-mmap-bin-files
  --num-workers 0
)

runtime_args=(
  --log-interval 1
  --tensorboard-log-interval 1
  --eval-iters 5
  --eval-interval 200
  --log-throughput
  --log-timers-to-tensorboard
  --timing-log-level 0
  --timing-log-option minmax
  --ckpt-format torch_dist
  --distributed-timeout-minutes 60
  --tensorboard-dir "$tensorboard_dir"
  --load "$load_path"
  --save "$save_path"
)

case "$recipe" in
  phase1-power)
    recipe_args=(
      --train-samples 107271866
      --lr-warmup-samples 1536000
      --lr 0.005
      --min-lr 0.0005
      --lr-decay-style power
      --lr-power-decay-samples 1536000
      --lr-power-exponent 0.5
      --pipeline-model-parallel-size 2
      --pipeline-model-parallel-layout 'Et*18|t*10L'
      --overlap-param-gather
      --diag-interval 100
      --save-interval 2500
      --non-persistent-save-interval 500
      --non-persistent-ckpt-type global
    )
    ;;
  phase2-transition)
    recipe_args=(
      --train-samples 341867221
      --lr 0.00103847632916
      --min-lr 1e-5
      --lr-decay-style linear
      --lr-warmup-samples 0
      --lr-decay-samples 234596053
      --override-opt-param-scheduler
      --reset-opt-param-scheduler-progress
      --reset-train-dataloader-progress
      --pipeline-model-parallel-size 4
      --pipeline-model-parallel-layout 'Et*9|t*9|t*9|tL'
      --diag-interval 1
      --dist-ckpt-strictness assume_ok_unexpected
      --use-checkpoint-args
      --no-load-rng
      --exit-on-missing-checkpoint
      --save-interval 2500
      --non-persistent-save-interval 500
      --non-persistent-ckpt-type global
    )
    ;;
  phase2-constant-continuation)
    recipe_args=(
      --train-samples 341867221
      --lr 4.077248127291725e-05
      --min-lr 4.077248127291725e-05
      --lr-decay-style constant
      --lr-warmup-samples 0
      --lr-decay-samples 234596053
      --override-opt-param-scheduler
      --phase-transition-iterations 69838
      --pipeline-model-parallel-size 4
      --pipeline-model-parallel-layout 'Et*9|t*9|t*9|tL'
      --overlap-param-gather
      --diag-interval 100
      --dist-ckpt-strictness assume_ok_unexpected
      --exit-on-missing-checkpoint
      --ckpt-step "${CKPT_STEP:-218000}"
      --save-interval 500
      --non-persistent-save-interval 100
      --non-persistent-ckpt-type global
      --non-persistent-ckpt-num-to-keep 4
    )
    ;;
  *)
    echo "unknown recipe: $recipe" >&2
    exit 2
    ;;
esac

exec "$python_bin" "$repo_root/pretrain_gpt.py" \
  "${model_args[@]}" \
  "${optimizer_args[@]}" \
  "${data_args[@]}" \
  "${runtime_args[@]}" \
  "${recipe_args[@]}" \
  "$@"
