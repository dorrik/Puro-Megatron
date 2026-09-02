#!/usr/bin/env bash
# Puro-0.6B: the Puro recipe at the 0.6B scale used for the paper's own
# schedule sweeps (arXiv:2608.27370 Section F.2).
#
# Relationship to run_puro_2b.sh
#   Puro-2B  = Qwen3-1.7B backbone, untied embeddings  -> 2,031,739,904 params
#   Puro-0.6B = Qwen3-0.6B backbone                    ->   596,049,920 params
# The only architectural deltas are hidden 2048->1024 and FFN 6144->3072.
#
# Section F.2 pins the 0.6B sweep configuration:
#   "0.6B decoder-only model, BF16 training, sequence length 4,096, GBS 512,
#    MuonH multiplier 3, linear WSD decay"
# with effective peak LR swept over {0.008, 0.012, 0.016, 0.020, 0.024} and the
# best WSD decay ratio per peak being {0.4, 1.0, 0.8, 0.8, 1.0}. We take the
# 0.012 column: effective peak 0.012, decay ratio 1.0.
#
# Effective LR relation (paper Table 9 / Section H.3, verified against the
# phase2-constant-continuation recipe where lr 4.077248e-05 x m=10 = 4.08e-4):
#
#     effective LR = --lr  x  --muon-hyperball-lr-mult
#
# so an effective peak of 0.012 at m=3 means --lr 0.004.
#
# Token budget is TPP 20 (paper: "At 20 tokens per parameter (TPP)"):
#   596,049,920 x 20 = 11.92B tokens = 2,910,208 samples of length 4096.
#
# usage: run_puro_0p6b.sh {phase1-wsd|phase1-power} [extra args...]
set -euo pipefail

recipe=${1:?"usage: $0 {phase1-wsd|phase1-power} [extra args...]"}
shift

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PURO_MEGATRON_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
python_bin=${PYTHON:-python}
train_data=${TRAIN_DATA_PATH:?set TRAIN_DATA_PATH}
valid_data=${VALID_DATA_PATH:?set VALID_DATA_PATH}
data_cache=${DATA_CACHE_PATH:?set DATA_CACHE_PATH}
save_path=${SAVE_PATH:?set SAVE_PATH}
load_path=${LOAD_PATH:-$save_path}
tensorboard_dir=${TENSORBOARD_DIR:-tensorboard_logs/puro_0p6b_${recipe}}

# ---------------------------------------------------------------- knobs ----
# Puro-2B unties the embedding from the LM head. At 0.6B that would add a
# second 151936x1024 matrix (+155.6M) and take the model to 751.6M, so the
# default here keeps Qwen3-0.6B's tied embeddings and the model is genuinely
# 0.6B. Set PURO_UNTIE=1 for the Puro-2B convention.
untie=${PURO_UNTIE:-0}
# Section F.2 sweeps ran in BF16. Blockwise FP8 is the Puro-2B production
# setting; enable with PURO_FP8=1.
use_fp8=${PURO_FP8:-0}
mbs=${MICRO_BATCH_SIZE:-2}          # F.2 used MBS 2 at effective peaks 0.008/0.012
gbs=${GLOBAL_BATCH_SIZE:-512}       # F.2: "GBS 512"
lr_mult=${MUON_LR_MULT:-3.0}        # F.2: "MuonH multiplier 3"
eff_peak=${EFFECTIVE_PEAK_LR:-0.012}
decay_ratio=${WSD_DECAY_RATIO:-1.0} # F.2 best ratio at effective peak 0.012
tpp=${TPP:-20}
pp=${PIPELINE_PARALLEL_SIZE:-1}
tp=${TENSOR_PARALLEL_SIZE:-1}

if [[ "$untie" == "1" ]]; then
  n_params=751574528
else
  n_params=596049920
fi

# Warmup is not specified for the F.2 sweeps (the paper only pins it for the
# 2B production run, at 1000 steps). 200 steps ~= 3.5% of this horizon.
warmup_steps=${LR_WARMUP_STEPS:-200}

read -r train_samples warmup_samples wsd_decay_samples base_lr <<EOF
$(python3 - "$n_params" "$tpp" "$gbs" "$warmup_steps" "$decay_ratio" "$eff_peak" "$lr_mult" <<'PY'
import sys
n, tpp, gbs, warm_steps, ratio, eff, mult = (
    int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]),
    int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]),
)
seq = 4096
train_samples = round(n * tpp / seq / gbs) * gbs
warmup_samples = warm_steps * gbs
# WSD decay ratio is measured over the post-warmup budget.
wsd_decay_samples = round(ratio * (train_samples - warmup_samples) / gbs) * gbs
print(train_samples, warmup_samples, wsd_decay_samples, eff / mult)
PY
)
EOF

model_args=(
  --use-mcore-models
  --num-layers 28
  --hidden-size 1024          # Puro-2B: 2048
  --ffn-hidden-size 3072      # Puro-2B: 6144
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
  --init-method-std "${INIT_METHOD_STD:-0.02}"   # Qwen3-0.6B initializer_range
  --normalization RMSNorm
  --qk-layernorm
  --disable-bias-linear
  --attention-softmax-in-fp32
  --accumulate-allreduce-grads-in-fp32
)
if [[ "$untie" == "1" ]]; then
  model_args+=(--untie-embeddings-and-output-weights)
fi

optimizer_args=(
  --micro-batch-size "$mbs"
  --global-batch-size "$gbs"
  --clip-grad 1.0
  --weight-decay 0.1
  --optimizer muon_hyperball
  --muon-hyperball-scalar-optimizer adam
  --muon-hyperball-eps 1e-12
  --muon-hyperball-lr-mult "$lr_mult"
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
  --overlap-param-gather
  --tensor-model-parallel-size "$tp"
  --pipeline-model-parallel-size "$pp"
  --context-parallel-size 1
)
# --sequence-parallel is only meaningful with TP > 1.
if (( tp > 1 )); then
  optimizer_args+=(--sequence-parallel --use-tp-pp-dp-mapping)
fi
if [[ "$use_fp8" == "1" ]]; then
  optimizer_args+=(--fp8-format e4m3 --fp8-recipe blockwise)
fi
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
  --log-interval 10
  --tensorboard-log-interval 10
  --eval-iters 20
  --eval-interval 500
  --log-throughput
  --log-timers-to-tensorboard
  --timing-log-level 0
  --timing-log-option minmax
  --ckpt-format torch_dist
  --distributed-timeout-minutes 60
  --tensorboard-dir "$tensorboard_dir"
  --load "$load_path"
  --save "$save_path"
  --save-interval 500
  --non-persistent-save-interval 100
  --non-persistent-ckpt-type global
  --non-persistent-ckpt-num-to-keep 2
)

case "$recipe" in
  phase1-wsd)
    # Section F.2: linear WSD decay, the schedule the sweeps actually used.
    recipe_args=(
      --train-samples "$train_samples"
      --lr "$base_lr"
      --min-lr 0.0
      --lr-warmup-samples "$warmup_samples"
      --lr-decay-style WSD
      --lr-wsd-decay-style linear
      --lr-wsd-decay-samples "$wsd_decay_samples"
      --diag-interval 100
    )
    ;;
  phase1-power)
    # The open-ended power schedule from the 2B production Phase 1, rescaled.
    # tau = 1000 steps, p = 0.5, floor = 0.1 x peak (paper Eq. 10).
    recipe_args=(
      --train-samples "$train_samples"
      --lr "$base_lr"
      --min-lr "$(python3 -c "print(${base_lr}*0.1)")"
      --lr-warmup-samples "$warmup_samples"
      --lr-decay-style power
      --lr-power-decay-samples "$(( 1000 * gbs ))"
      --lr-power-exponent 0.5
      --diag-interval 100
    )
    ;;
  *)
    echo "unknown recipe: $recipe" >&2
    exit 2
    ;;
esac

echo "=== Puro-0.6B / $recipe ==="
echo "  params            : $n_params $( [[ $untie == 1 ]] && echo '(untied)' || echo '(tied)' )"
echo "  tokens (TPP $tpp)  : $(python3 -c "print(f'{$train_samples*4096/1e9:.2f}B')")"
echo "  train-samples     : $train_samples  ($(( train_samples / gbs )) steps @ GBS $gbs)"
echo "  effective peak LR : $eff_peak  (--lr $base_lr x m=$lr_mult)"
echo "  WSD decay ratio   : $decay_ratio  ($wsd_decay_samples samples)"
echo "  parallelism       : TP=$tp PP=$pp MBS=$mbs  precision=$( [[ $use_fp8 == 1 ]] && echo 'blockwise FP8' || echo 'BF16' )"

# The upstream 2B script execs "$python_bin" quoted, so PYTHON can only ever be a
# single word -- "torchrun --nproc_per_node=8" would be exec'd as one filename.
# LAUNCHER takes a full multi-word command and replaces the interpreter.
if [[ -n ${LAUNCHER:-} ]]; then
  read -r -a run_cmd <<< "$LAUNCHER"
else
  run_cmd=("$python_bin")
fi

exec "${run_cmd[@]}" "$repo_root/pretrain_gpt.py" \
  "${model_args[@]}" \
  "${optimizer_args[@]}" \
  "${data_args[@]}" \
  "${runtime_args[@]}" \
  "${recipe_args[@]}" \
  "$@"
