# Distributed Optimizer

The motivation for the distributed optimizer is to save memory by distributing the optimizer state evenly across data parallel ranks (https://arxiv.org/abs/1910.02054), versus the naive method of replicating the optimizer state across data parallel ranks.

Theoretical memory savings vary depending on the combination of the datatype of the model's parameters (`param_dtype`) and main gradients accumulated across data-parallel replicas (`grad_dtype`). We always use `fp32` main parameters for optimizer steps. In the current implementation, the theoretical number of bytes per parameter is (where d is the data parallel size):

|        | Non-distributed optim | Distributed optim |
| ------ | ------ | ------ |
| `fp16` parameters, `fp16` gradients | 20 | 4 + 16/d |
| `bf16` parameters, `fp32` gradients    | 18 | 6 + 12/d |
| `fp32` parameters, `fp32` gradients       | 16 | 8 + 8/d  |

Our implementation of the distributed optimizer uses contiguous buffers for parameters and main gradients; model gradients are copied over to the main gradients as soon as they are fully computed.

The figures below illustrate the distributed optimizer's sharding scheme, and the key steps of the distributed optimizer's parameter update:

## Data flow

![Data flow](../../images/distrib_optimizer/data_flow.png)

## Sharding scheme

![Sharding scheme](../../images/distrib_optimizer/sharding_scheme.png)

## Key steps

_(note: using illustrations above, assuming `bf16` model weights, `bf16` model gradients that are computed by the backward pass and `fp32` main gradients that are also used for optimizer steps; we always use `fp32` main weights for optimizer steps)_

- Backward pass finishes (gradient buffer holds 16 `fp32` gradient elements).
- Call reduce-scatter on each DP rank.
- Each DP rank now has 4 elements within the gradient buffer that are fully reduced (remaining 12 elements are garbage).
  - DP rank 0 has gradient values for elements [0:4].
  - DP rank 1 has gradient values for elements [4:8].
  - DP rank 2 has gradient values for elements [8:12].
  - DP rank 3 has gradient values for elements [12:16].
- Optimizer.step().
- Each DP rank copies its 4 `fp32` main parameter elements into the corresponding `bf16` parameter buffer (each element is cast from fp32 to fp16).
- Call all-gather on each DP rank.
- The parameter buffer now contains all 16, fully updated, `bf16` model parameter elements. Parameters in PyTorch modules already point to the appropriate locations in this parameter buffer, and thus forward passes are ready to run after the all-gather completes.
- At this point, the gradient buffer is also ready to be zero'd for the next iteration.

## Muon Layerwise Memory Optimization

Muon-family optimizers use a layer-wise distributed optimizer path when an emerging optimizer
such as `muon` or `muon_hyperball` is selected with `--use-distributed-optimizer`. This path is
different from the standard Adam distributed optimizer above: Muon matrix tensors are updated as
whole tensors, while scalar fallback parameters are normally assigned by tensor ownership.

`--layerwise-optimizer-memory-balance` enables a memory-aware assignment for this
layer-wise path. It is disabled by default and only affects Muon/MuonHyperball-style emerging
optimizers.

With the flag enabled:

- Muon and MuonHyperball parameters stay whole-tensor sharded. Their estimated optimizer-state
  cost is derived from the optimizer-owned tensors and their dtypes. In the common bf16 path this
  is `numel * 8` for fp32 main parameters plus fp32 momentum. Hyperball's scalar radius state is
  included in the estimate.
- AdamW fallback parameters are placed into an Adam-only logical flat buffer and split into
  variable-size contiguous ranges. The ranges are not equal-sized; they are chosen to compensate
  for the Muon-family cost already assigned to each rank. In the common bf16 path Adam fallback
  cost is `numel * 12` for fp32 main parameters, fp32 `exp_avg`, and fp32 `exp_avg_sq`.
- Adam range construction follows the same alignment ideas as the standard distributed optimizer:
  parameter starts are aligned to 64 elements, bucket ends to `lcm(dp_size, 128)`, and shard
  boundaries prefer 128-element alignment.
- Model parameter shapes are unchanged. Flattening is only used to define local Adam state and
  update ranges.

Example:

```bash
--optimizer muon_hyperball \
--use-distributed-optimizer \
--ckpt-format torch_dist \
--layerwise-optimizer-memory-balance
```

Use `torch_dist` checkpoints with this mode. The memory-balanced Adam fallback exposes its
optimizer state in the same per-parameter checkpoint layout used by the previous ping-pong
layer-wise Adam path, so `torch_dist` checkpoints saved without this flag can be migrated into
the variable flat shards on load. New `torch_dist` checkpoints saved with this flag also reload
through the same remapping path. Torch-format layer-wise optimizer checkpoints store rank-local
files such as `layer_wise_optimizer_<dp_rank>.pt`, so they cannot be safely remapped when
optimizer ownership changes across shard strategies.

### Checkpoint GPU memory behavior

The memory-balanced Adam fallback still reconstructs Adam checkpoint tensors on GPU. During
`torch_dist` save, each data-parallel group all-gathers the flat Adam ranges and rebuilds the
per-parameter `fp32_param`, `exp_avg`, and `exp_avg_sq` tensors on the model parameter device.
Those tensors are checkpoint temporaries and are released after the save path drops the state
dict references. This reconstruction can create a transient checkpoint peak, but it should not
leave a permanent CUDA allocation after checkpoint cleanup.

One subtle source of persistent memory is the optimizer step counter. Megatron stores a shared
optimizer step as `common_step` instead of repeating the same `step` value in every parameter
state. `common_step` is part of the checkpoint common state dict, not part of the sharded tensor
payload. The common state dict is validated by broadcasting rank 0's Python object to the other
ranks with `broadcast_object_list`. If rank 0's `common_step` is a CUDA tensor, the receiving
ranks deserialize a tensor tagged with rank 0's device, usually `cuda:0`. That creates a CUDA
context on GPU 0 in every training process on the node. The context is owned by the process and
does not disappear after `gc.collect()` or `torch.cuda.empty_cache()`; it is released only when
the process exits. The context is not an optimizer tensor. It is the CUDA runtime/driver,
allocator, communication, and library bookkeeping created when a process first touches a GPU.
The exact size depends on the GPU, driver, CUDA, PyTorch, and NCCL stack; on recent 8-GPU nodes
we have observed about 500 MiB per extra process context. If seven non-zero local ranks
accidentally create a context on GPU 0, this can leave roughly 3.5 GiB of persistent GPU 0
memory after the first checkpoint.

For that reason, checkpoint code keeps `common_step` on CPU before it enters the common state
dict. This does not move the Adam checkpoint tensors to CPU and does not change runtime optimizer
state. It only prevents a scalar common object from creating persistent GPU 0 context memory on
non-zero local ranks after the first checkpoint.
