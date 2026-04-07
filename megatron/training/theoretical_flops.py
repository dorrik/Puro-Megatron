"""Structured theoretical FLOPs reporting for training startup logs."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from megatron.core.extensions.transformer_engine import TEQuantizationParams
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    is_linear_attention_variant,
)
from megatron.core.transformer.enums import LayerType
from megatron.core.quantization.utils import (
    get_quant_config_or_none,
    kitchen_quantization_recipe_config,
    load_quantization_recipe,
)


FWD_BWD_EXPANSION_FACTOR = 3
FMA_EXPANSION_FACTOR = 2


@dataclass(frozen=True)
class TheoreticalFlopOp:
    """One FLOPs-bearing operator term inside a grouped layer/module."""

    submodule: str
    operator_name: str
    operator_kind: str
    precision: str
    shape: str
    count: str
    flops: float


@dataclass(frozen=True)
class TheoreticalFlopGroup:
    """A homogeneous layer or pseudo-layer group."""

    group_name: str
    layer_type: str
    count: int
    ops: Tuple[TheoreticalFlopOp, ...]

    @property
    def total_flops(self) -> float:
        return self.count * sum(op.flops for op in self.ops)


@dataclass(frozen=True)
class TheoreticalFlopReport:
    """Structured report for one theoretical FLOPs calculation."""

    batch_size: int
    micro_batch_size: int
    data_parallel_size: int
    num_microbatches: int
    seq_length: int
    total_flops: float
    layer_groups: Tuple[TheoreticalFlopGroup, ...]
    stage_reports: Tuple["TheoreticalFlopStage", ...]
    submodule_totals: Dict[str, float]
    operator_totals: Dict[str, float]
    submodule_operator_totals: Dict[str, Dict[str, float]]
    submodule_operator_precision_totals: Dict[str, Dict[str, Dict[str, float]]]
    operator_shape_precision_totals: Dict[str, Dict[str, Dict[str, float]]]
    precision_totals: Dict[str, float]
    precision_operator_totals: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class TheoreticalFlopStage:
    stage_ids: Tuple[Tuple[int, int], ...]
    layer_groups: Tuple[TheoreticalFlopGroup, ...]
    total_flops: float
    submodule_operator_precision_totals: Dict[str, Dict[str, Dict[str, float]]]
    operator_shape_precision_totals: Dict[str, Dict[str, Dict[str, float]]]
    precision_operator_totals: Dict[str, Dict[str, float]]


def _getattr(args, name: str, default=None):
    return getattr(args, name, default)


def _default_training_precision(args) -> str:
    if _getattr(args, "bf16", False):
        return "bf16"
    if _getattr(args, "fp16", False):
        return "fp16"
    return "fp32"


def _high_precision_fallback(args) -> str:
    if _getattr(args, "fp8", None) is not None or _getattr(args, "fp4", None) is not None:
        return "bf16"
    return _default_training_precision(args)


def _layer_precision_layout(args) -> Optional[Tuple[str, ...]]:
    layout = _getattr(args, "layer_precision_layout", None)
    if layout is None:
        return None
    from megatron.core.transformer.transformer_config import TransformerConfig

    return tuple(TransformerConfig.expand_layer_precision_layout(layout))


def _is_edge_bf16_layer(args, layer_idx: int) -> bool:
    if not _getattr(args, "first_last_layers_bf16", False):
        return False
    num_layers = _getattr(args, "num_layers")
    num_bf16_layers_at_start = _getattr(args, "num_layers_at_start_in_bf16", 0)
    num_bf16_layers_at_end = _getattr(args, "num_layers_at_end_in_bf16", 0)
    is_first_layer = layer_idx < num_bf16_layers_at_start
    is_last_layer = layer_idx >= num_layers - num_bf16_layers_at_end
    return is_first_layer or is_last_layer


def _resolve_layer_precision(args, layer_idx: int) -> str:
    num_layers = _getattr(args, "num_layers")
    if num_layers is not None and num_layers > 0 and layer_idx >= num_layers:
        layer_idx = num_layers - 1
    layout = _layer_precision_layout(args)
    if layout is not None:
        return layout[layer_idx]
    if _getattr(args, "fp8", None) is not None:
        return "bf16" if _is_edge_bf16_layer(args, layer_idx) else "fp8"
    if _getattr(args, "fp4", None) is not None:
        return "bf16" if _is_edge_bf16_layer(args, layer_idx) else "fp4"
    return _default_training_precision(args)


def _load_quant_recipe(args):
    kitchen_config_file = _getattr(args, "kitchen_config_file", None)
    if kitchen_config_file is not None:
        return load_quantization_recipe(kitchen_config_file)
    kitchen_recipe_number = _getattr(args, "kitchen_recipe_number", None)
    if kitchen_recipe_number is not None:
        return kitchen_quantization_recipe_config(kitchen_recipe_number)
    te_precision_config_file = _getattr(args, "te_precision_config_file", None)
    if te_precision_config_file:
        return load_quantization_recipe(te_precision_config_file)
    return None


def _parallel_size(args, name: str) -> int:
    value = _getattr(args, name, 1)
    if value in (None, 0):
        return 1
    return int(value)


def _resolve_quantized_module_precision(module_path: str, quant_recipe) -> Optional[str]:
    if quant_recipe is None:
        return None
    quant_config = get_quant_config_or_none(module_path, quant_recipe)
    if quant_config is None:
        return None
    qparams = TEQuantizationParams.parse_from_config(quant_config)
    training_recipe = qparams.training_recipe
    if training_recipe.fp4_quantization_recipe is not None:
        return "fp4"
    if training_recipe.fp8_quantization_recipe is not None:
        return "fp8"
    return "bf16"


def _resolve_linear_precision(args, module_path: str, base_precision: str, quant_recipe) -> str:
    override_precision = _resolve_quantized_module_precision(module_path, quant_recipe)
    if override_precision is not None:
        return override_precision
    return base_precision


def _resolve_core_attention_precision(
    args, module_path: str, base_precision: str, quant_recipe
) -> str:
    override_precision = _resolve_quantized_module_precision(module_path, quant_recipe)
    if override_precision is not None:
        return override_precision
    if base_precision == "fp8" and (
        _getattr(args, "fp8_dot_product_attention", False)
        or _getattr(args, "fp8_multi_head_attention", False)
    ):
        return "fp8"
    return _high_precision_fallback(args)


def _resolve_non_gemm_precision(args, module_path: str, quant_recipe) -> str:
    override_precision = _resolve_quantized_module_precision(module_path, quant_recipe)
    if override_precision is not None:
        return override_precision
    return _high_precision_fallback(args)


def _make_op(
    *,
    submodule: str,
    operator_name: str,
    operator_kind: str,
    precision: str,
    shape: str,
    count: str,
    flops: float,
) -> TheoreticalFlopOp:
    return TheoreticalFlopOp(
        submodule=submodule,
        operator_name=operator_name,
        operator_kind=operator_kind,
        precision=precision,
        shape=shape,
        count=count,
        flops=float(flops),
    )

def _partitioned_dim(dim: int, partitions: int):
    if partitions <= 1:
        return dim
    if dim % partitions == 0:
        return dim // partitions
    return f"{dim}/{partitions}"


def _micro_batch_size(args) -> int:
    return int(_getattr(args, "micro_batch_size", 1) or 1)


def _divide_symbol(base: str, divisors: Sequence[Tuple[Union[int, str], str]]) -> str:
    active_labels = []
    for value, label in divisors:
        if isinstance(value, str) or (value and value > 1):
            active_labels.append(label)
    if not active_labels:
        return base
    if len(active_labels) == 1:
        return f"{base}/{active_labels[0]}"
    return f"{base}/({'*'.join(active_labels)})"


def _partition_symbol(base: str, partitions: int, label: str) -> str:
    return _divide_symbol(base, [(partitions, label)])


def _local_sequence_dim(args):
    seq_length = _getattr(args, "seq_length")
    context_parallel_size = max(1, _getattr(args, "context_parallel_size", 1))
    tensor_model_parallel_size = max(1, _getattr(args, "tensor_model_parallel_size", 1))
    divisor = context_parallel_size * (
        tensor_model_parallel_size if _getattr(args, "sequence_parallel", False) else 1
    )
    return _partitioned_dim(seq_length, divisor)


def _local_sequence_symbol(args):
    divisors = [(max(1, _getattr(args, "context_parallel_size", 1)), "cp")]
    if _getattr(args, "sequence_parallel", False):
        # Sequence parallel shards the token dimension across the TP group, but this
        # is distinct from row/column tensor-parallel weight partitioning.
        divisors.append((max(1, _getattr(args, "tensor_model_parallel_size", 1)), "sp"))
    return _divide_symbol("seq", divisors)


def _local_token_dim(args):
    local_sequence_dim = _local_sequence_dim(args)
    if isinstance(local_sequence_dim, int):
        return _micro_batch_size(args) * local_sequence_dim
    return f"{_micro_batch_size(args)}*{local_sequence_dim}"


def _local_token_symbol(args) -> str:
    return f"mbs*{_local_sequence_symbol(args)}"


def _tp_linear_compute_sequence_dim(args):
    # Sequence parallel shards activations across the TP group for storage/communication,
    # but TP linear GEMMs gather the sequence dimension before matmul and reduce-scatter
    # after matmul. The GEMM m dimension therefore only reflects CP sharding.
    seq_length = _getattr(args, "seq_length")
    context_parallel_size = max(1, _getattr(args, "context_parallel_size", 1))
    return _partitioned_dim(seq_length, context_parallel_size)


def _tp_linear_compute_sequence_symbol(args):
    return _divide_symbol("seq", [(max(1, _getattr(args, "context_parallel_size", 1)), "cp")])


def _tp_linear_compute_token_dim(args):
    seq_dim = _tp_linear_compute_sequence_dim(args)
    if isinstance(seq_dim, int):
        return _micro_batch_size(args) * seq_dim
    return f"{_micro_batch_size(args)}*{seq_dim}"


def _tp_linear_compute_token_symbol(args) -> str:
    return f"mbs*{_tp_linear_compute_sequence_symbol(args)}"


def _format_gemm_shape(
    *,
    m_symbol: str,
    m_value,
    n_symbol: str,
    n_value,
    k_symbol: str,
    k_value,
) -> str:
    return f"(m,n,k)=({m_symbol}, {n_symbol}, {k_symbol})=({m_value}, {n_value}, {k_value})"


def _column_parallel_gemm_shape(
    args,
    input_dim: int,
    output_dim: int,
    *,
    input_label: Optional[str] = None,
    output_label: Optional[str] = None,
    partitions: Optional[int] = None,
    partition_label: str = "tp",
):
    partitions = max(1, partitions if partitions is not None else _getattr(args, "tensor_model_parallel_size", 1))
    return _format_gemm_shape(
        m_symbol=_tp_linear_compute_token_symbol(args),
        m_value=_tp_linear_compute_token_dim(args),
        n_symbol=_partition_symbol(output_label or str(output_dim), partitions, partition_label),
        n_value=_partitioned_dim(output_dim, partitions),
        k_symbol=input_label or str(input_dim),
        k_value=input_dim,
    )


def _row_parallel_gemm_shape(
    args,
    input_dim: int,
    output_dim: int,
    *,
    input_label: Optional[str] = None,
    output_label: Optional[str] = None,
    partitions: Optional[int] = None,
    partition_label: str = "tp",
):
    partitions = max(1, partitions if partitions is not None else _getattr(args, "tensor_model_parallel_size", 1))
    return _format_gemm_shape(
        m_symbol=_tp_linear_compute_token_symbol(args),
        m_value=_tp_linear_compute_token_dim(args),
        n_symbol=output_label or str(output_dim),
        n_value=output_dim,
        k_symbol=_partition_symbol(input_label or str(input_dim), partitions, partition_label),
        k_value=_partitioned_dim(input_dim, partitions),
    )


def _replicated_gemm_shape(
    args,
    input_dim: int,
    output_dim: int,
    *,
    input_label: Optional[str] = None,
    output_label: Optional[str] = None,
):
    return _format_gemm_shape(
        m_symbol=_local_token_symbol(args),
        m_value=_local_token_dim(args),
        n_symbol=output_label or str(output_dim),
        n_value=output_dim,
        k_symbol=input_label or str(input_dim),
        k_value=input_dim,
    )


def _grouped_gemm_shape(
    args,
    *,
    output_dim: int,
    input_dim: int,
    output_label: Optional[str] = None,
    input_label: Optional[str] = None,
    output_partitions: int = 1,
    input_partitions: int = 1,
    output_partition_label: str = "etp",
    input_partition_label: str = "etp",
) -> str:
    num_experts = _getattr(args, "num_experts", None)
    topk = _getattr(args, "moe_router_topk")
    local_tokens = _local_token_dim(args)
    local_experts = (
        _partitioned_dim(num_experts, _parallel_size(args, "expert_model_parallel_size"))
        if num_experts is not None
        else "local_experts"
    )
    ep_size = _parallel_size(args, "expert_model_parallel_size")

    if num_experts is not None and isinstance(local_tokens, int):
        avg_m = local_tokens * topk / num_experts
        avg_m = int(avg_m) if float(avg_m).is_integer() else round(avg_m, 3)
    elif num_experts is not None:
        avg_m = f"{_local_token_symbol(args)}*topk/{num_experts}"
    else:
        avg_m = _local_token_symbol(args)
    avg_m_symbol = (
        f"{_local_token_symbol(args)}*topk/{num_experts}"
        if num_experts is not None
        else _local_token_symbol(args)
    )
    local_experts_symbol = "num_experts/ep" if num_experts is not None and ep_size > 1 else "num_experts"

    n_symbol = _partition_symbol(
        output_label or str(output_dim), output_partitions, output_partition_label
    )
    k_symbol = _partition_symbol(
        input_label or str(input_dim), input_partitions, input_partition_label
    )
    n_value = _partitioned_dim(output_dim, output_partitions)
    k_value = _partitioned_dim(input_dim, input_partitions)

    return (
        f"groups(~m,n,k)={local_experts_symbol}(~{avg_m_symbol}, {n_symbol}, {k_symbol})="
        f"{local_experts}(~{avg_m}, {n_value}, {k_value})"
    )


def _local_attention_shape(args, num_heads: int, head_dim: int, num_query_groups: Optional[int] = None):
    tp_size = max(1, _getattr(args, "tensor_model_parallel_size", 1))
    local_heads = _partitioned_dim(num_heads, tp_size)
    if num_query_groups is None:
        return (
            f"Q/K/V=(mbs={_micro_batch_size(args)}, "
            f"seq={_local_sequence_dim(args)}, h={local_heads}, d={head_dim})"
        )
    local_query_groups = _partitioned_dim(num_query_groups, tp_size)
    return (
        f"Q=(mbs={_micro_batch_size(args)}, seq={_local_sequence_dim(args)}, "
        f"h={local_heads}, d={head_dim}), "
        f"K/V=(mbs={_micro_batch_size(args)}, seq={_local_sequence_dim(args)}, "
        f"g={local_query_groups}, d={head_dim})"
    )


def _attention_layout(args) -> str:
    return _getattr(args, "attention_type", "sbhd")


def _core_attention_name(attn_kind: str, args) -> str:
    return f"({attn_kind})core_attn({_attention_layout(args)})"


def _standard_core_attention_name(args) -> str:
    if _getattr(args, "group_query_attention", False) and _getattr(
        args, "num_query_groups", None
    ) != _getattr(args, "num_attention_heads"):
        return _core_attention_name("gqa", args)
    return _core_attention_name("mha", args)


def _mla_core_attention_shape(args, num_heads: int, qk_head_dim: int, v_head_dim: int) -> str:
    tp_size = max(1, _getattr(args, "tensor_model_parallel_size", 1))
    local_heads = _partitioned_dim(num_heads, tp_size)
    return (
        f"(mbs={_micro_batch_size(args)}, seq={_local_sequence_dim(args)}, "
        f"h={local_heads}, qk_d={qk_head_dim}, v_d={v_head_dim})"
    )


def _share_operator_label(op: TheoreticalFlopOp) -> str:
    if op.operator_kind == "core_attention":
        return op.operator_name
    return op.operator_kind


def _moe_layer_pattern(args) -> List[int]:
    num_experts = _getattr(args, "num_experts", None)
    num_layers = _getattr(args, "num_layers")
    if num_experts is None:
        return [0] * num_layers
    moe_layer_freq = _getattr(args, "moe_layer_freq")
    if isinstance(moe_layer_freq, int):
        return [1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)]
    if isinstance(moe_layer_freq, list):
        assert len(moe_layer_freq) == num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_freq)}, "
            f"expected {num_layers}, current moe layer pattern: {moe_layer_freq}"
        )
        return list(moe_layer_freq)
    raise RuntimeError("Illegal --moe-layer-freq argument provided!")


def _linear_attention_pattern(args, total_num_layers: int) -> Optional[List[int]]:
    if not is_linear_attention_variant(_getattr(args, "experimental_attention_variant", None)):
        return None
    linear_attention_freq = _getattr(args, "linear_attention_freq", None)
    if isinstance(linear_attention_freq, int):
        return [0 if ((i + 1) % linear_attention_freq == 0) else 1 for i in range(total_num_layers)]
    if isinstance(linear_attention_freq, list):
        assert len(linear_attention_freq) == total_num_layers, (
            f"Invalid length of linear_attention_pattern: {len(linear_attention_freq)}, "
            f"expected {total_num_layers}, current linear attention pattern: "
            f"{linear_attention_freq}"
        )
        return list(linear_attention_freq)
    if linear_attention_freq is None:
        raise ValueError(
            f"Linear attention type {_getattr(args, 'experimental_attention_variant')} is "
            "specified but linear_attention_freq is None."
        )
    raise ValueError(
        f"Invalid linear_attention_freq: {type(linear_attention_freq)}, {linear_attention_freq}"
    )


def _standard_attention_ops(args, batch_size: int, layer_idx: int, prefix: str, quant_recipe):
    if not _getattr(args, "group_query_attention", False):
        num_query_groups = _getattr(args, "num_attention_heads")
    else:
        num_query_groups = _getattr(args, "num_query_groups")

    hidden_size = _getattr(args, "hidden_size")
    seq_length = _getattr(args, "seq_length")
    kv_channels = _getattr(args, "kv_channels")
    num_attention_heads = _getattr(args, "num_attention_heads")
    query_projection_size = kv_channels * num_attention_heads
    key_projection_size = kv_channels * num_query_groups
    value_projection_size = kv_channels * num_query_groups
    gate_projection_size = query_projection_size if _getattr(args, "attention_output_gate", False) else 0
    qkv_total = query_projection_size + key_projection_size + value_projection_size + gate_projection_size

    base_precision = _resolve_layer_precision(args, layer_idx)
    linear_qkv_precision = _resolve_linear_precision(
        args, f"{prefix}self_attention.linear_qkv", base_precision, quant_recipe
    )
    core_attention_precision = _resolve_core_attention_precision(
        args, f"{prefix}self_attention.core_attention", base_precision, quant_recipe
    )
    linear_proj_precision = _resolve_linear_precision(
        args, f"{prefix}self_attention.linear_proj", base_precision, quant_recipe
    )

    batch_seq = batch_size * seq_length
    qkv_flops = (
        batch_seq
        * FWD_BWD_EXPANSION_FACTOR
        * FMA_EXPANSION_FACTOR
        * hidden_size
        * qkv_total
    )
    core_attention_flops = (
        batch_seq
        * FWD_BWD_EXPANSION_FACTOR
        * FMA_EXPANSION_FACTOR
        * query_projection_size
        * seq_length
    )
    proj_flops = (
        batch_seq
        * FWD_BWD_EXPANSION_FACTOR
        * FMA_EXPANSION_FACTOR
        * query_projection_size
        * hidden_size
    )

    return [
        _make_op(
            submodule="attention",
            operator_name="qkv_projection",
            operator_kind="gemm",
            precision=linear_qkv_precision,
            shape=_column_parallel_gemm_shape(
                args, hidden_size, qkv_total, input_label="hidden", output_label="qkv"
            ),
            count="3",
            flops=qkv_flops,
        ),
        _make_op(
            submodule="attention",
            operator_name=_standard_core_attention_name(args),
            operator_kind="core_attention",
            precision=core_attention_precision,
            shape=_local_attention_shape(args, num_attention_heads, kv_channels, num_query_groups),
            count="2 x 3",
            flops=core_attention_flops,
        ),
        _make_op(
            submodule="attention",
            operator_name="output_projection",
            operator_kind="gemm",
            precision=linear_proj_precision,
            shape=_row_parallel_gemm_shape(
                args,
                query_projection_size,
                hidden_size,
                input_label="attn_proj",
                output_label="hidden",
            ),
            count="3",
            flops=proj_flops,
        ),
    ]


def _mla_attention_ops(args, batch_size: int, layer_idx: int, prefix: str, quant_recipe):
    hidden_size = _getattr(args, "hidden_size")
    seq_length = _getattr(args, "seq_length")
    num_attention_heads = _getattr(args, "num_attention_heads")
    qk_head_dim = _getattr(args, "qk_head_dim")
    qk_pos_emb_head_dim = _getattr(args, "qk_pos_emb_head_dim")
    v_head_dim = _getattr(args, "v_head_dim")
    kv_lora_rank = _getattr(args, "kv_lora_rank")
    q_lora_rank = _getattr(args, "q_lora_rank", None)
    base_precision = _resolve_layer_precision(args, layer_idx)
    batch_seq = batch_size * seq_length

    ops = []

    if q_lora_rank is None:
        q_proj_dim = num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim)
        ops.append(
            _make_op(
                submodule="attention",
                operator_name="q_projection",
                operator_kind="gemm",
                precision=_resolve_linear_precision(
                    args, f"{prefix}self_attention.linear_q_proj", base_precision, quant_recipe
                ),
                shape=_column_parallel_gemm_shape(
                    args, hidden_size, q_proj_dim, input_label="hidden", output_label="q_proj"
                ),
                count="3",
                flops=(
                    batch_seq
                    * FWD_BWD_EXPANSION_FACTOR
                    * FMA_EXPANSION_FACTOR
                    * hidden_size
                    * q_proj_dim
                ),
            )
        )
    else:
        ops.extend(
            [
                _make_op(
                    submodule="attention",
                    operator_name="q_down_projection",
                    operator_kind="gemm",
                    precision=_resolve_linear_precision(
                        args,
                        f"{prefix}self_attention.linear_q_down_proj",
                        base_precision,
                        quant_recipe,
                    ),
                    shape=_replicated_gemm_shape(
                        args,
                        hidden_size,
                        q_lora_rank,
                        input_label="hidden",
                        output_label="q_lora_rank",
                    ),
                    count="3",
                    flops=(
                        batch_seq
                        * FWD_BWD_EXPANSION_FACTOR
                        * FMA_EXPANSION_FACTOR
                        * hidden_size
                        * q_lora_rank
                    ),
                ),
                _make_op(
                    submodule="attention",
                    operator_name="q_up_projection",
                    operator_kind="gemm",
                    precision=_resolve_linear_precision(
                        args,
                        f"{prefix}self_attention.linear_q_up_proj",
                        base_precision,
                        quant_recipe,
                    ),
                    shape=_column_parallel_gemm_shape(
                        args,
                        q_lora_rank,
                        num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim),
                        input_label="q_lora_rank",
                        output_label="q_proj",
                    ),
                    count="3",
                    flops=(
                        batch_seq
                        * FWD_BWD_EXPANSION_FACTOR
                        * FMA_EXPANSION_FACTOR
                        * q_lora_rank
                        * num_attention_heads
                        * (qk_head_dim + qk_pos_emb_head_dim)
                    ),
                ),
                _make_op(
                    submodule="attention",
                    operator_name="q_norm_approx",
                    operator_kind="elementwise",
                    precision=_resolve_non_gemm_precision(
                        args, f"{prefix}self_attention.q_layernorm", quant_recipe
                    ),
                    shape=f"({_local_token_dim(args)}, {q_lora_rank})",
                    count="3 x 2",
                    flops=batch_seq * FWD_BWD_EXPANSION_FACTOR * FMA_EXPANSION_FACTOR * q_lora_rank,
                ),
            ]
        )

    ops.extend(
        [
            _make_op(
                submodule="attention",
                operator_name="kv_down_projection",
                operator_kind="gemm",
                precision=_resolve_linear_precision(
                    args,
                    f"{prefix}self_attention.linear_kv_down_proj",
                    base_precision,
                    quant_recipe,
                ),
                shape=_replicated_gemm_shape(
                    args,
                    hidden_size,
                    kv_lora_rank,
                    input_label="hidden",
                    output_label="kv_lora_rank",
                ),
                count="3",
                flops=(
                    batch_seq
                    * FWD_BWD_EXPANSION_FACTOR
                    * FMA_EXPANSION_FACTOR
                    * hidden_size
                    * kv_lora_rank
                ),
            ),
            _make_op(
                submodule="attention",
                operator_name="kv_up_projection",
                operator_kind="gemm",
                precision=_resolve_linear_precision(
                    args,
                    f"{prefix}self_attention.linear_kv_up_proj",
                    base_precision,
                    quant_recipe,
                ),
                shape=_column_parallel_gemm_shape(
                    args,
                    kv_lora_rank,
                    num_attention_heads * (qk_head_dim + v_head_dim),
                    input_label="kv_lora_rank",
                    output_label="kv_up",
                ),
                count="3",
                flops=(
                    batch_seq
                    * FWD_BWD_EXPANSION_FACTOR
                    * FMA_EXPANSION_FACTOR
                    * kv_lora_rank
                    * num_attention_heads
                    * (qk_head_dim + v_head_dim)
                ),
            ),
            _make_op(
                submodule="attention",
                operator_name="kv_norm_approx",
                operator_kind="elementwise",
                precision=_resolve_non_gemm_precision(
                    args, f"{prefix}self_attention.kv_layernorm", quant_recipe
                ),
                shape=f"({_local_token_dim(args)}, {kv_lora_rank})",
                count="3 x 2",
                flops=batch_seq * FWD_BWD_EXPANSION_FACTOR * FMA_EXPANSION_FACTOR * kv_lora_rank,
            ),
            _make_op(
                submodule="attention",
                operator_name="kv_rope_projection",
                operator_kind="gemm",
                precision=_resolve_non_gemm_precision(
                    args, f"{prefix}self_attention.rotary_pos_emb", quant_recipe
                ),
                shape=_replicated_gemm_shape(
                    args,
                    hidden_size,
                    qk_pos_emb_head_dim,
                    input_label="hidden",
                    output_label="rope_head",
                ),
                count="3",
                flops=(
                    batch_seq
                    * FWD_BWD_EXPANSION_FACTOR
                    * FMA_EXPANSION_FACTOR
                    * hidden_size
                    * qk_pos_emb_head_dim
                ),
            ),
            _make_op(
                submodule="attention",
                operator_name=_core_attention_name("mla", args),
                operator_kind="core_attention",
                precision=_resolve_core_attention_precision(
                    args, f"{prefix}self_attention.core_attention", base_precision, quant_recipe
                ),
                shape=_mla_core_attention_shape(
                    args,
                    num_attention_heads,
                    qk_head_dim + qk_pos_emb_head_dim,
                    v_head_dim,
                ),
                count="2 x 3",
                flops=(
                    batch_seq
                    * FWD_BWD_EXPANSION_FACTOR
                    * FMA_EXPANSION_FACTOR
                    * seq_length
                    * num_attention_heads
                    * ((qk_head_dim + qk_pos_emb_head_dim) + v_head_dim)
                    / 2
                ),
            ),
            _make_op(
                submodule="attention",
                operator_name="output_projection",
                operator_kind="gemm",
                precision=_resolve_linear_precision(
                    args, f"{prefix}self_attention.linear_proj", base_precision, quant_recipe
                ),
                shape=_row_parallel_gemm_shape(
                    args,
                    num_attention_heads * v_head_dim,
                    hidden_size,
                    input_label="attn_v",
                    output_label="hidden",
                ),
                count="3",
                flops=(
                    batch_seq
                    * FWD_BWD_EXPANSION_FACTOR
                    * FMA_EXPANSION_FACTOR
                    * num_attention_heads
                    * v_head_dim
                    * hidden_size
                ),
            ),
        ]
    )

    return ops


def _gated_delta_net_attention_ops(
    args, batch_size: int, layer_idx: int, prefix: str, quant_recipe
):
    hidden_size = _getattr(args, "hidden_size")
    seq_length = _getattr(args, "seq_length")
    linear_conv_kernel_dim = _getattr(args, "linear_conv_kernel_dim")
    qk_head_dim = _getattr(args, "linear_key_head_dim")
    v_head_dim = _getattr(args, "linear_value_head_dim")
    num_qk_heads = _getattr(args, "linear_num_key_heads")
    num_v_heads = _getattr(args, "linear_num_value_heads")
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    base_precision = _resolve_layer_precision(args, layer_idx)
    batch_seq = batch_size * seq_length

    return [
        _make_op(
            submodule="attention",
            operator_name="gdn_in_projection",
            operator_kind="gemm",
            precision=_resolve_linear_precision(
                args, f"{prefix}self_attention.in_proj", base_precision, quant_recipe
            ),
            shape=_column_parallel_gemm_shape(
                args,
                hidden_size,
                2 * qk_dim + 2 * v_dim + 2 * num_v_heads,
                input_label="hidden",
                output_label="gdn_in",
            ),
            count="3",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * hidden_size
                * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
            ),
        ),
        _make_op(
            submodule="attention",
            operator_name="gdn_conv1d",
            operator_kind="conv1d",
            precision=_resolve_non_gemm_precision(args, f"{prefix}self_attention.conv1d", quant_recipe),
            shape=(
                f"(micro_batch_size={_micro_batch_size(args)}, seq_per_rank={_local_sequence_dim(args)}, "
                f"channels_per_rank={_partitioned_dim(2 * qk_dim + v_dim, _parallel_size(args, 'tensor_model_parallel_size'))}, "
                f"kernel={linear_conv_kernel_dim})"
            ),
            count="3",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * linear_conv_kernel_dim
                * (2 * qk_dim + v_dim)
            ),
        ),
        _make_op(
            submodule="attention",
            operator_name="gated_delta_rule",
            operator_kind="linear_attention",
            precision=_resolve_non_gemm_precision(
                args, f"{prefix}self_attention.gated_delta_rule", quant_recipe
            ),
            shape=(
                f"(micro_batch_size={_micro_batch_size(args)}, seq_per_rank={_local_sequence_dim(args)}, "
                f"value_heads_per_rank={_partitioned_dim(num_v_heads, _parallel_size(args, 'tensor_model_parallel_size'))}, "
                f"value_head_dim={v_head_dim})"
            ),
            count="4 x 3",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * num_v_heads
                * (v_head_dim ** 2)
                * 4
            ),
        ),
        _make_op(
            submodule="attention",
            operator_name="gdn_out_projection",
            operator_kind="gemm",
            precision=_resolve_linear_precision(
                args, f"{prefix}self_attention.out_proj", base_precision, quant_recipe
            ),
            shape=_row_parallel_gemm_shape(
                args, v_dim, hidden_size, input_label="gdn_v", output_label="hidden"
            ),
            count="3",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * hidden_size
                * v_dim
            ),
        ),
    ]


def _dense_mlp_ops(
    args,
    batch_size: int,
    layer_idx: int,
    prefix: str,
    quant_recipe,
    *,
    submodule: str = "mlp",
    fc1_module: str = "mlp.linear_fc1",
    fc2_module: str = "mlp.linear_fc2",
    hidden_size: Optional[int] = None,
    ffn_hidden_size: Optional[int] = None,
) -> List[TheoreticalFlopOp]:
    hidden_size = _getattr(args, "hidden_size") if hidden_size is None else hidden_size
    ffn_hidden_size = _getattr(args, "ffn_hidden_size") if ffn_hidden_size is None else ffn_hidden_size
    seq_length = _getattr(args, "seq_length")
    batch_seq = batch_size * seq_length
    fc1_multiplier = 2 if _getattr(args, "swiglu", False) else 1
    base_precision = _resolve_layer_precision(args, layer_idx)

    return [
        _make_op(
            submodule=submodule,
            operator_name="fc1",
            operator_kind="gemm",
            precision=_resolve_linear_precision(
                args, f"{prefix}{fc1_module}", base_precision, quant_recipe
            ),
            shape=_column_parallel_gemm_shape(
                args,
                hidden_size,
                fc1_multiplier * ffn_hidden_size,
                input_label="hidden",
                output_label="ffn_hidden" if fc1_multiplier == 1 else "2*ffn_hidden",
            ),
            count="3",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * hidden_size
                * fc1_multiplier
                * ffn_hidden_size
            ),
        ),
        _make_op(
            submodule=submodule,
            operator_name="fc2",
            operator_kind="gemm",
            precision=_resolve_linear_precision(
                args, f"{prefix}{fc2_module}", base_precision, quant_recipe
            ),
            shape=_row_parallel_gemm_shape(
                args, ffn_hidden_size, hidden_size, input_label="ffn_hidden", output_label="hidden"
            ),
            count="3",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * hidden_size
                * ffn_hidden_size
            ),
        ),
    ]


def _moe_ops(args, batch_size: int, layer_idx: int, prefix: str, quant_recipe):
    moe_ffn_hidden_size = _getattr(args, "moe_ffn_hidden_size", None)
    if moe_ffn_hidden_size is None:
        moe_ffn_hidden_size = _getattr(args, "ffn_hidden_size")
    shared_expert_ffn_hidden_size = _getattr(args, "moe_shared_expert_intermediate_size", None)
    if shared_expert_ffn_hidden_size is None:
        shared_expert_ffn_hidden_size = 0
    num_experts_routed_to = _getattr(args, "moe_router_topk")
    seq_length = _getattr(args, "seq_length")
    hidden_size = _getattr(args, "hidden_size")
    batch_seq = batch_size * seq_length
    base_precision = _resolve_layer_precision(args, layer_idx)
    fc1_multiplier = 2 if _getattr(args, "swiglu", False) else 1
    expert_tp_size = max(
        1,
        _getattr(args, "expert_tensor_parallel_size", None)
        or _getattr(args, "tensor_model_parallel_size", 1),
    )
    routed_operator_kind = (
        "grouped_gemm"
        if _getattr(args, "moe_grouped_gemm", False) or _getattr(args, "moe_use_legacy_grouped_gemm", False)
        else "gemm"
    )

    ops = [
        _make_op(
            submodule="moe_routed_experts",
            operator_name="routed_fc1",
            operator_kind=routed_operator_kind,
            precision=_resolve_linear_precision(
                args, f"{prefix}mlp.experts.linear_fc1", base_precision, quant_recipe
            ),
            shape=(
                _grouped_gemm_shape(
                    args,
                    output_dim=fc1_multiplier * moe_ffn_hidden_size,
                    input_dim=hidden_size,
                    output_label=(
                        "moe_ffn_hidden" if fc1_multiplier == 1 else "2*moe_ffn_hidden"
                    ),
                    input_label="hidden",
                    output_partitions=expert_tp_size,
                    output_partition_label=(
                        "etp" if _getattr(args, "expert_tensor_parallel_size", None) else "tp"
                    ),
                )
                if routed_operator_kind == "grouped_gemm"
                else _column_parallel_gemm_shape(
                    args,
                    hidden_size,
                    fc1_multiplier * moe_ffn_hidden_size,
                    input_label="hidden",
                    output_label="moe_ffn_hidden" if fc1_multiplier == 1 else "2*moe_ffn_hidden",
                    partitions=expert_tp_size,
                    partition_label=(
                        "etp"
                        if _getattr(args, "expert_tensor_parallel_size", None)
                        else "tp"
                    ),
                )
            ),
            count="3" if routed_operator_kind == "grouped_gemm" else f"3 x topk={num_experts_routed_to}",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * hidden_size
                * fc1_multiplier
                * moe_ffn_hidden_size
                * num_experts_routed_to
            ),
        ),
        _make_op(
            submodule="moe_routed_experts",
            operator_name="routed_fc2",
            operator_kind=routed_operator_kind,
            precision=_resolve_linear_precision(
                args, f"{prefix}mlp.experts.linear_fc2", base_precision, quant_recipe
            ),
            shape=(
                _grouped_gemm_shape(
                    args,
                    output_dim=hidden_size,
                    input_dim=moe_ffn_hidden_size,
                    output_label="hidden",
                    input_label="moe_ffn_hidden",
                    input_partitions=expert_tp_size,
                    input_partition_label=(
                        "etp" if _getattr(args, "expert_tensor_parallel_size", None) else "tp"
                    ),
                )
                if routed_operator_kind == "grouped_gemm"
                else _row_parallel_gemm_shape(
                    args,
                    moe_ffn_hidden_size,
                    hidden_size,
                    input_label="moe_ffn_hidden",
                    output_label="hidden",
                    partitions=expert_tp_size,
                    partition_label=(
                        "etp"
                        if _getattr(args, "expert_tensor_parallel_size", None)
                        else "tp"
                    ),
                )
            ),
            count="3" if routed_operator_kind == "grouped_gemm" else f"3 x topk={num_experts_routed_to}",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * hidden_size
                * moe_ffn_hidden_size
                * num_experts_routed_to
            ),
        ),
    ]

    if shared_expert_ffn_hidden_size > 0:
        ops.extend(
            [
                _make_op(
                    submodule="moe_shared_experts",
                    operator_name="shared_fc1",
                    operator_kind="gemm",
                    precision=_resolve_linear_precision(
                        args, f"{prefix}mlp.shared_experts.linear_fc1", base_precision, quant_recipe
                    ),
                    shape=_column_parallel_gemm_shape(
                        args,
                        hidden_size,
                        fc1_multiplier * shared_expert_ffn_hidden_size,
                        input_label="hidden",
                        output_label=(
                            "shared_ffn_hidden"
                            if fc1_multiplier == 1
                            else "2*shared_ffn_hidden"
                        ),
                    ),
                    count="3",
                    flops=(
                        batch_seq
                        * FWD_BWD_EXPANSION_FACTOR
                        * FMA_EXPANSION_FACTOR
                        * hidden_size
                        * fc1_multiplier
                        * shared_expert_ffn_hidden_size
                    ),
                ),
                _make_op(
                    submodule="moe_shared_experts",
                    operator_name="shared_fc2",
                    operator_kind="gemm",
                    precision=_resolve_linear_precision(
                        args, f"{prefix}mlp.shared_experts.linear_fc2", base_precision, quant_recipe
                    ),
                    shape=_row_parallel_gemm_shape(
                        args,
                        shared_expert_ffn_hidden_size,
                        hidden_size,
                        input_label="shared_ffn_hidden",
                        output_label="hidden",
                    ),
                    count="3",
                    flops=(
                        batch_seq
                        * FWD_BWD_EXPANSION_FACTOR
                        * FMA_EXPANSION_FACTOR
                        * hidden_size
                        * shared_expert_ffn_hidden_size
                    ),
                ),
            ]
        )

    return ops


def _mtp_preamble_ops(args, batch_size: int, mtp_layer_idx: int, quant_recipe):
    hidden_size = _getattr(args, "hidden_size")
    seq_length = _getattr(args, "seq_length")
    batch_seq = batch_size * seq_length
    layer_idx_for_precision = max(_getattr(args, "num_layers") - 1, 0)
    base_precision = _resolve_layer_precision(args, layer_idx_for_precision)
    prefix = f"mtp_block.layers.{mtp_layer_idx}."

    return [
        _make_op(
            submodule="mtp_preamble",
            operator_name="eh_norm_and_final_norm_approx",
            operator_kind="elementwise",
            precision=_resolve_non_gemm_precision(args, f"{prefix}layer_norm", quant_recipe),
            shape=f"({_local_token_dim(args)}, {hidden_size})",
            count="3 x 2",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * 3
                * hidden_size
            ),
        ),
        _make_op(
            submodule="mtp_preamble",
            operator_name="eh_projection",
            operator_kind="gemm",
            precision=_resolve_linear_precision(args, f"{prefix}eh_proj", base_precision, quant_recipe),
            shape=_replicated_gemm_shape(
                args, hidden_size, hidden_size, input_label="hidden", output_label="hidden"
            ),
            count="3 x 2",
            flops=(
                batch_seq
                * FWD_BWD_EXPANSION_FACTOR
                * FMA_EXPANSION_FACTOR
                * 2
                * hidden_size
                * hidden_size
            ),
        ),
    ]


def _output_head_ops(args, batch_size: int, output_count: int, quant_recipe):
    hidden_size = _getattr(args, "hidden_size")
    seq_length = _getattr(args, "seq_length")
    vocab_size = _getattr(args, "padded_vocab_size")
    batch_seq = batch_size * seq_length
    precision = _resolve_linear_precision(
        args,
        "output_layer",
        _high_precision_fallback(args),
        quant_recipe,
    )
    return TheoreticalFlopGroup(
        group_name="output_head",
        layer_type="output_head",
        count=output_count,
        ops=(
            _make_op(
                submodule="output_head",
                operator_name="logits_projection",
                operator_kind="gemm",
                precision=precision,
                shape=_column_parallel_gemm_shape(
                    args, hidden_size, vocab_size, input_label="hidden", output_label="padded_vocab"
                ),
                count="3",
                flops=(
                    batch_seq
                    * FWD_BWD_EXPANSION_FACTOR
                    * FMA_EXPANSION_FACTOR
                    * hidden_size
                    * vocab_size
                ),
            ),
        ),
    )


def _group_signature(group: TheoreticalFlopGroup) -> Tuple:
    return (
        group.group_name,
        group.layer_type,
        tuple(
            (
                op.submodule,
                op.operator_name,
                op.operator_kind,
                op.precision,
                op.shape,
                op.count,
                round(op.flops, 6),
            )
            for op in group.ops
        ),
    )


def _merge_groups(groups: Sequence[TheoreticalFlopGroup]) -> Tuple[TheoreticalFlopGroup, ...]:
    merged: Dict[Tuple, Dict[str, object]] = {}
    for group in groups:
        key = _group_signature(group)
        if key not in merged:
            merged[key] = {
                "group_name": group.group_name,
                "layer_type": group.layer_type,
                "count": group.count,
                "ops": group.ops,
            }
        else:
            merged[key]["count"] += group.count
    return tuple(
        TheoreticalFlopGroup(
            group_name=value["group_name"],
            layer_type=value["layer_type"],
            count=value["count"],
            ops=value["ops"],
        )
        for value in sorted(
            merged.values(), key=lambda item: (-sum(op.flops for op in item["ops"]) * item["count"], item["group_name"])
        )
    )


def _stage_ids(args) -> List[Tuple[int, int]]:
    pp_size = max(1, int(_getattr(args, "pipeline_model_parallel_size", 1) or 1))
    vp_size = max(1, int(_getattr(args, "virtual_pipeline_model_parallel_size", 1) or 1))
    return [(pp_rank, vp_rank) for vp_rank in range(vp_size) for pp_rank in range(pp_size)]


def _build_stage_assignment(
    args, num_layers: int, mtp_num_layers: int
) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, Tuple[int, int]], Tuple[int, int], List[Tuple[int, int]]]:
    stage_list = _stage_ids(args)
    pp_size = max(1, int(_getattr(args, "pipeline_model_parallel_size", 1) or 1))
    vp_size = max(1, int(_getattr(args, "virtual_pipeline_model_parallel_size", 1) or 1))
    decoder_stage_map: Dict[int, Tuple[int, int]] = {}
    mtp_stage_map: Dict[int, Tuple[int, int]] = {}
    output_stage = stage_list[-1]

    pipeline_layout = _getattr(args, "pipeline_model_parallel_layout", None)
    if pipeline_layout is not None:
        from megatron.core.transformer.pipeline_parallel_layer_layout import (
            PipelineParallelLayerLayout,
        )

        layout = PipelineParallelLayerLayout(pipeline_layout, pp_size)
        stage_list = _stage_ids(args)
        decoder_offset = 0
        mtp_offset = 0
        for pp_rank, vp_rank in stage_list:
            stage_layers = layout.layout[pp_rank][vp_rank]
            decoder_count = stage_layers.count(LayerType.decoder)
            mtp_count = stage_layers.count(LayerType.mtp)

            for layer_id in range(decoder_offset, decoder_offset + decoder_count):
                decoder_stage_map[layer_id] = (pp_rank, vp_rank)
            decoder_offset += decoder_count

            for mtp_idx in range(mtp_offset, mtp_offset + mtp_count):
                mtp_stage_map[mtp_idx] = (pp_rank, vp_rank)
            mtp_offset += mtp_count

            if stage_layers.count(LayerType.loss) > 0:
                output_stage = (pp_rank, vp_rank)
        return decoder_stage_map, mtp_stage_map, output_stage, stage_list

    if num_layers > 0:
        num_layers_per_pipeline_rank = num_layers // pp_size
        num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
        total_virtual_chunks = num_layers // vp_size
        for vp_rank in range(vp_size):
            for pp_rank in range(pp_size):
                offset = vp_rank * total_virtual_chunks + pp_rank * num_layers_per_virtual_rank
                for layer_id in range(offset, offset + num_layers_per_virtual_rank):
                    decoder_stage_map[layer_id] = (pp_rank, vp_rank)

    for mtp_idx in range(mtp_num_layers):
        mtp_stage_map[mtp_idx] = output_stage

    return decoder_stage_map, mtp_stage_map, output_stage, stage_list


def _build_transformer_groups(args, batch_size: int, quant_recipe) -> Tuple[TheoreticalFlopGroup, ...]:
    num_layers = _getattr(args, "num_layers")
    moe_pattern = _moe_layer_pattern(args)
    num_mtp_layers = _getattr(args, "mtp_num_layers", None)
    mtp_num_layers = 0 if num_mtp_layers is None else num_mtp_layers
    total_num_layers = num_layers + mtp_num_layers
    linear_attention_pattern = _linear_attention_pattern(args, total_num_layers)
    last_layer_is_moe = moe_pattern[-1] if moe_pattern else 0

    groups: List[TheoreticalFlopGroup] = []

    def build_layer_group(
        *,
        global_layer_idx: int,
        prefix: str,
        dense_or_moe: str,
        layer_label: str,
    ):
        if _getattr(args, "multi_latent_attention", False):
            attention_variant = "mla"
            attn_ops = _mla_attention_ops(args, batch_size, global_layer_idx, prefix, quant_recipe)
        elif linear_attention_pattern is not None and linear_attention_pattern[global_layer_idx]:
            attention_variant = _getattr(args, "experimental_attention_variant")
            attn_ops = _gated_delta_net_attention_ops(
                args, batch_size, global_layer_idx, prefix, quant_recipe
            )
        else:
            attention_variant = "standard_attention"
            attn_ops = _standard_attention_ops(args, batch_size, global_layer_idx, prefix, quant_recipe)

        mlp_ops = (
            _dense_mlp_ops(args, batch_size, global_layer_idx, prefix, quant_recipe)
            if dense_or_moe == "dense"
            else _moe_ops(args, batch_size, global_layer_idx, prefix, quant_recipe)
        )
        groups.append(
            TheoreticalFlopGroup(
                group_name=f"{layer_label}:{dense_or_moe}:{attention_variant}",
                layer_type=layer_label,
                count=1,
                ops=tuple(attn_ops + mlp_ops),
            )
        )

    for layer_idx in range(num_layers):
        dense_or_moe = "moe" if moe_pattern[layer_idx] else "dense"
        build_layer_group(
            global_layer_idx=layer_idx,
            prefix=f"decoder.layers.{layer_idx}.",
            dense_or_moe=dense_or_moe,
            layer_label="transformer_layer",
        )

    for mtp_idx in range(mtp_num_layers):
        global_layer_idx = num_layers + mtp_idx
        dense_or_moe = "moe" if last_layer_is_moe else "dense"
        build_layer_group(
            global_layer_idx=global_layer_idx,
            prefix=f"mtp_block.layers.{mtp_idx}.transformer_layer.",
            dense_or_moe=dense_or_moe,
            layer_label="mtp_transformer_layer",
        )
        groups.append(
            TheoreticalFlopGroup(
                group_name="mtp_preamble",
                layer_type="mtp_preamble",
                count=1,
                ops=tuple(_mtp_preamble_ops(args, batch_size, mtp_idx, quant_recipe)),
            )
        )

    groups.append(_output_head_ops(args, batch_size, mtp_num_layers + 1, quant_recipe))
    return _merge_groups(groups)


def _build_transformer_stage_groups(
    args, batch_size: int, quant_recipe
) -> Dict[Tuple[int, int], Tuple[TheoreticalFlopGroup, ...]]:
    num_layers = _getattr(args, "num_layers")
    moe_pattern = _moe_layer_pattern(args)
    num_mtp_layers = _getattr(args, "mtp_num_layers", None)
    mtp_num_layers = 0 if num_mtp_layers is None else num_mtp_layers
    total_num_layers = num_layers + mtp_num_layers
    linear_attention_pattern = _linear_attention_pattern(args, total_num_layers)
    last_layer_is_moe = moe_pattern[-1] if moe_pattern else 0
    decoder_stage_map, mtp_stage_map, output_stage, stage_list = _build_stage_assignment(
        args, num_layers, mtp_num_layers
    )
    stage_groups: Dict[Tuple[int, int], List[TheoreticalFlopGroup]] = {
        stage_id: [] for stage_id in stage_list
    }

    def build_layer_group(
        *,
        global_layer_idx: int,
        prefix: str,
        dense_or_moe: str,
        layer_label: str,
    ) -> TheoreticalFlopGroup:
        if _getattr(args, "multi_latent_attention", False):
            attention_variant = "mla"
            attn_ops = _mla_attention_ops(args, batch_size, global_layer_idx, prefix, quant_recipe)
        elif linear_attention_pattern is not None and linear_attention_pattern[global_layer_idx]:
            attention_variant = _getattr(args, "experimental_attention_variant")
            attn_ops = _gated_delta_net_attention_ops(
                args, batch_size, global_layer_idx, prefix, quant_recipe
            )
        else:
            attention_variant = "standard_attention"
            attn_ops = _standard_attention_ops(args, batch_size, global_layer_idx, prefix, quant_recipe)

        mlp_ops = (
            _dense_mlp_ops(args, batch_size, global_layer_idx, prefix, quant_recipe)
            if dense_or_moe == "dense"
            else _moe_ops(args, batch_size, global_layer_idx, prefix, quant_recipe)
        )
        return TheoreticalFlopGroup(
            group_name=f"{layer_label}:{dense_or_moe}:{attention_variant}",
            layer_type=layer_label,
            count=1,
            ops=tuple(attn_ops + mlp_ops),
        )

    for layer_idx in range(num_layers):
        dense_or_moe = "moe" if moe_pattern[layer_idx] else "dense"
        stage_groups[decoder_stage_map[layer_idx]].append(
            build_layer_group(
                global_layer_idx=layer_idx,
                prefix=f"decoder.layers.{layer_idx}.",
                dense_or_moe=dense_or_moe,
                layer_label="transformer_layer",
            )
        )

    for mtp_idx in range(mtp_num_layers):
        global_layer_idx = num_layers + mtp_idx
        dense_or_moe = "moe" if last_layer_is_moe else "dense"
        stage_id = mtp_stage_map[mtp_idx]
        stage_groups[stage_id].append(
            build_layer_group(
                global_layer_idx=global_layer_idx,
                prefix=f"mtp_block.layers.{mtp_idx}.transformer_layer.",
                dense_or_moe=dense_or_moe,
                layer_label="mtp_transformer_layer",
            )
        )
        stage_groups[stage_id].append(
            TheoreticalFlopGroup(
                group_name="mtp_preamble",
                layer_type="mtp_preamble",
                count=1,
                ops=tuple(_mtp_preamble_ops(args, batch_size, mtp_idx, quant_recipe)),
            )
        )

    stage_groups[output_stage].append(_output_head_ops(args, batch_size, mtp_num_layers + 1, quant_recipe))
    return {stage_id: _merge_groups(groups) for stage_id, groups in stage_groups.items() if groups}


def _hybrid_layer_types(args) -> List[str]:
    pattern = _getattr(args, "hybrid_override_pattern", None)
    if pattern:
        mapping = {"*": "hybrid_attention_layer", "M": "hybrid_mamba_layer", "-": "hybrid_mlp_layer", "E": "hybrid_moe_layer"}
        return [mapping[layer_type] for layer_type in pattern if layer_type in mapping]

    num_layers = _getattr(args, "num_layers")
    num_attn_layers = round(num_layers * _getattr(args, "hybrid_attention_ratio"))
    num_mlp_layers = round(num_layers * _getattr(args, "hybrid_mlp_ratio"))
    num_mamba_layers = num_layers - num_attn_layers - num_mlp_layers
    return (
        ["hybrid_attention_layer"] * num_attn_layers
        + ["hybrid_mamba_layer"] * num_mamba_layers
        + ["hybrid_mlp_layer"] * num_mlp_layers
    )


def _hybrid_attention_ops(args, batch_size: int, layer_idx: int, prefix: str, quant_recipe):
    hidden_size = _getattr(args, "hidden_size")
    seq_length = _getattr(args, "seq_length")
    num_attention_heads = _getattr(args, "num_attention_heads")
    kv_channels = _getattr(args, "kv_channels")
    p = (kv_channels * num_attention_heads / hidden_size) if kv_channels else 1
    gqa_groups = _getattr(args, "num_query_groups") if _getattr(args, "group_query_attention", False) else num_attention_heads
    g = gqa_groups
    base_precision = _resolve_layer_precision(args, layer_idx)
    batch_seq = batch_size * seq_length
    flops = (
        3
        * 4
        * batch_seq
        * hidden_size
        * p
        * (
            hidden_size
            + (hidden_size * (g / num_attention_heads))
            + (seq_length / 2)
        )
    )
    return [
        _make_op(
            submodule="attention",
            operator_name="hybrid_attention_block",
            operator_kind="attention_block",
            precision=_resolve_linear_precision(args, f"{prefix}attention", base_precision, quant_recipe),
            shape=_local_attention_shape(args, num_attention_heads, kv_channels, gqa_groups),
            count="1 x 3",
            flops=flops,
        )
    ]


def _hybrid_mamba_ops(args, batch_size: int, layer_idx: int, prefix: str, quant_recipe):
    hidden_size = _getattr(args, "hidden_size")
    seq_length = _getattr(args, "seq_length")
    state_dim = _getattr(args, "mamba_state_dim")
    head_dim = _getattr(args, "mamba_head_dim")
    num_groups = _getattr(args, "mamba_num_groups")
    num_heads = _getattr(args, "mamba_num_heads")
    d_in = 2 * hidden_size
    nheads = num_heads if num_heads else d_in // head_dim
    batch_seq = batch_size * seq_length
    base_precision = _resolve_layer_precision(args, layer_idx)

    return [
        _make_op(
            submodule="mamba",
            operator_name="in_projection",
            operator_kind="gemm",
            precision=_resolve_linear_precision(args, f"{prefix}mixer.in_proj", base_precision, quant_recipe),
            shape=_column_parallel_gemm_shape(
                args,
                hidden_size,
                2 * d_in + 2 * num_groups * state_dim + nheads,
                input_label="hidden",
                output_label="mamba_in",
            ),
            count="3",
            flops=(
                3
                * 2
                * batch_seq
                * hidden_size
                * (2 * d_in + 2 * num_groups * state_dim + nheads)
            ),
        ),
        _make_op(
            submodule="mamba",
            operator_name="ssd_scan",
            operator_kind="ssm_scan",
            precision=_resolve_non_gemm_precision(args, f"{prefix}mixer.ssm_scan", quant_recipe),
            shape=f"({_local_token_dim(args)}, {d_in}, state_dim={state_dim})",
            count="1 x 3",
            flops=3 * 7 * batch_seq * d_in * state_dim,
        ),
        _make_op(
            submodule="mamba",
            operator_name="out_projection",
            operator_kind="gemm",
            precision=_resolve_linear_precision(args, f"{prefix}mixer.out_proj", base_precision, quant_recipe),
            shape=_row_parallel_gemm_shape(
                args, d_in, hidden_size, input_label="mamba_inner", output_label="hidden"
            ),
            count="3",
            flops=3 * 2 * batch_seq * d_in * hidden_size,
        ),
    ]


def _hybrid_moe_ops(args, batch_size: int, layer_idx: int, prefix: str, quant_recipe):
    hidden_size = _getattr(args, "hidden_size")
    moe_ffn_hidden_size = _getattr(args, "moe_ffn_hidden_size", None)
    if moe_ffn_hidden_size is None:
        moe_ffn_hidden_size = _getattr(args, "ffn_hidden_size")
    shared_expert_ffn_hidden_size = _getattr(args, "moe_shared_expert_intermediate_size", None)
    if shared_expert_ffn_hidden_size is None:
        shared_expert_ffn_hidden_size = 0
    moe_latent_size = _getattr(args, "moe_latent_size", None)
    num_experts_routed_to = _getattr(args, "moe_router_topk")
    seq_length = _getattr(args, "seq_length")
    batch_seq = batch_size * seq_length
    fc1_multiplier = 2 if _getattr(args, "swiglu", False) else 1
    base_precision = _resolve_layer_precision(args, layer_idx)
    ops: List[TheoreticalFlopOp] = []

    routed_input_size = hidden_size if moe_latent_size is None else moe_latent_size
    if moe_latent_size is not None:
        ops.extend(
            [
                _make_op(
                    submodule="moe_routed_experts",
                    operator_name="latent_up_projection",
                    operator_kind="gemm",
                    precision=_resolve_linear_precision(
                        args, f"{prefix}mlp.experts.latent_up_proj", base_precision, quant_recipe
                    ),
                    shape=_replicated_gemm_shape(
                        args,
                        hidden_size,
                        moe_latent_size,
                        input_label="hidden",
                        output_label="moe_latent",
                    ),
                    count=f"3 x topk={num_experts_routed_to}",
                    flops=3 * 2 * batch_seq * hidden_size * moe_latent_size,
                ),
                _make_op(
                    submodule="moe_routed_experts",
                    operator_name="latent_down_projection",
                    operator_kind="gemm",
                    precision=_resolve_linear_precision(
                        args, f"{prefix}mlp.experts.latent_down_proj", base_precision, quant_recipe
                    ),
                    shape=_replicated_gemm_shape(
                        args,
                        moe_latent_size,
                        hidden_size,
                        input_label="moe_latent",
                        output_label="hidden",
                    ),
                    count=f"3 x topk={num_experts_routed_to}",
                    flops=3 * 2 * batch_seq * hidden_size * moe_latent_size,
                ),
            ]
        )

    ops.extend(
        [
            _make_op(
                submodule="moe_routed_experts",
                operator_name="routed_fc1",
                operator_kind="gemm",
                precision=_resolve_linear_precision(
                    args, f"{prefix}mlp.experts.linear_fc1", base_precision, quant_recipe
                ),
                shape=_column_parallel_gemm_shape(
                    args,
                    routed_input_size,
                    fc1_multiplier * moe_ffn_hidden_size,
                    input_label="moe_routed_in",
                    output_label="moe_ffn_hidden" if fc1_multiplier == 1 else "2*moe_ffn_hidden",
                ),
                count=f"3 x topk={num_experts_routed_to}",
                flops=(
                    3
                    * 2
                    * batch_seq
                    * routed_input_size
                    * moe_ffn_hidden_size
                    * num_experts_routed_to
                    * fc1_multiplier
                ),
            ),
            _make_op(
                submodule="moe_routed_experts",
                operator_name="routed_fc2",
                operator_kind="gemm",
                precision=_resolve_linear_precision(
                    args, f"{prefix}mlp.experts.linear_fc2", base_precision, quant_recipe
                ),
                shape=_row_parallel_gemm_shape(
                    args,
                    moe_ffn_hidden_size,
                    routed_input_size,
                    input_label="moe_ffn_hidden",
                    output_label="moe_routed_in",
                ),
                count=f"3 x topk={num_experts_routed_to}",
                flops=(
                    3
                    * 2
                    * batch_seq
                    * routed_input_size
                    * moe_ffn_hidden_size
                    * num_experts_routed_to
                ),
            ),
        ]
    )

    if shared_expert_ffn_hidden_size > 0:
        ops.extend(
            [
                _make_op(
                    submodule="moe_shared_experts",
                    operator_name="shared_fc1",
                    operator_kind="gemm",
                    precision=_resolve_linear_precision(
                        args, f"{prefix}mlp.shared_experts.linear_fc1", base_precision, quant_recipe
                    ),
                    shape=_column_parallel_gemm_shape(
                        args,
                        hidden_size,
                        fc1_multiplier * shared_expert_ffn_hidden_size,
                        input_label="hidden",
                        output_label=(
                            "shared_ffn_hidden"
                            if fc1_multiplier == 1
                            else "2*shared_ffn_hidden"
                        ),
                    ),
                    count="3",
                    flops=(
                        3
                        * 2
                        * batch_seq
                        * hidden_size
                        * shared_expert_ffn_hidden_size
                        * fc1_multiplier
                    ),
                ),
                _make_op(
                    submodule="moe_shared_experts",
                    operator_name="shared_fc2",
                    operator_kind="gemm",
                    precision=_resolve_linear_precision(
                        args, f"{prefix}mlp.shared_experts.linear_fc2", base_precision, quant_recipe
                    ),
                    shape=_row_parallel_gemm_shape(
                        args,
                        shared_expert_ffn_hidden_size,
                        hidden_size,
                        input_label="shared_ffn_hidden",
                        output_label="hidden",
                    ),
                    count="3",
                    flops=3 * 2 * batch_seq * hidden_size * shared_expert_ffn_hidden_size,
                ),
            ]
        )
    return ops


def _build_hybrid_groups(args, batch_size: int, quant_recipe) -> Tuple[TheoreticalFlopGroup, ...]:
    groups: List[TheoreticalFlopGroup] = []
    for layer_idx, layer_type in enumerate(_hybrid_layer_types(args)):
        prefix = f"decoder.layers.{layer_idx}."
        if layer_type == "hybrid_attention_layer":
            ops = _hybrid_attention_ops(args, batch_size, layer_idx, prefix, quant_recipe)
        elif layer_type == "hybrid_mamba_layer":
            ops = _hybrid_mamba_ops(args, batch_size, layer_idx, prefix, quant_recipe)
        elif layer_type == "hybrid_mlp_layer":
            ops = _dense_mlp_ops(args, batch_size, layer_idx, prefix, quant_recipe)
        elif layer_type == "hybrid_moe_layer":
            ops = _hybrid_moe_ops(args, batch_size, layer_idx, prefix, quant_recipe)
        else:
            continue
        groups.append(
            TheoreticalFlopGroup(group_name=layer_type, layer_type=layer_type, count=1, ops=tuple(ops))
        )

    groups.append(_output_head_ops(args, batch_size, 1, quant_recipe))
    return _merge_groups(groups)


def _build_hybrid_stage_groups(
    args, batch_size: int, quant_recipe
) -> Dict[Tuple[int, int], Tuple[TheoreticalFlopGroup, ...]]:
    layer_types = _hybrid_layer_types(args)
    decoder_stage_map, _, output_stage, stage_list = _build_stage_assignment(args, len(layer_types), 0)
    stage_groups: Dict[Tuple[int, int], List[TheoreticalFlopGroup]] = {
        stage_id: [] for stage_id in stage_list
    }

    for layer_idx, layer_type in enumerate(layer_types):
        prefix = f"decoder.layers.{layer_idx}."
        if layer_type == "hybrid_attention_layer":
            ops = _hybrid_attention_ops(args, batch_size, layer_idx, prefix, quant_recipe)
        elif layer_type == "hybrid_mamba_layer":
            ops = _hybrid_mamba_ops(args, batch_size, layer_idx, prefix, quant_recipe)
        elif layer_type == "hybrid_mlp_layer":
            ops = _dense_mlp_ops(args, batch_size, layer_idx, prefix, quant_recipe)
        elif layer_type == "hybrid_moe_layer":
            ops = _hybrid_moe_ops(args, batch_size, layer_idx, prefix, quant_recipe)
        else:
            continue
        stage_groups[decoder_stage_map[layer_idx]].append(
            TheoreticalFlopGroup(group_name=layer_type, layer_type=layer_type, count=1, ops=tuple(ops))
        )

    stage_groups[output_stage].append(_output_head_ops(args, batch_size, 1, quant_recipe))
    return {stage_id: _merge_groups(groups) for stage_id, groups in stage_groups.items() if groups}


def _infer_num_microbatches(args, batch_size: int, explicit_num_microbatches: Optional[int]) -> int:
    if explicit_num_microbatches is not None:
        return int(explicit_num_microbatches)
    micro_batch_size = max(1, _micro_batch_size(args))
    data_parallel_size = max(1, int(_getattr(args, "data_parallel_size", 1) or 1))
    effective_micro_batch = micro_batch_size * data_parallel_size
    if batch_size % effective_micro_batch == 0:
        return max(1, batch_size // effective_micro_batch)
    return 1


def _aggregate_share_totals(
    layer_groups: Sequence[TheoreticalFlopGroup],
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, float],
    Dict[str, Dict[str, float]],
]:
    submodule_totals: Dict[str, float] = defaultdict(float)
    operator_totals: Dict[str, float] = defaultdict(float)
    submodule_operator_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    submodule_operator_precision_totals: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    operator_shape_precision_totals: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    precision_totals: Dict[str, float] = defaultdict(float)
    precision_operator_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for group in layer_groups:
        for op in group.ops:
            global_flops = group.count * op.flops
            share_operator = _share_operator_label(op)
            submodule_totals[op.submodule] += global_flops
            operator_totals[op.operator_kind] += global_flops
            submodule_operator_totals[op.submodule][share_operator] += global_flops
            submodule_operator_precision_totals[op.submodule][share_operator][op.precision] += global_flops
            operator_shape_precision_totals[share_operator][_shape_share_key(op.shape)][
                op.precision
            ] += global_flops
            precision_totals[op.precision] += global_flops
            precision_operator_totals[op.precision][share_operator] += global_flops

    return (
        dict(sorted(submodule_totals.items(), key=lambda item: (-item[1], item[0]))),
        dict(sorted(operator_totals.items(), key=lambda item: (-item[1], item[0]))),
        {
            submodule: dict(sorted(operator_totals.items(), key=lambda item: (-item[1], item[0])))
            for submodule, operator_totals in sorted(
                submodule_operator_totals.items(), key=lambda item: (-sum(item[1].values()), item[0])
            )
        },
        {
            submodule: {
                operator_kind: dict(
                    sorted(precision_totals.items(), key=lambda item: (-item[1], item[0]))
                )
                for operator_kind, precision_totals in sorted(
                    operator_totals.items(), key=lambda item: (-sum(item[1].values()), item[0])
                )
            }
            for submodule, operator_totals in sorted(
                submodule_operator_precision_totals.items(),
                key=lambda item: (
                    -sum(
                        flops
                        for precision_totals in item[1].values()
                        for flops in precision_totals.values()
                    ),
                    item[0],
                ),
            )
        },
        {
            operator_kind: {
                shape: dict(
                    sorted(shape_precision_totals.items(), key=lambda item: (-item[1], item[0]))
                )
                for shape, shape_precision_totals in sorted(
                    shape_totals.items(), key=lambda item: (-sum(item[1].values()), item[0])
                )
            }
            for operator_kind, shape_totals in sorted(
                operator_shape_precision_totals.items(),
                key=lambda item: (
                    -sum(
                        flops
                        for shape_precision_totals in item[1].values()
                        for flops in shape_precision_totals.values()
                    ),
                    item[0],
                ),
            )
        },
        dict(sorted(precision_totals.items(), key=lambda item: (-item[1], item[0]))),
        {
            precision: dict(sorted(operator_totals.items(), key=lambda item: (-item[1], item[0])))
            for precision, operator_totals in sorted(
                precision_operator_totals.items(), key=lambda item: (-sum(item[1].values()), item[0])
            )
        },
    )


def _build_stage_reports(
    stage_group_map: Dict[Tuple[int, int], Tuple[TheoreticalFlopGroup, ...]]
) -> Tuple[TheoreticalFlopStage, ...]:
    merged: Dict[Tuple, Dict[str, object]] = {}
    for stage_id, layer_groups in stage_group_map.items():
        merged_groups = _merge_groups(layer_groups)
        key = tuple(
            (
                group.count,
                _group_signature(group),
            )
            for group in merged_groups
        )
        (
            _submodule_totals,
            _operator_totals,
            _submodule_operator_totals,
            submodule_operator_precision_totals,
            operator_shape_precision_totals,
            _precision_totals,
            precision_operator_totals,
        ) = _aggregate_share_totals(merged_groups)
        if key not in merged:
            merged[key] = {
                "stage_ids": [stage_id],
                "layer_groups": merged_groups,
                "total_flops": sum(group.total_flops for group in merged_groups),
                "submodule_operator_precision_totals": submodule_operator_precision_totals,
                "operator_shape_precision_totals": operator_shape_precision_totals,
                "precision_operator_totals": precision_operator_totals,
            }
        else:
            merged[key]["stage_ids"].append(stage_id)

    return tuple(
        TheoreticalFlopStage(
            stage_ids=tuple(value["stage_ids"]),
            layer_groups=value["layer_groups"],
            total_flops=value["total_flops"],
            submodule_operator_precision_totals=value["submodule_operator_precision_totals"],
            operator_shape_precision_totals=value["operator_shape_precision_totals"],
            precision_operator_totals=value["precision_operator_totals"],
        )
        for value in sorted(merged.values(), key=lambda item: (item["stage_ids"][0][1], item["stage_ids"][0][0]))
    )


def get_theoretical_flop_report(
    args, batch_size: int, num_microbatches: Optional[int] = None
) -> TheoreticalFlopReport:
    """Build a detailed theoretical FLOPs report aligned with the existing formulas."""
    quant_recipe = _load_quant_recipe(args)
    batch_size = int(batch_size)
    num_microbatches = _infer_num_microbatches(args, batch_size, num_microbatches)
    if _getattr(args, "is_hybrid_model", False):
        layer_groups = _build_hybrid_groups(args, batch_size, quant_recipe)
        stage_group_map = _build_hybrid_stage_groups(args, batch_size, quant_recipe)
    else:
        layer_groups = _build_transformer_groups(args, batch_size, quant_recipe)
        stage_group_map = _build_transformer_stage_groups(args, batch_size, quant_recipe)

    total_flops = sum(group.total_flops for group in layer_groups)
    stage_reports = _build_stage_reports(stage_group_map)
    (
        submodule_totals,
        operator_totals,
        submodule_operator_totals,
        submodule_operator_precision_totals,
        operator_shape_precision_totals,
        precision_totals,
        precision_operator_totals,
    ) = _aggregate_share_totals(layer_groups)

    return TheoreticalFlopReport(
        batch_size=batch_size,
        micro_batch_size=_micro_batch_size(args),
        data_parallel_size=max(1, int(_getattr(args, "data_parallel_size", 1) or 1)),
        num_microbatches=num_microbatches,
        seq_length=_getattr(args, "seq_length"),
        total_flops=total_flops,
        layer_groups=layer_groups,
        stage_reports=stage_reports,
        submodule_totals=submodule_totals,
        operator_totals=operator_totals,
        submodule_operator_totals=submodule_operator_totals,
        submodule_operator_precision_totals=submodule_operator_precision_totals,
        operator_shape_precision_totals=operator_shape_precision_totals,
        precision_totals=precision_totals,
        precision_operator_totals=precision_operator_totals,
    )


def _format_share(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{100.0 * numerator / denominator:.2f}%"


def _format_count(
    raw_count: str, num_microbatches: int, micro_batch_size: int, data_parallel_size: int
) -> str:
    factors = [factor.strip() for factor in raw_count.split(" x ")]
    fbw_value: Optional[str] = None
    extra_parts: List[Tuple[str, str]] = []

    for factor in factors:
        if factor == str(FWD_BWD_EXPANSION_FACTOR):
            fbw_value = str(FWD_BWD_EXPANSION_FACTOR)
            continue
        if factor.startswith("topk="):
            extra_parts.append(("topk", factor.split("=", 1)[1]))
            continue
        value = int(factor)
        if value != 1:
            extra_parts.append(("terms", str(value)))

    symbolic_parts: List[str] = []
    numeric_parts: List[str] = []
    if fbw_value is not None:
        symbolic_parts.append("fbw")
        numeric_parts.append(fbw_value)
    symbolic_parts.append("mbs")
    numeric_parts.append(str(micro_batch_size))
    symbolic_parts.append("nbs")
    numeric_parts.append(str(num_microbatches))
    symbolic_parts.append("dp")
    numeric_parts.append(str(data_parallel_size))
    for label, value in extra_parts:
        symbolic_parts.append(label)
        numeric_parts.append(value)

    total = 1
    for value in numeric_parts:
        total *= int(value)

    if len(symbolic_parts) == 1:
        return f"count={symbolic_parts[0]}={numeric_parts[0]}"
    return f"count={'*'.join(symbolic_parts)}={'*'.join(numeric_parts)}={total}"


def _shape_share_key(shape: str) -> str:
    if shape.startswith("(m,n,k)="):
        parts = shape.split("=", 2)
        if len(parts) == 3:
            return parts[2]
    if shape.startswith("groups(~m,n,k)="):
        parts = shape.rsplit("=", 1)
        if len(parts) == 2:
            return parts[1]
    return shape


def _format_op_notes(op: TheoreticalFlopOp) -> str:
    notes: List[str] = []

    if op.operator_kind == "core_attention":
        notes.append("(causal)")

    if op.operator_name.endswith("fc1") and "2*" in op.shape:
        notes.append("(swiglu)")

    if not notes:
        return ""
    return " " + " ".join(notes)


def _append_three_level_share(
    lines: List[str],
    title: str,
    totals: Dict[str, Dict[str, Dict[str, float]]],
    total_flops: float,
    base_indent: str = "  ",
) -> None:
    lines.append("")
    lines.append(f"{base_indent}{title}:")
    level1_indent = base_indent + "  "
    level2_indent = base_indent + "    "
    level3_indent = base_indent + "      "
    for level1_label, level2_totals in totals.items():
        level1_flops = sum(
            flops for level3_totals in level2_totals.values() for flops in level3_totals.values()
        )
        if len(level2_totals) == 1:
            level2_label, level3_totals = next(iter(level2_totals.items()))
            if len(level3_totals) == 1:
                level3_label, flops = next(iter(level3_totals.items()))
                lines.append(
                    level1_indent +
                    f"{level1_label} | {level2_label} | {level3_label}: "
                    f"{flops / 10**12:.3f} ({_format_share(flops, total_flops)})"
                )
                continue

            lines.append(
                level1_indent +
                f"{level1_label} | {level2_label}: "
                f"{level1_flops / 10**12:.3f} ({_format_share(level1_flops, total_flops)})"
            )
            for level3_label, flops in level3_totals.items():
                lines.append(
                    level2_indent +
                    f"{level3_label}: {flops / 10**12:.3f} ({_format_share(flops, total_flops)})"
                )
            continue

        lines.append(
            f"{level1_indent}{level1_label}: {level1_flops / 10**12:.3f} ({_format_share(level1_flops, total_flops)})"
        )
        for level2_label, level3_totals in level2_totals.items():
            level2_flops = sum(level3_totals.values())
            if len(level3_totals) == 1:
                level3_label, flops = next(iter(level3_totals.items()))
                lines.append(
                    level2_indent +
                    f"{level2_label} | {level3_label}: {flops / 10**12:.3f} ({_format_share(flops, total_flops)})"
                )
                continue

            lines.append(
                f"{level2_indent}{level2_label}: {level2_flops / 10**12:.3f} ({_format_share(level2_flops, total_flops)})"
            )
            for level3_label, flops in level3_totals.items():
                lines.append(
                    level3_indent +
                    f"{level3_label}: {flops / 10**12:.3f} ({_format_share(flops, total_flops)})"
                )


def _append_two_level_share(
    lines: List[str],
    title: str,
    totals: Dict[str, Dict[str, float]],
    total_flops: float,
    base_indent: str = "  ",
) -> None:
    lines.append("")
    lines.append(f"{base_indent}{title}:")
    level1_indent = base_indent + "  "
    level2_indent = base_indent + "    "
    for level1_label, level2_totals in totals.items():
        level1_flops = sum(level2_totals.values())
        if len(level2_totals) == 1:
            level2_label, flops = next(iter(level2_totals.items()))
            lines.append(
                level1_indent +
                f"{level1_label} | {level2_label}: "
                f"{flops / 10**12:.3f} ({_format_share(flops, total_flops)})"
            )
            continue

        lines.append(
            f"{level1_indent}{level1_label}: {level1_flops / 10**12:.3f} ({_format_share(level1_flops, total_flops)})"
        )
        for level2_label, flops in level2_totals.items():
            lines.append(
                f"{level2_indent}{level2_label}: {flops / 10**12:.3f} ({_format_share(flops, total_flops)})"
            )


def _format_stage_id(stage_id: Tuple[int, int], has_vpp: bool) -> str:
    pp_rank, vp_rank = stage_id
    if has_vpp:
        return f"pp={pp_rank},vpp={vp_rank}"
    return f"pp={pp_rank}"


def _format_stage_ids(stage_ids: Tuple[Tuple[int, int], ...], has_vpp: bool) -> str:
    if not has_vpp:
        return f"pp={','.join(str(pp_rank) for pp_rank, _ in stage_ids)}"

    grouped_pp: Dict[int, List[int]] = defaultdict(list)
    for pp_rank, vp_rank in stage_ids:
        grouped_pp[vp_rank].append(pp_rank)

    parts = []
    for vp_rank in sorted(grouped_pp):
        pp_list = ",".join(str(pp_rank) for pp_rank in sorted(grouped_pp[vp_rank]))
        parts.append(f"pp={pp_list},vpp={vp_rank}")
    return "; ".join(parts)


def _scale_two_level_totals(
    totals: Dict[str, Dict[str, float]], factor: int
) -> Dict[str, Dict[str, float]]:
    if factor == 1:
        return totals
    return {
        level1_label: {level2_label: flops * factor for level2_label, flops in level2_totals.items()}
        for level1_label, level2_totals in totals.items()
    }


def format_theoretical_flop_report(
    report: TheoreticalFlopReport, *, reference_total_flops: Optional[float] = None
) -> str:
    """Format a startup-friendly textual report."""
    total_flops = report.total_flops
    total_tflops = total_flops / 10**12
    seq_tokens = report.batch_size * report.seq_length
    lines = [
        "",
        "#" * 100,
        "### THEORETICAL FLOPS REPORT START ###",
        "#" * 100,
        "Theoretical FLOPs report (startup)",
        "  emitted before torch.distributed and model-parallel group initialization",
        (
            f"  global_batch_size={report.batch_size}, local_num_microbatches={report.num_microbatches}, "
            f"micro_batch_size_merged_in_shape={report.micro_batch_size}, seq_length={report.seq_length}, "
            f"tokens_per_global_batch={seq_tokens}"
        ),
        (
            f"  tflops={total_tflops:.3f} / global batch, "
            f"per_sequence={total_tflops / max(report.batch_size, 1):.6f} TFLOP, "
            f"per_token={total_tflops / max(seq_tokens, 1):.9f} TFLOP"
        ),
        (
            "  shape scope: per-rank operator call. For TP linear GEMMs, m is the "
            "matmul sequence extent after CP sharding only; SP is communication on "
            "the sequence axis, not an extra TP GEMM partition"
        ),
        "  TP fact: ColumnParallelLinear weight=[out/tp, in], RowParallelLinear weight=[out, in/tp]",
        "  SP fact: sequence_parallel shards/gathers the first dimension (seq), separate from TP weight partition",
        "  flops/share scope: global aggregated theoretical FLOPs",
        "  grouped-layer and share values below use TFLOP units",
    ]
    if reference_total_flops is not None:
        rel_err = abs(reference_total_flops - total_flops) / max(abs(reference_total_flops), 1.0)
        lines.append(
            f"  reference_total={reference_total_flops / 10**12:.3f} TFLOP, relative_error={rel_err:.3e}"
        )

    lines.append("")
    lines.append("  pp/vpp stages:")
    has_vpp = any(stage_id[1] != 0 for stage in report.stage_reports for stage_id in stage.stage_ids)
    stage_precision_operator_totals: Dict[str, Dict[str, Dict[str, float]]] = {}
    for stage in report.stage_reports:
        stage_label = _format_stage_ids(stage.stage_ids, has_vpp)
        stage_precision_operator_totals[stage_label] = _scale_two_level_totals(
            stage.precision_operator_totals, len(stage.stage_ids)
        )
        lines.append(
            f"    [{stage_label}] merged_stage_count={len(stage.stage_ids)}, "
            f"stage_tflops={stage.total_flops / 10**12:.3f}"
        )
        lines.append("      grouped layers / pseudo-layers:")
        for group in stage.layer_groups:
            lines.append(
                f"        [{group.group_name}] merged_layer_count={group.count}, "
                f"per_layer_tflops={sum(op.flops for op in group.ops) / 10**12:.3f}"
            )
            for op in group.ops:
                lines.append(
                    "          "
                    f"{op.submodule} | {op.operator_name} | {op.operator_kind} | precision={op.precision}"
                )
                lines.append(
                    "            "
                    f"{_format_count(op.count, report.num_microbatches, report.micro_batch_size, report.data_parallel_size)}"
                )
                lines.append(f"            shape={op.shape}{_format_op_notes(op)}")
                lines.append(f"            per_layer_tflops={op.flops / 10**12:.3f}")

    lines.append("")
    lines.append("total share:")
    _append_three_level_share(
        lines,
        "submodule -> operator -> precision share",
        report.submodule_operator_precision_totals,
        total_flops,
        base_indent="  ",
    )
    _append_three_level_share(
        lines,
        "operator -> shape -> precision share",
        report.operator_shape_precision_totals,
        total_flops,
        base_indent="  ",
    )
    _append_three_level_share(
        lines,
        "stage -> precision -> operator share",
        stage_precision_operator_totals,
        total_flops,
        base_indent="  ",
    )

    lines.extend(
        [
            "",
            "#" * 100,
            "### THEORETICAL FLOPS REPORT END ###",
            "#" * 100,
            "",
        ]
    )

    return "\n".join(lines)
