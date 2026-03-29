from types import SimpleNamespace

import pytest

from megatron.training.theoretical_flops import (
    format_theoretical_flop_report,
    get_theoretical_flop_report,
)


def _make_args(**overrides):
    args = dict(
        is_hybrid_model=False,
        num_layers=4,
        seq_length=16,
        hidden_size=32,
        kv_channels=8,
        num_attention_heads=4,
        group_query_attention=False,
        num_query_groups=4,
        attention_output_gate=False,
        num_experts=None,
        moe_layer_freq=1,
        moe_router_topk=2,
        moe_ffn_hidden_size=None,
        moe_shared_expert_intermediate_size=None,
        ffn_hidden_size=64,
        swiglu=False,
        mtp_num_layers=None,
        multi_latent_attention=False,
        experimental_attention_variant=None,
        linear_attention_freq=None,
        padded_vocab_size=128,
        bf16=True,
        fp16=False,
        fp8=None,
        fp4=None,
        first_last_layers_bf16=False,
        num_layers_at_start_in_bf16=0,
        num_layers_at_end_in_bf16=0,
        layer_precision_layout=None,
        fp8_dot_product_attention=False,
        fp8_multi_head_attention=False,
        kitchen_config_file=None,
        kitchen_recipe_number=None,
        te_precision_config_file=None,
        moe_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=False,
        micro_batch_size=1,
        data_parallel_size=1,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        attention_type=None,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _dense_transformer_reference_total(args, batch_size):
    if not args.group_query_attention:
        num_query_groups = args.num_attention_heads
    else:
        num_query_groups = args.num_query_groups

    fb = 3
    fma = 2
    query_projection_size = args.kv_channels * args.num_attention_heads
    key_projection_size = args.kv_channels * num_query_groups
    value_projection_size = args.kv_channels * num_query_groups
    gate_projection_size = query_projection_size if args.attention_output_gate else 0
    self_attention_term = fb * fma * (
        args.hidden_size
        * (
            query_projection_size
            + key_projection_size
            + value_projection_size
            + gate_projection_size
        )
        + query_projection_size * args.seq_length
        + query_projection_size * args.hidden_size
    )
    mlp_term = fb * fma * args.hidden_size * (args.ffn_hidden_size * 2) * args.num_layers
    attn_term = self_attention_term * args.num_layers
    logits_term = fb * fma * args.hidden_size * args.padded_vocab_size
    return batch_size * args.seq_length * (mlp_term + attn_term + logits_term)


def test_theoretical_flops_report_matches_dense_transformer_formula():
    args = _make_args()
    batch_size = 8

    report = get_theoretical_flop_report(args, batch_size)
    expected_total = _dense_transformer_reference_total(args, batch_size)

    assert report.total_flops == pytest.approx(expected_total)
    assert sum(group.total_flops for group in report.layer_groups) == pytest.approx(report.total_flops)
    assert sum(report.submodule_totals.values()) == pytest.approx(report.total_flops)
    assert sum(report.precision_totals.values()) == pytest.approx(report.total_flops)


def test_theoretical_flops_report_merges_edge_bf16_layers():
    args = _make_args(
        bf16=False,
        fp8="e4m3",
        first_last_layers_bf16=True,
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
    )

    report = get_theoretical_flop_report(args, batch_size=8)
    transformer_groups = [group for group in report.layer_groups if group.layer_type == "transformer_layer"]

    assert len(transformer_groups) == 2
    assert sorted(group.count for group in transformer_groups) == [2, 2]

    qkv_precisions = {
        next(op.precision for op in group.ops if op.operator_name == "qkv_projection")
        for group in transformer_groups
    }
    assert qkv_precisions == {"bf16", "fp8"}


def test_theoretical_flops_report_applies_te_precision_overrides(tmp_path):
    recipe_path = tmp_path / "te_precision.yaml"
    recipe_path.write_text(
        "\n".join(
            [
                "configs:",
                "  bf16:",
                '    transformer_engine_config_type: "TEQuantizationParams"',
                "    training_recipe: {}",
                "matchers:",
                "  qkv_bf16:",
                '    config: "bf16"',
                '    type: "glob"',
                '    pattern: "*.linear_qkv"',
                "    enabled: true",
            ]
        )
    )

    args = _make_args(bf16=False, fp8="e4m3", te_precision_config_file=str(recipe_path))
    report = get_theoretical_flop_report(args, batch_size=8)

    transformer_group = next(group for group in report.layer_groups if group.layer_type == "transformer_layer")
    qkv_op = next(op for op in transformer_group.ops if op.operator_name == "qkv_projection")
    proj_op = next(op for op in transformer_group.ops if op.operator_name == "output_projection")

    assert qkv_op.precision == "bf16"
    assert proj_op.precision == "fp8"
    assert sum(report.precision_totals.values()) == pytest.approx(report.total_flops)


def test_theoretical_flops_report_formats_count_and_tp_mlp_shapes():
    args = _make_args(
        num_layers=1,
        seq_length=16,
        hidden_size=32,
        ffn_hidden_size=64,
        padded_vocab_size=128,
        micro_batch_size=4,
        data_parallel_size=2,
        tensor_model_parallel_size=2,
    )

    report = get_theoretical_flop_report(args, batch_size=32, num_microbatches=4)
    transformer_group = next(group for group in report.layer_groups if group.layer_type == "transformer_layer")
    fc1_op = next(op for op in transformer_group.ops if op.operator_name == "fc1")
    fc2_op = next(op for op in transformer_group.ops if op.operator_name == "fc2")
    output_group = next(group for group in report.layer_groups if group.layer_type == "output_head")
    output_op = next(op for op in output_group.ops if op.operator_name == "logits_projection")

    assert fc1_op.shape == "(m,n,k)=(mbs*seq, ffn/tp, hidden)=(64, 32, 32)"
    assert fc2_op.shape == "(m,n,k)=(mbs*seq, hidden, ffn/tp)=(64, 32, 32)"
    assert output_op.shape == "(m,n,k)=(mbs*seq, padded_vocab/tp, hidden)=(64, 64, 32)"

    formatted = format_theoretical_flop_report(report)
    assert "count=fbw*nbs=3*4=12" in formatted
    assert "attention: " in formatted
    assert "gemm: " in formatted
    assert "bf16: " in formatted
