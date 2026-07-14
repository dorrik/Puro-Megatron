# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank

from .optimizer_config import ParamKey, ParamPredicate, ParamWithNamePredicate

try:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object


logger = logging.getLogger(__name__)


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


def _scalar_param_overrides_factory(scalar_optimizer: str) -> Dict[ParamKey, Dict[str, Any]]:
    """Route embeddings and non-matrix parameters to the scalar optimizer."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': scalar_optimizer}
    }


def _muon_hyperball_param_overrides_factory(
    scalar_optimizer: str,
) -> Dict[ParamKey, Dict[str, Any]]:
    """Keep selected matrices on MuonH and route all other parameters to AdamW."""
    hyperball_matrix = ParamWithNamePredicate(
        name="hyperball_matrix",
        fn=_is_hyperball_matrix,
    )
    overrides = {
        ParamKey(with_name_predicate=hyperball_matrix): {'wd_mult': 0.0},
        ParamKey(
            predicate=ParamPredicate(
                name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
            )
        ): {'optimizer': scalar_optimizer},
        ParamKey(attr="is_attention_output_gate"): {'optimizer': scalar_optimizer},
        ParamKey(name=("*.router.weight", "*.router.bias", "*.gate_weight")): {
            'optimizer': scalar_optimizer
        },
    }
    return overrides


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class.
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(
        default_factory=_default_param_overrides_factory
    )


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if entry.config_to_kwargs is None:
        raise ValueError(f"No configuration adapter registered for optimizer '{eopt_name}'.")
    eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
    optimizer = entry.optimizer_cls(param_groups, **eopt_kwargs)
    return optimizer, entry.init_state_fn


def get_emerging_optimizer_param_overrides(
    eopt_name: str, config, entry: Optional[EmergingOptimizerEntry] = None
) -> Dict[ParamKey, Dict[str, Any]]:
    """Return parameter-routing overrides for an emerging optimizer.

    Puro-Megatron keeps parameter routing deliberately narrow: Muon-family optimizers
    handle selected matrix weights and AdamW handles the remaining parameters.
    """
    if eopt_name == 'muon':
        return _scalar_param_overrides_factory(config.muon_scalar_optimizer)
    if eopt_name == 'muon_hyperball':
        return _muon_hyperball_param_overrides_factory(config.muon_hyperball_scalar_optimizer)
    if entry is None:
        entry = _EMERGING_OPTIMIZERS[eopt_name]
    return entry.default_param_overrides


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _is_hyperball_matrix(param, name: str) -> bool:
    """Return whether a parameter follows the fixed-radius MuonH update."""
    if len(param.shape) != 2:
        return False
    if getattr(param, 'is_embedding_or_output_parameter', False):
        return False
    if getattr(param, 'is_attention_output_gate', False):
        return False
    if name.endswith(".bias"):
        return False
    if ".router." in name or name.endswith(".gate_weight"):
        return False
    return True


def _get_qkv_split_shapes(model_cfg) -> List[int]:
    """Compute QKV split shapes from model config."""
    return [
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels,
        model_cfg.kv_channels,
        model_cfg.kv_channels,
    ]


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


class _TensorParallelHyperballMixin:
    """Shared TP-aware norm helpers for Hyperball-style optimizers.

    Hyperball normalization should use the norm of the *global* TP parameter,
    not the local shard norm. For duplicated or non-TP parameters, the local
    norm is already the correct one and no collective is needed.
    """

    pg_collection: Optional[ProcessGroupCollection]
    hyperball_eps: float

    def _get_hyperball_tp_group(
        self, p: torch.Tensor
    ) -> Optional[torch.distributed.ProcessGroup]:
        """Return the TP process group relevant to ``p`` if one exists."""
        if self.pg_collection is None or not torch.distributed.is_initialized():
            return None
        if getattr(p, "expert_tp", False):
            return getattr(self.pg_collection, "expt_tp", None)
        return getattr(self.pg_collection, "tp", None)

    def _uses_tp_global_norm(self, p: torch.Tensor) -> bool:
        """Whether Hyperball norm for ``p`` must be reduced across TP ranks."""
        tp_group = self._get_hyperball_tp_group(p)
        if tp_group is None or get_pg_size(tp_group) <= 1:
            return False
        partition_dim = getattr(p, "partition_dim", None)
        return partition_dim is not None and partition_dim != -1

    def _hyperball_vector_norm(
        self, p: torch.Tensor, tensor: torch.Tensor, detach: bool = True
    ) -> torch.Tensor:
        """Compute the Frobenius norm of ``tensor`` using TP-global semantics."""
        value = tensor.detach() if detach else tensor
        norm_sq = value.float().pow(2).sum()
        if self._uses_tp_global_norm(p):
            tp_group = self._get_hyperball_tp_group(p)
            torch.distributed.all_reduce(norm_sq, group=tp_group)
        return norm_sq.sqrt().clamp_min(self.hyperball_eps)

    def _hyperball_vector_norm_sq(
        self, p: torch.Tensor, tensor: torch.Tensor, detach: bool = True
    ) -> torch.Tensor:
        """Compute an unclamped squared Frobenius norm with TP-global semantics."""
        value = tensor.detach() if detach else tensor
        norm_sq = value.float().pow(2).sum()
        if self._uses_tp_global_norm(p):
            tp_group = self._get_hyperball_tp_group(p)
            torch.distributed.all_reduce(norm_sq, group=tp_group)
        return norm_sq


# ===========================================================================
# Muon
# ===========================================================================


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    diagnostic_family = "muon"

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        effective_lr_mult: float | None = None,
        strict_effective_lr: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if effective_lr_mult is not None and effective_lr_mult <= 0.0:
            raise ValueError(f"Invalid Muon effective LR multiplier: {effective_lr_mult}")
        if strict_effective_lr and effective_lr_mult is None:
            raise ValueError("Muon strict effective LR requires an effective LR multiplier")
        if strict_effective_lr and not use_decoupled_weight_decay:
            raise ValueError("Muon strict effective LR requires decoupled weight decay")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.effective_lr_mult = effective_lr_mult
        self.strict_effective_lr = strict_effective_lr
        self._effective_lr_eps = 1e-12
        self._diagnostic_enabled = False
        self._diagnostic_stats: dict[str, dict[str, float]] = {}
        self._current_lr_by_param: dict[int, float] = {}
        self._current_weight_decay_by_param: dict[int, float] = {}

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def _get_muon_tp_group(
        self, p: torch.Tensor
    ) -> Optional[torch.distributed.ProcessGroup]:
        """Return the TP process group relevant to a parameter."""
        if self.pg_collection is None or not torch.distributed.is_initialized():
            return None
        if getattr(p, "expert_tp", False):
            return getattr(self.pg_collection, "expt_tp", None)
        return getattr(self.pg_collection, "tp", None)

    def _uses_muon_tp_global_rms(self, p: torch.Tensor) -> bool:
        tp_group = self._get_muon_tp_group(p)
        if tp_group is None or get_pg_size(tp_group) <= 1:
            return False
        partition_dim = getattr(p, "partition_dim", None)
        return partition_dim is not None and partition_dim != -1

    def _muon_tensor_rms(
        self,
        p: torch.Tensor,
        tensor: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        """Compute local or TP-global RMS in FP32."""
        value = tensor.detach() if detach else tensor
        norm_sq = value.float().pow(2).sum()
        numel = torch.tensor(float(value.numel()), device=value.device, dtype=torch.float32)
        if self._uses_muon_tp_global_rms(p):
            stats = torch.stack((norm_sq, numel))
            torch.distributed.all_reduce(stats, group=self._get_muon_tp_group(p))
            norm_sq, numel = stats[0], stats[1]
        return (norm_sq / numel.clamp_min(1.0)).sqrt()

    def _get_effective_lr_scale(
        self, p: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        """Scale an update so ||lr * update|| / ||weight|| equals lr times M."""
        assert self.effective_lr_mult is not None
        weight_rms = self._muon_tensor_rms(p, p)
        update_rms = self._muon_tensor_rms(p, update, detach=False)
        return (
            float(self.effective_lr_mult)
            * weight_rms.clamp_min(self._effective_lr_eps)
            / update_rms.clamp_min(self._effective_lr_eps)
        )

    def _tensor_geometry(
        self, p: torch.Tensor, update: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return TP-global weight/update squared norms and inner product in FP32."""
        weight = p.detach().float()
        update_fp32 = update.detach().float()
        stats = torch.stack(
            (
                weight.pow(2).sum(),
                update_fp32.pow(2).sum(),
                (weight * update_fp32).sum(),
            )
        )
        if self._uses_muon_tp_global_rms(p):
            torch.distributed.all_reduce(stats, group=self._get_muon_tp_group(p))
        return stats[0], stats[1], stats[2]

    def _get_strict_effective_lr_scale(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
        base_lr: float,
        weight_decay: float,
    ) -> torch.Tensor:
        """Solve the update scale for the target normalized pre/post weight distance."""
        assert self.effective_lr_mult is not None
        weight_norm_sq, update_norm_sq, weight_update_dot = self._tensor_geometry(p, update)
        eps = torch.tensor(self._effective_lr_eps, device=p.device, dtype=torch.float32)
        if base_lr == 0.0:
            return torch.zeros((), device=p.device, dtype=torch.float32)

        weight_norm = weight_norm_sq.clamp_min(0.0).sqrt().clamp_min(eps)
        safe_weight_norm_sq = weight_norm.square()
        radial = weight_update_dot / safe_weight_norm_sq
        tangential = (
            update_norm_sq / safe_weight_norm_sq - radial.square()
        ).clamp_min(0.0).sqrt().clamp_min(eps)

        max_chord = 2.0 - torch.finfo(torch.float32).eps
        target = torch.tensor(
            float(base_lr) * float(self.effective_lr_mult),
            device=p.device,
            dtype=torch.float32,
        ).clamp(min=0.0, max=max_chord)
        cosine = 1.0 - 0.5 * target.square()
        sine = target * (1.0 - 0.25 * target.square()).clamp_min(0.0).sqrt()
        denominator = (
            cosine * tangential + sine * (float(weight_decay) + radial)
        ).clamp_min(eps)
        actual_lr = sine / denominator
        return actual_lr / float(base_lr)

    @staticmethod
    def _get_param_module_tag(p: torch.Tensor) -> str:
        if getattr(p, "is_qkv", False):
            return "linear_qkv"
        name = getattr(p, "param_name", "")
        if "self_attention.linear_proj.weight" in name:
            return "attention_proj"
        if "mlp.linear_fc1.weight" in name:
            return "mlp_fc1"
        if "mlp.linear_fc2.weight" in name:
            return "mlp_fc2"
        return "matrix"

    def set_diagnostic_context(self, iteration: int, interval: int) -> None:
        """Enable effective-LR diagnostics for the selected optimizer step."""
        self._diagnostic_enabled = interval > 0 and iteration % interval == 0
        self._diagnostic_stats = {}
        if self._diagnostic_enabled:
            self._refresh_param_group_values()

    def pop_diagnostic_stats(self) -> dict[str, dict[str, float]]:
        stats = self._diagnostic_stats
        self._diagnostic_stats = {}
        return stats

    def _refresh_param_group_values(self) -> None:
        self._current_lr_by_param = {}
        self._current_weight_decay_by_param = {}
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            for p in group["params"]:
                self._current_lr_by_param[id(p)] = lr
                self._current_weight_decay_by_param[id(p)] = weight_decay

    def _record_update(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
        base_lr: float,
        applied_lr: float,
    ) -> None:
        tag = self._get_param_module_tag(p)
        stats = self._diagnostic_stats.setdefault(
            tag,
            {
                "count": 0.0,
                "weight_norm_sq": 0.0,
                "update_norm_sq": 0.0,
                "lr_sum": 0.0,
            },
        )
        weight_norm = float(p.detach().float().norm().item())
        update_norm = abs(float(base_lr)) * float(update.detach().float().norm().item())
        stats["count"] += 1.0
        stats["weight_norm_sq"] += weight_norm * weight_norm
        stats["update_norm_sq"] += update_norm * update_norm
        stats["lr_sum"] += float(applied_lr)

    def _apply_weight_decay_inplace(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        weight_decay: float,
    ) -> None:
        """Defer aligned Muon weight decay until the per-tensor LR is known."""
        if (
            self.effective_lr_mult is not None
            and getattr(self, "weight_decay_method", "l2") == "decoupled"
        ):
            return
        super()._apply_weight_decay_inplace(p, grad, lr, weight_decay)

    def _apply_deferred_weight_decay(self, p: torch.Tensor, lr: float) -> None:
        if getattr(self, "weight_decay_method", "l2") != "decoupled":
            return
        weight_decay = self._current_weight_decay_by_param.get(id(p), 0.0)
        if weight_decay:
            p.add_(p, alpha=(-float(weight_decay) * float(lr)))

    def pre_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Apply optional relative-update alignment before the Muon weight update."""
        base_lr = self._current_lr_by_param.get(id(p), 0.0)
        applied_lr = base_lr
        if self.effective_lr_mult is not None:
            weight_decay = self._current_weight_decay_by_param.get(id(p), 0.0)
            if self.strict_effective_lr:
                scale = self._get_strict_effective_lr_scale(
                    p, update, base_lr, weight_decay
                )
            else:
                scale = self._get_effective_lr_scale(p, update)
            update.mul_(scale.to(device=update.device, dtype=update.dtype))
            applied_lr = base_lr * float(scale.detach().float().item())
            self._apply_deferred_weight_decay(p, applied_lr)
        if self._diagnostic_enabled:
            self._record_update(p, update, base_lr, applied_lr)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        if not self._current_lr_by_param and (
            self.effective_lr_mult is not None or self._diagnostic_enabled
        ):
            self._refresh_param_group_values()
        try:
            return super().step(closure)
        finally:
            self._current_lr_by_param = {}
            self._current_weight_decay_by_param = {}

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, ' f'split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class TensorParallelMuonHyperball(_TensorParallelHyperballMixin, TensorParallelMuon):
    """Tensor Parallel MuonHyperball optimizer.

    Mirrors ``emerging_optimizers.orthogonalized_optimizers.MuonHyperball`` while
    preserving Megatron's tensor-parallel Muon path, including TP-aware Newton-Schulz
    and optional QKV splitting.
    """

    diagnostic_family = "hyperball"

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        hyperball_eps: float = 1e-8,
        hyperball_radius: float | None = None,
        lr_mult: float = 1.0,
    ) -> None:
        if hyperball_eps <= 0.0:
            raise ValueError(f"Invalid hyperball epsilon value: {hyperball_eps}")
        if hyperball_radius is not None and hyperball_radius <= 0.0:
            raise ValueError(f"Invalid hyperball radius value: {hyperball_radius}")
        if lr_mult <= 0.0:
            raise ValueError(f"Invalid hyperball learning-rate multiplier: {lr_mult}")
        self.hyperball_eps = hyperball_eps
        self.hyperball_radius = hyperball_radius
        self.hyperball_lr_mult = lr_mult
        self._hyperball_radius_cache: Dict[torch.Tensor, torch.Tensor] = {}
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    p_norm_sq = self._hyperball_vector_norm_sq(p, p)
                    if p_norm_sq.item() == 0:
                        raise ValueError(
                            "MuonHyperball requires all parameters to have non-zero norm. "
                            "Found parameter with zero norm."
                        )
                    p_norm = p_norm_sq.sqrt().clamp_min(self.hyperball_eps)
                    if self.hyperball_radius is not None:
                        p.mul_(self.hyperball_radius / p_norm.clamp_min(self.hyperball_eps))

    def _get_or_init_hyperball_radius(self, p: torch.Tensor) -> torch.Tensor:
        """Return the fixed hyperball radius for *p*, initializing it lazily.

        The radius cache is intentionally kept outside ``optimizer.state`` so it
        does not get serialized as a per-parameter optimizer tensor during
        dist-checkpoint save.
        """
        radius = self._hyperball_radius_cache.get(p)
        if radius is not None:
            return radius

        if self.hyperball_radius is None:
            radius = self._hyperball_vector_norm(p, p)
        else:
            radius = torch.tensor(
                self.hyperball_radius, device=p.device, dtype=torch.float32
            ).clamp_min(self.hyperball_eps)

        self._hyperball_radius_cache[p] = radius
        return radius

    def pre_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Apply MuonHyperball's norm-preserving pre-update normalization."""
        radius = self._get_or_init_hyperball_radius(p)
        update_norm = self._hyperball_vector_norm(p, update, detach=False)
        update.mul_(radius / update_norm)
        if self._diagnostic_enabled:
            effective_lr = self._current_lr_by_param.get(id(p), 0.0)
            self._record_update(p, update, effective_lr, effective_lr)

    def post_weight_update_fn_inplace(self, p: torch.Tensor) -> None:
        """Project the updated weight back to the stored radius."""
        radius = self._get_or_init_hyperball_radius(p)
        p_norm = self._hyperball_vector_norm(p, p, detach=False)
        p.mul_(radius / p_norm)

    def _refresh_param_group_values(self) -> None:
        """Cache the effective matrix-group LR, including the MuonH multiplier."""
        self._current_lr_by_param = {}
        self._current_weight_decay_by_param = {}
        for group in self.param_groups:
            lr = float(group["lr"]) * self.hyperball_lr_mult
            weight_decay = float(group.get("weight_decay", 0.0))
            for p in group["params"]:
                self._current_lr_by_param[id(p)] = lr
                self._current_weight_decay_by_param[id(p)] = weight_decay

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Apply the MuonH-only LR multiplier without changing scheduler groups."""
        if self._diagnostic_enabled and not self._current_lr_by_param:
            self._refresh_param_group_values()
        if self.hyperball_lr_mult == 1.0:
            return super().step(closure)

        for group in self.param_groups:
            group["lr"] *= self.hyperball_lr_mult
        try:
            return super().step(closure)
        finally:
            for group in self.param_groups:
                group["lr"] /= self.hyperball_lr_mult

    def load_state_dict(self, state_dict):
        """Reload optimizer state and rebuild lazy hyperball radii from current params."""
        super().load_state_dict(state_dict)
        self._hyperball_radius_cache.clear()


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _muon_hyperball_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuonHyperball constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.pop("effective_lr_mult", None)
    kwargs.pop("strict_effective_lr", None)
    kwargs.update(_kwargs_from_config(TensorParallelMuonHyperball, "muon_hyperball", config))
    kwargs["hyperball_eps"] = config.muon_hyperball_eps
    kwargs["hyperball_radius"] = config.muon_hyperball_radius
    return kwargs


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
        ),
        'muon_hyperball': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuonHyperball,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_hyperball_config_to_kwargs,
        ),
    }
)
