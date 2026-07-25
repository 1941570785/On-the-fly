from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Iterable


@dataclass(frozen=True)
class PoseReliabilityConfig:
    enabled: bool = True
    retry_attempts: int = 2
    retry_min_2d3d: int = 500
    retry_min_pnp_inliers: int = 20
    multi_hypothesis_attempts: int = 2
    multi_hypothesis_min_2d3d: int = 2000
    multi_hypothesis_min_pnp_inliers: int = 700
    multi_hypothesis_max_pnp_ratio: float = 0.45
    multi_hypothesis_max_miniba_residual: float = 1.0
    min_support_gain: float = 0.15
    min_residual_gain: float = 0.20
    min_residual_support_ratio: float = 0.75
    rotation_floor_deg: float = 8.0
    rotation_margin_deg: float = 4.0
    rotation_scale: float = 1.75
    translation_floor: float = 0.35
    translation_margin: float = 0.12
    translation_scale: float = 2.5


@dataclass(frozen=True)
class ResponseSamplingConfig:
    enabled: bool = True
    alpha: float = 0.03
    min_selectivity: float = 1.8
    min_coverage_deficit: float = 0.08
    guide_min: float = 0.25
    guide_max: float = 4.0
    weight_min: float = 0.55
    weight_max: float = 1.75
    guard_min_evaluated: int = 3
    guard_max_bad_ratio: float = 0.67
    response_min_observations: int = 4
    response_min_improvement: float = 0.01
    response_max_degradation_ratio: float = 0.02


@dataclass(frozen=True)
class TransactionalRefinementConfig:
    enabled: bool = True
    fraction: float = 0.25
    max_extra_iterations: int = 8
    min_coverage_deficit: float = 0.08
    max_coverage_deficit: float = 0.35
    min_response_observations: int = 4
    min_relative_improvement: float = 0.03
    response_scale_reference: float = 0.12
    max_response_rebound_ratio: float = 0.01
    max_reference_views: int = 2
    runtime_target_ratio: float = 0.08
    dssim_relative_tolerance: float = 0.002
    depth_relative_tolerance: float = 0.01
    reference_relative_tolerance: float = 0.001
    consecutive_rejection_limit: int = 2


@dataclass(frozen=True)
class ASRGSConfig:
    method: str
    pose: PoseReliabilityConfig
    sampling: ResponseSamplingConfig
    refinement: TransactionalRefinementConfig

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


FINAL_CONFIG = ASRGSConfig(
    method="asr-gs",
    pose=PoseReliabilityConfig(),
    sampling=ResponseSamplingConfig(),
    refinement=TransactionalRefinementConfig(),
)


def resolve_config(
    method: str,
    ablations: Iterable[str] = (),
) -> ASRGSConfig:
    normalized_method = str(method).strip().lower()
    if normalized_method not in {"baseline", "asr-gs"}:
        raise ValueError(f"unsupported method: {method}")

    disabled = {str(name).strip().lower() for name in ablations}
    unknown = disabled.difference({"a", "b", "c"})
    if unknown:
        raise ValueError(f"unsupported ablation modules: {sorted(unknown)}")

    if normalized_method == "baseline":
        disabled = {"a", "b", "c"}

    return ASRGSConfig(
        method=normalized_method,
        pose=replace(FINAL_CONFIG.pose, enabled="a" not in disabled),
        sampling=replace(FINAL_CONFIG.sampling, enabled="b" not in disabled),
        refinement=replace(
            FINAL_CONFIG.refinement,
            enabled="c" not in disabled,
        ),
    )


def resolve_runtime_config(
    args: Any,
    *,
    inference_mode: bool,
) -> ASRGSConfig:
    config = getattr(args, "asr_gs_config", None)
    if isinstance(config, ASRGSConfig):
        return config
    if inference_mode:
        return resolve_config("baseline")
    raise AttributeError(
        "training requires args.asr_gs_config; construct arguments with get_args()"
    )
