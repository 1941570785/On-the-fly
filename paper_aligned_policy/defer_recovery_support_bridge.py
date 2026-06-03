"""Defer-recoverable recovery support bridge (V1) — no R/V/Q/tau/density changes."""
from __future__ import annotations

from typing import Any


def bridge_enabled(args: Any) -> bool:
    return str(getattr(args, "paper_aligned_defer_recovery_support_bridge", "off") or "off") == "v1"


def _anchor_id_for_keyframe(scene_model: Any, keyframe_id: int) -> int:
    for anchor_id, anchor in enumerate(getattr(scene_model, "anchors", []) or []):
        if int(keyframe_id) in [int(x) for x in getattr(anchor, "keyframe_ids", [])]:
            return int(anchor_id)
    return -1


def _kf_quality_score(keyframe: Any) -> float:
    has_pt3d = float(keyframe.desc_kpts.has_pt3d.sum().item())
    origin = str(keyframe.info.get("_paper_aligned_commit_origin", ""))
    is_direct = origin in {"direct_admit", "direct"} or bool(
        keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False)
    )
    is_recovery = origin in {"true_recovery_commit", "early_seed_recovery_commit"}
    bonus = 500.0 if is_direct else (200.0 if is_recovery else 0.0)
    return has_pt3d + bonus


class DeferRecoverySupportBridge:
    """Anchor-transition bridge + recovery reference propagation (support-only)."""

    def __init__(self, args: Any) -> None:
        self.max_bridge_refs = int(getattr(args, "paper_aligned_bridge_max_refs", 4) or 4)
        self.top_k_per_anchor = int(getattr(args, "paper_aligned_bridge_top_k_per_anchor", 6) or 6)
        self._anchor_count_seen = len([])
        self._bridge_inventory: dict[int, list[int]] = {}
        self._last_meta: dict[str, Any] = {}

    def observe_anchor_count(self, scene_model: Any) -> list[dict[str, Any]]:
        """Call after place_anchor_if_needed; returns transition events."""
        events: list[dict[str, Any]] = []
        n_anchors = len(getattr(scene_model, "anchors", []) or [])
        if n_anchors <= self._anchor_count_seen:
            return events
        for new_anchor_id in range(self._anchor_count_seen, n_anchors):
            prev_anchor_id = int(new_anchor_id - 1)
            if prev_anchor_id < 0:
                continue
            prev_anchor = scene_model.anchors[prev_anchor_id]
            ranked: list[tuple[float, int]] = []
            for kid in getattr(prev_anchor, "keyframe_ids", []) or []:
                kid = int(kid)
                if kid < 0 or kid >= len(scene_model.keyframes):
                    continue
                kf = scene_model.keyframes[kid]
                if not bool(kf.info.get("_paper_aligned_support_eligible_recovery_keyframe", False)):
                    origin = str(kf.info.get("_paper_aligned_commit_origin", ""))
                    if origin not in {
                        "direct_admit",
                        "direct",
                        "true_recovery_commit",
                        "early_seed_recovery_commit",
                    }:
                        continue
                ranked.append((_kf_quality_score(kf), kid))
            ranked.sort(key=lambda x: -x[0])
            picked = [kid for _, kid in ranked[: self.top_k_per_anchor]]
            self._bridge_inventory[prev_anchor_id] = picked
            events.append(
                {
                    "event_type": "anchor_transition_inventory",
                    "new_anchor_id": int(new_anchor_id),
                    "bridge_source_anchor": int(prev_anchor_id),
                    "bridged_reference_ids": list(picked),
                    "bridge_reason": "anchor_transition_support_bridge_v1",
                }
            )
        self._anchor_count_seen = n_anchors
        return events

    def extend_recovery_candidates(
        self,
        scene_model: Any,
        desc_kpts: Any,
        candidate_indices: list[int],
        source_frame_id: int,
    ) -> tuple[list[int], dict[str, Any]]:
        current_anchor = _anchor_id_for_keyframe(
            scene_model, int(scene_model.keyframes[-1].index) if scene_model.keyframes else -1
        )
        if current_anchor < 0 and scene_model.keyframes:
            current_anchor = _anchor_id_for_keyframe(scene_model, int(scene_model.keyframes[-1].index))
        active_anchor = getattr(scene_model, "active_anchor", None)
        if active_anchor is not None:
            current_anchor = max(
                current_anchor,
                len(getattr(scene_model, "anchors", []) or []) - 1,
            )
        bridge_ids: list[int] = []
        sources: list[int] = []
        for src_anchor, kids in sorted(self._bridge_inventory.items()):
            if int(src_anchor) >= int(current_anchor):
                continue
            for kid in kids:
                if kid not in candidate_indices and kid not in bridge_ids:
                    bridge_ids.append(int(kid))
                    sources.append(int(src_anchor))
        bridge_ids = bridge_ids[: self.max_bridge_refs]
        meta = {
            "frame_id": int(source_frame_id),
            "anchor_id": int(current_anchor),
            "bridged_reference_ids": bridge_ids,
            "bridge_source_anchors": sources[: len(bridge_ids)],
            "bridge_reason": "recovery_pose_reference_bridge_v1",
            "candidate_count_before": len(candidate_indices),
        }
        self._last_meta = dict(meta)
        return bridge_ids, meta

    def mark_bridge_keyframes(self, scene_model: Any, bridge_ids: list[int], meta: dict[str, Any]) -> None:
        src_anchors = meta.get("bridge_source_anchors", []) or []
        for i, kid in enumerate(bridge_ids):
            if kid < 0 or kid >= len(scene_model.keyframes):
                continue
            kf = scene_model.keyframes[int(kid)]
            kf.info["_paper_aligned_bridge_ref"] = True
            kf.info["_paper_aligned_bridge_source_anchor"] = (
                int(src_anchors[i]) if i < len(src_anchors) else -1
            )
            kf.info["_paper_aligned_bridge_support_only"] = True

    def log_recovery_support_trace(
        self,
        runtime_gate: Any,
        source_frame_id: int,
        current_frame_id: int,
        pose_debug: dict[str, Any],
        support_trace: dict[str, Any],
        consensus_trace: dict[str, Any],
        bridge_meta: dict[str, Any],
    ) -> None:
        ref_ids = list(pose_debug.get("ref_keyframe_ids", []) or [])
        pnp_ids = set(int(x) for x in (pose_debug.get("pnp_ref_keyframe_ids", []) or []))
        miniba_ids = set(int(x) for x in (pose_debug.get("miniba_ref_keyframe_ids", []) or []))
        bridged_used = [int(x) for x in ref_ids if int(x) in set(bridge_meta.get("bridged_reference_ids", []) or [])]
        row = {
            "source_frame_id": int(source_frame_id),
            "current_frame_id": int(current_frame_id),
            "anchor_id": bridge_meta.get("anchor_id", -1),
            "recovery_pool_entered": True,
            "recovery_attempted": True,
            "recovery_pose_success": bool(int(pose_debug.get("num_pnp_inliers", 0) or 0) >= 4)
            and str(pose_debug.get("failure_reason", "") or "") == "",
            "miniba_success": str(pose_debug.get("failure_reason", "") or "") == "",
            "true_source_materialized": False,
            "materialization_fail_reason": str(pose_debug.get("failure_reason", "") or ""),
            "match_count": int(pose_debug.get("match_count_total", 0) or 0),
            "valid_2d3d_count": int(pose_debug.get("num_2d3d_correspondences", 0) or 0),
            "pnp_inliers": int(pose_debug.get("num_pnp_inliers", 0) or 0),
            "miniba_inliers": int(pose_debug.get("num_miniba_inliers", 0) or 0),
            "selected_ref_count": len(ref_ids),
            "effective_ref_count_in_matching": len(ref_ids),
            "effective_ref_count_in_pnp": len(pnp_ids),
            "effective_ref_count_in_miniba": len(miniba_ids),
            "bridged_reference_ids": list(bridge_meta.get("bridged_reference_ids", []) or []),
            "bridged_refs_used_in_matching": bridged_used,
            "bridge_source_anchors": list(bridge_meta.get("bridge_source_anchors", []) or []),
            "valid_2d3d_before_consensus": int(consensus_trace.get("valid_2d3d_before_consensus", 0) or 0),
            "valid_2d3d_after_pose": int(pose_debug.get("num_2d3d_correspondences", 0) or 0),
            "pnp_inliers_before_consensus": int(consensus_trace.get("pnp_inliers_before_consensus", 0) or 0),
            "pnp_inliers_after_pose": int(pose_debug.get("num_pnp_inliers", 0) or 0),
            "miniba_inliers_after_pose": int(pose_debug.get("num_miniba_inliers", 0) or 0),
            "verified_2d2d_match_count": int(support_trace.get("verified_2d2d_match_count", 0) or 0),
        }
        runtime_gate.append_recovery_support_trace_event(row)

    def log_anchor_bridge_usage(
        self,
        runtime_gate: Any,
        frame_id: int,
        bridge_meta: dict[str, Any],
        pose_debug: dict[str, Any],
        before_valid: int,
        before_pnp: int,
        before_miniba: int,
    ) -> None:
        runtime_gate.append_anchor_transition_bridge_event(
            {
                "frame_id": int(frame_id),
                "anchor_id": int(bridge_meta.get("anchor_id", -1)),
                "bridged_reference_ids": list(bridge_meta.get("bridged_reference_ids", []) or []),
                "bridge_source_anchor": "|".join(
                    str(x) for x in bridge_meta.get("bridge_source_anchors", []) or []
                ),
                "bridge_reason": str(bridge_meta.get("bridge_reason", "")),
                "match_count_before": int(pose_debug.get("match_count_total", 0) or 0),
                "match_count_after": int(pose_debug.get("match_count_total", 0) or 0),
                "valid_2d3d_before": int(before_valid),
                "valid_2d3d_after": int(pose_debug.get("num_2d3d_correspondences", 0) or 0),
                "pnp_inliers_before": int(before_pnp),
                "pnp_inliers_after": int(pose_debug.get("num_pnp_inliers", 0) or 0),
                "miniba_inliers_before": int(before_miniba),
                "miniba_inliers_after": int(pose_debug.get("num_miniba_inliers", 0) or 0),
            }
        )
