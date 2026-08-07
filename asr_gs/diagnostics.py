from __future__ import annotations

from typing import Iterable, Mapping


def should_track_render_response(
    *,
    sampling_enabled: bool,
    refinement_enabled: bool,
    is_latest_keyframe: bool,
    is_test: bool,
) -> bool:
    return (
        (sampling_enabled or refinement_enabled)
        and is_latest_keyframe
        and not is_test
    )


def summarize_pose_reliability(
    events: Iterable[Mapping[str, object]],
) -> dict[str, int | float]:
    values = list(events)
    attempted = [event for event in values if event.get("review_attempted")]
    accepted = [event for event in values if event.get("review_success")]
    registered = [event for event in values if event.get("registered")]
    attempts = [
        len(event.get("attempts", []))
        for event in values
        if isinstance(event.get("attempts"), list)
    ]
    pnp_inliers = [
        int(event.get("num_pnp_inliers", 0) or 0)
        for event in registered
    ]
    miniba_inliers = [
        int(event.get("num_miniba_inliers", 0) or 0)
        for event in registered
    ]
    return {
        "events": len(values),
        "registered": len(registered),
        "registration_rate": len(registered) / max(len(values), 1),
        "review_attempted": len(attempted),
        "review_rate": len(attempted) / max(len(values), 1),
        "review_accepted": len(accepted),
        "acceptance_rate": len(accepted) / max(len(attempted), 1),
        "pose_hypotheses": sum(attempts),
        "mean_pnp_inliers": sum(pnp_inliers) / max(len(pnp_inliers), 1),
        "mean_miniba_inliers": sum(miniba_inliers)
        / max(len(miniba_inliers), 1),
    }
