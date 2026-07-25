#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def summarize_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = [event for event in payload.get("events", []) if isinstance(event, dict)]
    photometric = [
        event
        for event in events
        if isinstance(event.get("photometric_verification"), dict)
    ]
    attempted = [
        event for event in photometric if event.get("photometric_verification_attempted")
    ]
    accepted = [
        event for event in attempted if event.get("photometric_verification_accepted")
    ]

    def values(key: str) -> np.ndarray:
        return np.asarray(
            [
                float(event["photometric_verification"][key])
                for event in attempted
                if key in event["photometric_verification"]
            ],
            dtype=np.float64,
        )

    def stats(key: str) -> dict[str, float]:
        selected = values(key)
        if selected.size == 0:
            return {"mean": 0.0, "median": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(selected)),
            "median": float(np.median(selected)),
            "max": float(np.max(selected)),
        }

    relative_gain = []
    for event in attempted:
        review = event["photometric_verification"]
        start = float(review.get("start_loss", 0.0) or 0.0)
        end = float(review.get("end_loss", start) or start)
        if start > 0:
            relative_gain.append((start - end) / start)
    return {
        "events": len(events),
        "geometric_attempts": sum(
            bool(event.get("verification_attempted", False)) for event in events
        ),
        "geometric_accepts": sum(
            bool(event.get("verification_accepted", False)) for event in events
        ),
        "photometric_requested": len(photometric),
        "photometric_attempts": len(attempted),
        "photometric_accepts": len(accepted),
        "photometric_test_attempts": sum(
            bool(event.get("a_stage_is_test", False)) for event in attempted
        ),
        "photometric_test_accepts": sum(
            bool(event.get("a_stage_is_test", False)) for event in accepted
        ),
        "photometric_accept_rate": len(accepted) / max(len(attempted), 1),
        "reasons": dict(
            Counter(
                str(event.get("photometric_verification_reason", ""))
                for event in photometric
            )
        ),
        "relative_gain": {
            "mean": float(np.mean(relative_gain)) if relative_gain else 0.0,
            "median": float(np.median(relative_gain)) if relative_gain else 0.0,
            "max": float(np.max(relative_gain)) if relative_gain else 0.0,
        },
        "rotation_delta_deg": stats("rotation_delta_deg"),
        "translation_delta": stats("translation_delta"),
        "validation_support_ratio": stats("validation_support_ratio"),
        "post_a_pose_count": sum(bool(event.get("post_a_Rt")) for event in events),
        "final_pose_count": sum(
            bool(event.get("photometric_verification", {}).get("final_Rt"))
            for event in events
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_trace(args.trace), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
