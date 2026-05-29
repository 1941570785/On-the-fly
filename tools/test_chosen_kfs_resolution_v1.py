#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scene.keyframe import resolve_chosen_keyframes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    scene = [SimpleNamespace(index=i, info={"image_name": f"kf_{i:03d}"}) for i in range(3)]

    # Reproduce old equivalent failure: direct list indexing with invalid chosen id.
    reproduced_previous_failure = False
    try:
        _ = scene[7]
    except IndexError:
        reproduced_previous_failure = True

    # New behavior: explicit resolution in paper_aligned path.
    (
        resolved_indices,
        invalid_ids,
        chosen_id_types,
        fallback_used,
        row_to_resolved,
    ) = resolve_chosen_keyframes([7], scene, mode="paper_aligned_true_recovery")

    index_error_count = 0
    try:
        _ = [scene[idx] for idx in resolved_indices]
    except IndexError:
        index_error_count += 1

    invalid_ids_logged = len(invalid_ids) > 0
    fallback_behavior_verified = bool(fallback_used and len(resolved_indices) > 0)
    passed = (
        reproduced_previous_failure
        and index_error_count == 0
        and invalid_ids_logged
        and fallback_behavior_verified
    )

    result = {
        "unit_test_returncode": 0 if passed else 1,
        "reproduced_previous_failure": bool(reproduced_previous_failure),
        "simulated_equivalent_failure": bool(reproduced_previous_failure),
        "index_error_count": int(index_error_count),
        "invalid_ids_logged": bool(invalid_ids_logged),
        "fallback_behavior_verified": bool(fallback_behavior_verified),
        "resolved_keyframe_list_indices": [int(x) for x in resolved_indices],
        "chosen_kfs_id_types": chosen_id_types,
        "row_to_resolved": [(int(a), int(b)) for a, b in row_to_resolved],
        "invalid_chosen_kfs_ids": invalid_ids,
    }
    (out / "chosen_kfs_resolution_unit_test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "chosen_kfs_resolution_unit_test_report.md").write_text(
        "\n".join(
            [
                "# chosen_kfs resolution unit test",
                "",
                f"- unit_test_returncode: {result['unit_test_returncode']}",
                f"- reproduced_previous_failure: {result['reproduced_previous_failure']}",
                f"- index_error_count: {result['index_error_count']}",
                f"- invalid_ids_logged: {result['invalid_ids_logged']}",
                f"- fallback_behavior_verified: {result['fallback_behavior_verified']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
