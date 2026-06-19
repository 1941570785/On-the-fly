from __future__ import annotations


def baseline_eval_metadata(
    *,
    sequence_index: int,
    image_name: str,
    test_hold: int,
    start_at: int = 0,
) -> dict[str, object]:
    sequence_index = int(sequence_index)
    test_hold = int(test_hold)
    start_at = int(start_at)
    is_eval = bool(test_hold > 0 and sequence_index % test_hold == 0)
    return {
        "is_test": is_eval,
        "name": image_name,
        "image_name": image_name,
        "_baseline_eval_frame": is_eval,
        "_baseline_eval_hold": test_hold,
        "_baseline_eval_sequence_index": sequence_index if is_eval else -1,
        "_baseline_eval_original_index": sequence_index + start_at if is_eval else -1,
    }
