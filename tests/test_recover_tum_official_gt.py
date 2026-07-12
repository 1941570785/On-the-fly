from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from recover_tum_official_gt import (  # noqa: E402
    build_pose_arrays,
    interpolate_groundtruth,
    parse_groundtruth,
    parse_rgb_list,
    quaternion_xyzw_to_rotation,
    rotation_to_quaternion_wxyz,
    slerp_xyzw,
)


def test_parse_tum_files_ignores_comments():
    rgb_ts, paths = parse_rgb_list("# header\n1.0 rgb/a.png\n2.0 rgb/b.png\n")
    gt_ts, translations, quaternions = parse_groundtruth(
        "# header\n1.0 1 2 3 0 0 0 1\n2.0 2 3 4 0 0 0 1\n"
    )
    np.testing.assert_allclose(rgb_ts, [1.0, 2.0])
    assert paths == ["rgb/a.png", "rgb/b.png"]
    np.testing.assert_allclose(gt_ts, [1.0, 2.0])
    np.testing.assert_allclose(translations[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(quaternions[0], [0.0, 0.0, 0.0, 1.0])


def test_groundtruth_duplicate_timestamp_uses_last_record():
    timestamps, translations, quaternions = parse_groundtruth(
        "1.0 1 2 3 0 0 0 1\n"
        "1.0 4 5 6 0 0 0 1\n"
        "2.0 7 8 9 0 0 0 1\n"
    )
    np.testing.assert_allclose(timestamps, [1.0, 2.0])
    np.testing.assert_allclose(translations[0], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(quaternions[0], [0.0, 0.0, 0.0, 1.0])


def test_slerp_uses_shortest_path_and_normalizes():
    q = slerp_xyzw(
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, 0.0, -1.0]),
        0.5,
    )
    np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_interpolation_marks_out_of_range_and_large_gaps_invalid():
    result = interpolate_groundtruth(
        rgb_timestamps=np.asarray([-0.1, 0.0, 0.5, 1.0, 1.1, 3.0]),
        gt_timestamps=np.asarray([0.0, 1.0, 3.0]),
        gt_translations=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        gt_quaternions_xyzw=np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        ),
        max_bracket_gap_s=1.5,
    )
    assert result["valid"].tolist() == [False, True, True, True, False, True]
    np.testing.assert_allclose(result["translations"][2], [0.5, 0.0, 0.0])


def test_interpolation_tolerates_decimal_timestamp_roundoff():
    result = interpolate_groundtruth(
        rgb_timestamps=np.asarray([0.025]),
        gt_timestamps=np.asarray([0.0, 0.0500002]),
        gt_translations=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        gt_quaternions_xyzw=np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        ),
        max_bracket_gap_s=0.05,
    )
    assert result["valid"].tolist() == [True]


def test_pose_arrays_are_exact_inverses():
    angle = np.deg2rad(35.0)
    q_xyzw = np.asarray([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])
    c2w, w2c = build_pose_arrays(
        np.asarray([[1.0, 2.0, 3.0]]),
        q_xyzw[None],
        np.asarray([True]),
    )
    np.testing.assert_allclose(c2w[0] @ w2c[0], np.eye(4), atol=1e-12)
    rotation = quaternion_xyzw_to_rotation(q_xyzw)
    q_wxyz = rotation_to_quaternion_wxyz(rotation)
    np.testing.assert_allclose(q_wxyz, [q_xyzw[3], *q_xyzw[:3]], atol=1e-12)
