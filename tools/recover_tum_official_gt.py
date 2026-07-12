#!/usr/bin/env python3
"""Recover frame-aligned official TUM RGB-D ground-truth poses.

The baseline dataset renames rectified RGB frames to 1.png, 2.png, ... and
does not keep the original TUM timestamps. This tool restores the timestamp
mapping from the official archives, verifies every processed image against
the corresponding official RGB frame, interpolates the 100 Hz mocap poses at
the RGB timestamps, and writes auditable pose data under sparse/GT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SceneSpec:
    scene: str
    sequence: str
    archive: str
    archive_url: str
    groundtruth_url: str
    camera_params: tuple[float, ...]


SCENES = {
    "desk1": SceneSpec(
        scene="desk1",
        sequence="rgbd_dataset_freiburg1_desk",
        archive="rgbd_dataset_freiburg1_desk.tgz",
        archive_url=(
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
            "rgbd_dataset_freiburg1_desk.tgz"
        ),
        groundtruth_url=(
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
            "rgbd_dataset_freiburg1_desk-groundtruth.txt"
        ),
        camera_params=(
            517.306408,
            516.469215,
            318.643040,
            255.313989,
            0.262383,
            -0.953104,
            -0.005358,
            0.002628,
            1.163314,
        ),
    ),
    "desk2": SceneSpec(
        scene="desk2",
        sequence="rgbd_dataset_freiburg2_xyz",
        archive="rgbd_dataset_freiburg2_xyz.tgz",
        archive_url=(
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/"
            "rgbd_dataset_freiburg2_xyz.tgz"
        ),
        groundtruth_url=(
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/"
            "rgbd_dataset_freiburg2_xyz-groundtruth.txt"
        ),
        camera_params=(
            520.908620,
            521.007327,
            325.141442,
            249.701764,
            0.231222,
            -0.784899,
            -0.003257,
            -0.000105,
            0.917205,
        ),
    ),
    "long_office_household": SceneSpec(
        scene="long_office_household",
        sequence="rgbd_dataset_freiburg3_long_office_household",
        archive="rgbd_dataset_freiburg3_long_office_household.tgz",
        archive_url=(
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/"
            "rgbd_dataset_freiburg3_long_office_household.tgz"
        ),
        groundtruth_url=(
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/"
            "rgbd_dataset_freiburg3_long_office_household-groundtruth.txt"
        ),
        camera_params=(535.4, 539.2, 320.1, 247.6, 0.0, 0.0, 0.0, 0.0),
    ),
}


def _data_lines(text: str) -> list[list[str]]:
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line.split())
    return rows


def parse_rgb_list(text: str) -> tuple[np.ndarray, list[str]]:
    rows = _data_lines(text)
    if any(len(row) != 2 for row in rows):
        raise ValueError("TUM rgb.txt must contain timestamp/path pairs")
    timestamps = np.asarray([float(row[0]) for row in rows], dtype=np.float64)
    paths = [row[1] for row in rows]
    _assert_strictly_increasing(timestamps, "RGB timestamps")
    if len(set(paths)) != len(paths):
        raise ValueError("TUM rgb.txt contains duplicate image paths")
    return timestamps, paths


def parse_groundtruth(
    text: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _data_lines(text)
    if any(len(row) != 8 for row in rows):
        raise ValueError("TUM groundtruth.txt must contain 8 columns")
    # The official TUM associate.py loader constructs dict(list), so the last
    # record wins when rounded mocap timestamps are duplicated. Match that
    # behavior, then sort for deterministic interpolation.
    by_timestamp: dict[float, list[float]] = {}
    for row in rows:
        values = [float(value) for value in row]
        by_timestamp[values[0]] = values
    values = np.asarray([by_timestamp[key] for key in sorted(by_timestamp)])
    timestamps = values[:, 0]
    translations = values[:, 1:4]
    quaternions_xyzw = values[:, 4:8]
    _assert_strictly_increasing(timestamps, "ground-truth timestamps")
    norms = np.linalg.norm(quaternions_xyzw, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("groundtruth.txt contains a zero quaternion")
    quaternions_xyzw = quaternions_xyzw / norms[:, None]
    return timestamps, translations, quaternions_xyzw


def _assert_strictly_increasing(values: np.ndarray, label: str) -> None:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{label} must be a non-empty vector")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{label} are not strictly increasing")


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - alpha) * theta) / sin_theta
    w1 = math.sin(alpha * theta) / sin_theta
    result = w0 * q0 + w1 * q1
    return result / np.linalg.norm(result)


def quaternion_xyzw_to_rotation(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r[2, 1] - r[1, 2]) / scale
        qy = (r[0, 2] - r[2, 0]) / scale
        qz = (r[1, 0] - r[0, 1]) / scale
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        scale = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / scale
        qx = 0.25 * scale
        qy = (r[0, 1] + r[1, 0]) / scale
        qz = (r[0, 2] + r[2, 0]) / scale
    elif r[1, 1] > r[2, 2]:
        scale = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / scale
        qx = (r[0, 1] + r[1, 0]) / scale
        qy = 0.25 * scale
        qz = (r[1, 2] + r[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / scale
        qx = (r[0, 2] + r[2, 0]) / scale
        qy = (r[1, 2] + r[2, 1]) / scale
        qz = 0.25 * scale
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def interpolate_groundtruth(
    rgb_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_translations: np.ndarray,
    gt_quaternions_xyzw: np.ndarray,
    max_bracket_gap_s: float,
) -> dict[str, np.ndarray]:
    count = len(rgb_timestamps)
    translations = np.full((count, 3), np.nan, dtype=np.float64)
    quaternions = np.full((count, 4), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    left_times = np.full(count, np.nan, dtype=np.float64)
    right_times = np.full(count, np.nan, dtype=np.float64)
    alphas = np.full(count, np.nan, dtype=np.float64)
    nearest_gaps = np.full(count, np.nan, dtype=np.float64)
    bracket_gaps = np.full(count, np.nan, dtype=np.float64)

    for index, timestamp in enumerate(rgb_timestamps):
        right = int(np.searchsorted(gt_timestamps, timestamp, side="left"))
        if right < len(gt_timestamps) and abs(gt_timestamps[right] - timestamp) < 1e-9:
            translations[index] = gt_translations[right]
            quaternions[index] = gt_quaternions_xyzw[right]
            valid[index] = True
            left_times[index] = gt_timestamps[right]
            right_times[index] = gt_timestamps[right]
            alphas[index] = 0.0
            nearest_gaps[index] = 0.0
            bracket_gaps[index] = 0.0
            continue
        if right == 0 or right == len(gt_timestamps):
            continue
        left = right - 1
        t0 = float(gt_timestamps[left])
        t1 = float(gt_timestamps[right])
        bracket_gap = t1 - t0
        nearest_gap = min(timestamp - t0, t1 - timestamp)
        left_times[index] = t0
        right_times[index] = t1
        nearest_gaps[index] = nearest_gap
        bracket_gaps[index] = bracket_gap
        # TUM decimal timestamps can turn an exact 50 ms interval into
        # 0.05000019 after float conversion. Keep a one-microsecond tolerance.
        if bracket_gap <= 0.0 or bracket_gap > max_bracket_gap_s + 1e-6:
            continue
        alpha = float((timestamp - t0) / bracket_gap)
        if alpha < 0.0 or alpha > 1.0:
            continue
        translations[index] = (
            (1.0 - alpha) * gt_translations[left] + alpha * gt_translations[right]
        )
        quaternions[index] = slerp_xyzw(
            gt_quaternions_xyzw[left], gt_quaternions_xyzw[right], alpha
        )
        valid[index] = True
        alphas[index] = alpha

    return {
        "translations": translations,
        "quaternions_xyzw": quaternions,
        "valid": valid,
        "left_times": left_times,
        "right_times": right_times,
        "alphas": alphas,
        "nearest_gaps": nearest_gaps,
        "bracket_gaps": bracket_gaps,
    }


def build_pose_arrays(
    translations: np.ndarray,
    quaternions_xyzw: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(valid)
    c2w = np.full((count, 4, 4), np.nan, dtype=np.float64)
    w2c = np.full((count, 4, 4), np.nan, dtype=np.float64)
    for index in np.flatnonzero(valid):
        rotation = quaternion_xyzw_to_rotation(quaternions_xyzw[index])
        c2w[index] = np.eye(4, dtype=np.float64)
        c2w[index, :3, :3] = rotation
        c2w[index, :3, 3] = translations[index]
        w2c[index] = np.eye(4, dtype=np.float64)
        w2c[index, :3, :3] = rotation.T
        w2c[index, :3, 3] = -rotation.T @ translations[index]
    return c2w, w2c


def numeric_image_names(image_dir: Path) -> list[str]:
    names = [path.name for path in image_dir.iterdir() if path.suffix.lower() == ".png"]
    try:
        names.sort(key=lambda name: int(Path(name).stem))
    except ValueError as exc:
        raise ValueError(f"Non-numeric image name in {image_dir}") from exc
    expected = [f"{index}.png" for index in range(1, len(names) + 1)]
    if names != expected:
        raise ValueError(f"Image names in {image_dir} are not continuous from 1.png")
    return names


def archive_member_text(archive: Path, member_name: str) -> str:
    with tarfile.open(archive, "r:gz") as tar:
        handle = tar.extractfile(member_name)
        if handle is None:
            raise FileNotFoundError(f"{member_name} not found in {archive}")
        return handle.read().decode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def effective_rectified_k(
    spec: SceneSpec, width: int = 640, height: int = 480
) -> np.ndarray:
    import cv2

    params = np.asarray(spec.camera_params, dtype=np.float64)
    k_in = np.asarray(
        [[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    k_out = cv2.getOptimalNewCameraMatrix(
        k_in, params[4:], (width, height), 1, (width, height), True
    )[0]
    square_focal = float((k_out[0, 0] + k_out[1, 1]) / 2.0)
    k_out[0, 0] = square_focal
    k_out[1, 1] = square_focal
    return k_out


def verify_processed_images(
    spec: SceneSpec,
    archive: Path,
    source_paths: list[str],
    image_dir: Path,
) -> dict[str, object]:
    import cv2

    params = np.asarray(spec.camera_params, dtype=np.float64)
    sample = cv2.imread(str(image_dir / "1.png"), cv2.IMREAD_UNCHANGED)
    if sample is None or sample.shape != (480, 640, 4):
        raise ValueError(f"Expected 640x480 RGBA images in {image_dir}")
    height, width = sample.shape[:2]
    k_in = np.asarray(
        [[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    k_out = effective_rectified_k(spec, width=width, height=height)
    rectify_map = cv2.initUndistortRectifyMap(
        k_in, params[4:], None, k_out, (width, height), cv2.CV_32FC2
    )[0]
    initial_mask = np.full((height, width), 255, dtype=np.uint8)
    expected_mask = cv2.remap(initial_mask, rectify_map, None, cv2.INTER_LINEAR)
    expected_mask[expected_mask <= 0] = 0
    expected_mask[expected_mask != 0] = 255

    member_to_index = {
        f"{spec.sequence}/{source_path}": index
        for index, source_path in enumerate(source_paths)
    }
    verified = np.zeros(len(source_paths), dtype=bool)
    mismatches: list[dict[str, object]] = []

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            index = member_to_index.get(member.name)
            if index is None:
                continue
            source_handle = tar.extractfile(member)
            if source_handle is None:
                raise RuntimeError(f"Cannot extract {member.name}")
            raw_bytes = np.frombuffer(source_handle.read(), dtype=np.uint8)
            raw_image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            if raw_image is None:
                raise RuntimeError(f"Cannot decode {member.name}")
            rectified = cv2.remap(
                raw_image,
                rectify_map,
                None,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            rectified[expected_mask == 0] = 0
            expected = np.concatenate([rectified, expected_mask[..., None]], axis=-1)
            image_name = f"{index + 1}.png"
            current = cv2.imread(str(image_dir / image_name), cv2.IMREAD_UNCHANGED)
            if current is None or current.shape != expected.shape:
                mismatches.append({"image_name": image_name, "reason": "shape_or_read"})
            elif not np.array_equal(current, expected):
                difference = np.abs(current.astype(np.int16) - expected.astype(np.int16))
                mismatches.append(
                    {
                        "image_name": image_name,
                        "reason": "pixel_mismatch",
                        "max_abs_difference": int(difference.max()),
                        "mean_abs_difference": float(difference.mean()),
                    }
                )
            verified[index] = True
            if bool(np.all(verified)):
                break

    missing_members = [
        f"{index + 1}.png" for index in np.flatnonzero(~verified).tolist()
    ]
    if missing_members or mismatches:
        preview = {"missing": missing_members[:10], "mismatches": mismatches[:10]}
        raise RuntimeError(f"Full image-order verification failed: {preview}")
    return {
        "mode": "full_pixel_exact",
        "verified_images": int(verified.sum()),
        "mismatched_images": 0,
        "effective_rectified_K": k_out.tolist(),
    }


def write_colmap_text_model(
    output_dir: Path,
    camera_matrix: np.ndarray,
    width: int,
    height: int,
    image_names: list[str],
    w2c: np.ndarray,
    valid: np.ndarray,
) -> None:
    (output_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 SIMPLE_PINHOLE {width} {height} "
        f"{camera_matrix[0, 0]:.17g} {camera_matrix[0, 2]:.17g} "
        f"{camera_matrix[1, 2]:.17g}\n",
        encoding="utf-8",
    )

    valid_count = int(valid.sum())
    with (output_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {valid_count}, mean observations per image: 0\n")
        for index in np.flatnonzero(valid):
            rotation = w2c[index, :3, :3]
            translation = w2c[index, :3, 3]
            qvec = rotation_to_quaternion_wxyz(rotation)
            values = [
                str(index + 1),
                *(f"{value:.17g}" for value in qvec),
                *(f"{value:.17g}" for value in translation),
                "1",
                image_names[index],
            ]
            handle.write(" ".join(values) + "\n\n")

    (output_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )


def write_frame_map(
    output_dir: Path,
    image_names: list[str],
    source_paths: list[str],
    rgb_timestamps: np.ndarray,
    interpolation: dict[str, np.ndarray],
) -> None:
    translations = interpolation["translations"]
    quaternions = interpolation["quaternions_xyzw"]
    valid = interpolation["valid"]
    fields = [
        "frame_index",
        "image_name",
        "rgb_timestamp",
        "source_rgb_path",
        "gt_valid",
        "gt_left_timestamp",
        "gt_right_timestamp",
        "interpolation_alpha",
        "nearest_gt_gap_s",
        "bracket_gt_gap_s",
        "tx",
        "ty",
        "tz",
        "qx",
        "qy",
        "qz",
        "qw",
    ]
    with (output_dir / "frame_map.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, image_name in enumerate(image_names):
            pose_values: Iterable[float | str]
            if valid[index]:
                pose_values = [*translations[index], *quaternions[index]]
            else:
                pose_values = [""] * 7
            tx, ty, tz, qx, qy, qz, qw = pose_values
            writer.writerow(
                {
                    "frame_index": index + 1,
                    "image_name": image_name,
                    "rgb_timestamp": f"{rgb_timestamps[index]:.6f}",
                    "source_rgb_path": source_paths[index],
                    "gt_valid": int(valid[index]),
                    "gt_left_timestamp": _optional_float(interpolation["left_times"][index]),
                    "gt_right_timestamp": _optional_float(interpolation["right_times"][index]),
                    "interpolation_alpha": _optional_float(interpolation["alphas"][index]),
                    "nearest_gt_gap_s": _optional_float(interpolation["nearest_gaps"][index]),
                    "bracket_gt_gap_s": _optional_float(interpolation["bracket_gaps"][index]),
                    "tx": _optional_float(tx),
                    "ty": _optional_float(ty),
                    "tz": _optional_float(tz),
                    "qx": _optional_float(qx),
                    "qy": _optional_float(qy),
                    "qz": _optional_float(qz),
                    "qw": _optional_float(qw),
                }
            )


def _optional_float(value: object) -> str:
    if value == "" or value is None:
        return ""
    number = float(value)
    return "" if not np.isfinite(number) else f"{number:.17g}"


def write_tum_trajectory(
    output_dir: Path,
    rgb_timestamps: np.ndarray,
    interpolation: dict[str, np.ndarray],
) -> None:
    translations = interpolation["translations"]
    quaternions = interpolation["quaternions_xyzw"]
    valid = interpolation["valid"]
    with (output_dir / "trajectory_c2w.txt").open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for index in np.flatnonzero(valid):
            values = [
                rgb_timestamps[index],
                *translations[index],
                *quaternions[index],
            ]
            handle.write(" ".join(f"{value:.17g}" for value in values) + "\n")


def validate_artifacts(
    output_dir: Path,
    image_names: list[str],
    expected_valid: np.ndarray,
) -> dict[str, object]:
    count = len(image_names)
    timestamps = np.load(output_dir / "timestamps.npy")
    valid = np.load(output_dir / "valid_mask.npy")
    c2w = np.load(output_dir / "poses_c2w.npy")
    w2c = np.load(output_dir / "poses_w2c.npy")
    if timestamps.shape != (count,):
        raise RuntimeError("timestamps.npy shape mismatch")
    if valid.shape != (count,) or not np.array_equal(valid, expected_valid):
        raise RuntimeError("valid_mask.npy mismatch")
    if c2w.shape != (count, 4, 4) or w2c.shape != (count, 4, 4):
        raise RuntimeError("pose array shape mismatch")
    if not np.isfinite(c2w[valid]).all() or not np.isfinite(w2c[valid]).all():
        raise RuntimeError("valid pose arrays contain non-finite values")
    if (~valid).any() and (
        not np.isnan(c2w[~valid]).all() or not np.isnan(w2c[~valid]).all()
    ):
        raise RuntimeError("invalid pose entries must be NaN")
    products = c2w[valid] @ w2c[valid]
    inverse_error = float(np.max(np.abs(products - np.eye(4))))
    if inverse_error > 1e-10:
        raise RuntimeError(f"c2w/w2c inverse error is {inverse_error}")

    with (output_dir / "frame_map.csv").open(encoding="utf-8", newline="") as handle:
        frame_rows = list(csv.DictReader(handle))
    if len(frame_rows) != count:
        raise RuntimeError("frame_map.csv row count mismatch")
    for index, row in enumerate(frame_rows):
        if int(row["frame_index"]) != index + 1 or row["image_name"] != image_names[index]:
            raise RuntimeError("frame_map.csv ordering mismatch")
        if bool(int(row["gt_valid"])) != bool(valid[index]):
            raise RuntimeError("frame_map.csv validity mismatch")

    image_lines = [
        line.split()
        for line in (output_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    valid_indices = np.flatnonzero(valid)
    if len(image_lines) != len(valid_indices):
        raise RuntimeError("COLMAP images.txt count mismatch")
    for values, index in zip(image_lines, valid_indices):
        if int(values[0]) != index + 1 or values[9] != image_names[index]:
            raise RuntimeError("COLMAP images.txt ordering mismatch")

    trajectory_rows = _data_lines(
        (output_dir / "trajectory_c2w.txt").read_text(encoding="utf-8")
    )
    if len(trajectory_rows) != int(valid.sum()):
        raise RuntimeError("trajectory_c2w.txt count mismatch")
    source_rgb_count = len(
        _data_lines((output_dir / "source_rgb.txt").read_text(encoding="utf-8"))
    )
    if source_rgb_count != count:
        raise RuntimeError("source_rgb.txt count mismatch")
    return {
        "frame_map_rows": len(frame_rows),
        "pose_array_shape": list(c2w.shape),
        "colmap_valid_images": len(image_lines),
        "trajectory_valid_poses": len(trajectory_rows),
        "max_c2w_w2c_inverse_error": inverse_error,
    }


def recover_scene(
    repo_root: Path,
    source_root: Path,
    spec: SceneSpec,
    max_bracket_gap_s: float,
    verify_images: bool,
    overwrite: bool,
) -> dict[str, object]:
    scene_dir = repo_root / "datasets" / "TUM" / spec.scene
    image_dir = scene_dir / "images"
    archive = source_root / spec.archive
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing {archive}. Download it from {spec.archive_url}"
        )
    image_names = numeric_image_names(image_dir)
    rgb_text = archive_member_text(archive, f"{spec.sequence}/rgb.txt")
    groundtruth_text = archive_member_text(
        archive, f"{spec.sequence}/groundtruth.txt"
    )
    rgb_timestamps, source_paths = parse_rgb_list(rgb_text)
    if len(image_names) != len(rgb_timestamps):
        raise ValueError(
            f"{spec.scene}: {len(image_names)} processed images but "
            f"{len(rgb_timestamps)} official RGB records"
        )

    verification = (
        verify_processed_images(spec, archive, source_paths, image_dir)
        if verify_images
        else {
            "mode": "disabled",
            "verified_images": 0,
            "mismatched_images": None,
            "effective_rectified_K": effective_rectified_k(spec).tolist(),
        }
    )
    gt_timestamps, gt_translations, gt_quaternions = parse_groundtruth(
        groundtruth_text
    )
    source_gt_record_count = len(_data_lines(groundtruth_text))
    duplicate_gt_timestamp_count = source_gt_record_count - len(gt_timestamps)
    interpolation = interpolate_groundtruth(
        rgb_timestamps,
        gt_timestamps,
        gt_translations,
        gt_quaternions,
        max_bracket_gap_s=max_bracket_gap_s,
    )
    c2w, w2c = build_pose_arrays(
        interpolation["translations"],
        interpolation["quaternions_xyzw"],
        interpolation["valid"],
    )

    target = scene_dir / "sparse" / "GT"
    temporary = scene_dir / "sparse" / f".GT.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        (temporary / "source_rgb.txt").write_text(rgb_text, encoding="utf-8")
        (temporary / "source_groundtruth.txt").write_text(
            groundtruth_text, encoding="utf-8"
        )
        np.save(temporary / "timestamps.npy", rgb_timestamps)
        np.save(temporary / "valid_mask.npy", interpolation["valid"])
        np.save(temporary / "poses_c2w.npy", c2w)
        np.save(temporary / "poses_w2c.npy", w2c)
        write_frame_map(
            temporary,
            image_names,
            source_paths,
            rgb_timestamps,
            interpolation,
        )
        write_tum_trajectory(temporary, rgb_timestamps, interpolation)
        write_colmap_text_model(
            temporary,
            np.asarray(verification["effective_rectified_K"], dtype=np.float64),
            640,
            480,
            image_names,
            w2c,
            interpolation["valid"],
        )
        artifact_validation = validate_artifacts(
            temporary, image_names, interpolation["valid"]
        )
        valid = interpolation["valid"]
        metadata = {
            "schema_version": 1,
            "scene": spec.scene,
            "official_sequence": spec.sequence,
            "archive_url": spec.archive_url,
            "groundtruth_url": spec.groundtruth_url,
            "archive_file": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "image_count": len(image_names),
            "source_groundtruth_record_count": source_gt_record_count,
            "duplicate_groundtruth_timestamp_count": duplicate_gt_timestamp_count,
            "duplicate_groundtruth_policy": "last_record_wins_official_associate_py",
            "pose_valid_count": int(valid.sum()),
            "pose_invalid_count": int((~valid).sum()),
            "pose_coverage": float(valid.mean()),
            "first_valid_frame": int(np.flatnonzero(valid)[0] + 1),
            "last_valid_frame": int(np.flatnonzero(valid)[-1] + 1),
            "max_bracket_gap_s": max_bracket_gap_s,
            "interpolation": {
                "translation": "linear",
                "rotation": "quaternion_slerp_shortest_path",
                "out_of_range": "invalid_no_extrapolation",
            },
            "coordinate_conventions": {
                "poses_c2w.npy": "T_world_from_camera",
                "poses_w2c.npy": "T_camera_from_world",
                "trajectory_c2w.txt": "TUM timestamp tx ty tz qx qy qz qw",
                "images.txt": "COLMAP world-to-camera qvec(wxyz), tvec",
            },
            "frame_mapping": (
                "Processed i.png is mapped to the i-th chronological RGB record. "
                "The mapping is accepted only after image verification."
            ),
            "verification": verification,
            "artifact_validation": artifact_validation,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.txt").write_text(
            "Official TUM RGB-D ground-truth poses aligned to images/1.png..N.png.\n"
            "\n"
            "poses_c2w.npy and poses_w2c.npy have shape [N,4,4]. Invalid entries\n"
            "are NaN and are identified by valid_mask.npy and frame_map.csv. No\n"
            "pose is extrapolated outside the official mocap interval. images.txt\n"
            "contains only valid poses in COLMAP text format while preserving the\n"
            "original 1-based frame id and image name. See metadata.json for the\n"
            "source URLs, coordinate conventions, interpolation, and verification.\n",
            encoding="utf-8",
        )

        if target.exists():
            if not overwrite:
                raise FileExistsError(f"{target} already exists; use --overwrite")
            backup = target.with_name(f"GT.backup-{os.getpid()}")
            target.rename(backup)
            temporary.rename(target)
            shutil.rmtree(backup)
        else:
            temporary.rename(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Directory containing the three official .tgz archives",
    )
    parser.add_argument(
        "--scene",
        action="append",
        choices=sorted(SCENES),
        help="Scene to recover; repeat as needed. Defaults to all three.",
    )
    parser.add_argument("--max-bracket-gap-s", type=float, default=0.05)
    parser.add_argument("--skip-image-verification", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_root = (
        args.source_root.resolve()
        if args.source_root is not None
        else repo_root / ".cache" / "tum_official"
    )
    scenes = args.scene or list(SCENES)
    summaries = []
    for scene in scenes:
        print(f"Recovering {scene}...", flush=True)
        metadata = recover_scene(
            repo_root=repo_root,
            source_root=source_root,
            spec=SCENES[scene],
            max_bracket_gap_s=args.max_bracket_gap_s,
            verify_images=not args.skip_image_verification,
            overwrite=args.overwrite,
        )
        summaries.append(metadata)
        print(
            f"  {metadata['pose_valid_count']}/{metadata['image_count']} valid poses; "
            f"verification={metadata['verification']['mode']}",
            flush=True,
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
