#!/usr/bin/env bash
set -euo pipefail

REPO=/data2/zxd/3D_Reconstruction/On_the_fly_pose_render_coupling_v2
PY=/home/zxd/miniconda3/envs/otf/bin/python
DATA=/data2/zxd/3D_Reconstruction/On_the_fly_padded_datasets
PROFILE=baseline_render_lock_intra_frame_v31
GPU=7
PHASE="${1:-probe3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${2:-$REPO/results/BRANCH_EXPERIMENTS_20260703/v31_original_recovery_${PHASE}_${STAMP}}"

case "$PHASE" in
  probe3)
    SCENES=(
      "bonsai|$DATA/MipNerf360/bonsai|8"
      "desk|$DATA/TUM/rgbd_dataset_freiburg1_desk|30"
      "long_office|$DATA/TUM/rgbd_dataset_freiburg3_long_office_household|30"
    )
    ;;
  full9)
    SCENES=(
      "bonsai|$DATA/MipNerf360/bonsai|8"
      "counter|$DATA/MipNerf360/counter|8"
      "garden|$DATA/MipNerf360/garden|8"
      "forest1|$DATA/StaticHikes/forest1|10"
      "forest2|$DATA/StaticHikes/forest2|10"
      "university2|$DATA/StaticHikes/university2|10"
      "desk|$DATA/TUM/rgbd_dataset_freiburg1_desk|30"
      "xyz|$DATA/TUM/rgbd_dataset_freiburg2_xyz|30"
      "long_office|$DATA/TUM/rgbd_dataset_freiburg3_long_office_household|30"
    )
    ;;
  *)
    echo "Unknown phase: $PHASE (expected probe3 or full9)" >&2
    exit 2
    ;;
esac

cd "$REPO"
mkdir -p "$ROOT"

{
  echo "phase=$PHASE"
  echo "root=$ROOT"
  echo "profile=$PROFILE"
  echo "gpu=$GPU"
  echo "branch=$(git branch --show-current)"
  echo "started_at=$(date --iso-8601=seconds)"
  sha256sum \
    scene/pose_render_extra_optimization.py \
    scene/pose_render_texture_sampling.py \
    scene/scene_model.py \
    paper_aligned_policy/config.py \
    args.py \
    train.py
} | tee "$ROOT/run_manifest.txt"

for item in "${SCENES[@]}"; do
  IFS='|' read -r scene source hold <<< "$item"
  running="$(ps -u "$(id -un)" -o pid=,args= | grep "$PY train.py" | grep -v grep || true)"
  if [[ -n "$running" ]]; then
    echo "Refusing to start $scene because another train.py is running:" >&2
    echo "$running" >&2
    exit 9
  fi

  scene_dir="$ROOT/$scene"
  mkdir -p "$scene_dir"
  echo "START scene=$scene hold=$hold source=$source at=$(date --iso-8601=seconds)"
  start_ts="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
    -s "$source" \
    -m "$scene_dir/model" \
    --eval_poses \
    --test_hold "$hold" \
    --risk_admission_mode on_the_fly_innovation_v1 \
    --paper_aligned_pose_render_assimilation_profile "$PROFILE" \
    > "$scene_dir/train.log" 2>&1
  rc=$?
  set -e
  end_ts="$(date +%s)"
  echo "DONE scene=$scene rc=$rc wall_sec=$((end_ts - start_ts)) at=$(date --iso-8601=seconds)"
  if [[ "$rc" -ne 0 ]]; then
    tail -100 "$scene_dir/train.log" || true
    exit "$rc"
  fi
done

"$PY" - "$ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
rows = []
for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    metadata_path = scene_dir / "model" / "metadata.json"
    data = json.loads(metadata_path.read_text())
    sampling = data.get("pose_render_texture_sampling", {})
    extra = data.get("pose_render_extra_optimization", {})
    rows.append(
        {
            "scene": scene_dir.name,
            "PSNR": data.get("PSNR"),
            "SSIM": data.get("SSIM"),
            "LPIPS": data.get("LPIPS"),
            "time": data.get("time"),
            "num_keyframes": data.get("num keyframes"),
            "sampling_applied": sampling.get("applied"),
            "sampling_events": sampling.get("events"),
            "coverage_deficit_mean": sampling.get("coverage_deficit_mean"),
            "extra_applied": extra.get("applied"),
            "extra_events": extra.get("events"),
            "extra_iterations_mean": extra.get("extra_iterations_mean"),
            "model_dir": str(scene_dir / "model"),
        }
    )

fieldnames = list(rows[0]) if rows else []
with (root / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
(root / "summary.json").write_text(json.dumps(rows, indent=2))
print(f"SUMMARY={root / 'summary.csv'}")
for row in rows:
    print(json.dumps(row, sort_keys=True))
PY

echo "FINISHED root=$ROOT at=$(date --iso-8601=seconds)"
