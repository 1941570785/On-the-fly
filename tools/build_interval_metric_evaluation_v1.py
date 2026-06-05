#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aligned_policy.interval_metrics import (  # noqa: E402
    INTERVAL_FIELDS,
    build_interval_metric_evaluation,
    load_csv_rows,
    render_interval_report,
    write_csv,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_metrics_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_json")
    parser.add_argument("--interval_size", type=int, default=100)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_json(args.baseline_json)
    evaluation = build_interval_metric_evaluation(
        load_csv_rows(args.frame_metrics_csv),
        interval_size=args.interval_size,
        baseline=baseline,
    )

    write_json(output_dir / "interval_metric_summary.json", evaluation)
    write_csv(output_dir / "interval_metric_table.csv", evaluation["interval_rows"], INTERVAL_FIELDS)
    (output_dir / "interval_metric_report.md").write_text(render_interval_report(evaluation), encoding="utf-8")
    return 0


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
