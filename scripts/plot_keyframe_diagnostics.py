import argparse
import csv
import math
import os


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot arbitrary per-keyframe diagnostics from keyframe_metrics.csv or "
            "diagnostics/keyframe_diagnostics.csv."
        )
    )
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[
            "psnr",
            "pnp_inlier_ratio",
            "miniba_final_residual",
            "depth_align_error_mean",
            "spawn_uncertainty_mean",
        ],
    )
    p.add_argument("--title", type=str, default="")
    p.add_argument("--only", choices=["all", "train", "test"], default="all")
    return p.parse_args()


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ["1", "true", "yes", "y", "t"]


def _read_rows(path: str, only: str):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_test = _as_bool(row.get("is_test", "false"))
            if only == "test" and not is_test:
                continue
            if only == "train" and is_test:
                continue
            rows.append(row)
    return rows


def _to_float(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _fmt(x):
    if x != x or math.isinf(x):
        return "nan"
    return f"{x:.3f}"


def _plot_metric(out_path, title, xs, ys, metric_name):
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency: pillow\nInstall with `pip install pillow`.") from e

    valid = [(x, y) for x, y in zip(xs, ys) if y == y and not math.isinf(y)]
    width, height = 1500, 700
    pad_l, pad_r, pad_t, pad_b = 120, 40, 80, 90
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((pad_l, 24), f"{title} [{metric_name}]" if title else metric_name, fill=(17, 17, 17), font=font)
    if not valid:
        draw.text((pad_l, pad_t), "No valid values", fill=(150, 0, 0), font=font)
        img.save(out_path, quality=95)
        return

    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        x1 = x0 + 1
    y_vals = [y for _, y in valid]
    y0, y1 = min(y_vals), max(y_vals)
    if abs(y1 - y0) < 1e-8:
        pad = max(abs(y1) * 0.05, 1e-3)
        y0 -= pad
        y1 += pad
    else:
        pad = 0.05 * (y1 - y0)
        y0 -= pad
        y1 += pad

    def tx(x):
        return pad_l + (x - x0) * plot_w / (x1 - x0)

    def ty(y):
        return pad_t + (y1 - y) * plot_h / (y1 - y0)

    axis_color = (34, 34, 34)
    draw.line([(pad_l, pad_t), (pad_l, pad_t + plot_h)], fill=axis_color, width=2)
    draw.line([(pad_l, pad_t + plot_h), (pad_l + plot_w, pad_t + plot_h)], fill=axis_color, width=2)

    for i in range(6):
        y = y0 + (y1 - y0) * i / 5.0
        yy = ty(y)
        draw.line([(pad_l - 8, yy), (pad_l, yy)], fill=axis_color, width=1)
        draw.text((pad_l - 80, yy - 6), _fmt(y), fill=(17, 17, 17), font=font)
        draw.line([(pad_l, yy), (pad_l + plot_w, yy)], fill=(225, 225, 225), width=1)

    points = [(tx(x), ty(y)) for x, y in valid]
    draw.line(points, fill=(31, 119, 180), width=3)
    for (x, y), (_, value) in zip(points, valid):
        draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=(214, 39, 40))
        draw.text((x - 10, y - 18), _fmt(value), fill=(17, 17, 17), font=font)

    draw.text((width // 2 - 60, height - 40), "关键帧编号", fill=(17, 17, 17), font=font)
    draw.text((20, 24), metric_name, fill=(17, 17, 17), font=font)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, quality=95)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = _read_rows(args.csv_path, args.only)
    xs = list(range(1, len(rows) + 1))
    for metric in args.metrics:
        ys = [_to_float(row, metric) for row in rows]
        _plot_metric(
            os.path.join(args.out_dir, f"{metric}_{args.only}.jpg"),
            args.title,
            xs,
            ys,
            metric,
        )


if __name__ == "__main__":
    main()
