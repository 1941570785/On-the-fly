import argparse
import csv
import math
import os


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot per-keyframe PSNR/SSIM/LPIPS from keyframe_metrics.csv.\n"
            "- X axis: keyframe count (1..N)\n"
            "- Y axis ranges: PSNR [10,30], SSIM/LPIPS [0,1]\n"
            "- Every point is marked and annotated with value (2 decimals)\n"
            "- Output: JPG (requires pillow)\n"
        )
    )
    p.add_argument("--metrics_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--title",
        type=str,
        default="",
        help="Plot title (e.g. 'TUM/desk1').",
    )
    p.add_argument(
        "--only",
        choices=["all", "train", "test"],
        default="all",
        help="Filter points by is_test flag.",
    )
    return p.parse_args()


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ["1", "true", "yes", "y", "t"]


def _read_rows(path: str, only: str):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            try:
                is_test = _as_bool(d.get("is_test", "false"))
                if only != "all":
                    if only == "test" and not is_test:
                        continue
                    if only == "train" and is_test:
                        continue
                rows.append(
                    {
                        "keyframe_id": int(float(d.get("keyframe_id", "0"))),
                        "frame_id": int(float(d.get("frame_id", "0"))) if "frame_id" in d else -1,
                        "image_name": d.get("image_name", ""),
                        "is_test": is_test,
                        "psnr": float(d.get("psnr", "nan")),
                        "ssim": float(d.get("ssim", "nan")),
                        "lpips": float(d.get("lpips", "nan")),
                    }
                )
            except Exception:
                continue
    # Keep chronological order by keyframe_id then frame_id
    rows.sort(key=lambda x: (x["keyframe_id"], x["frame_id"]))
    return rows


def _fmt2(x: float) -> str:
    if x != x or math.isinf(x):
        return "nan"
    return f"{x:.2f}"


def _nice_step(n: int, target_ticks: int = 10) -> int:
    if n <= 1:
        return 1
    raw = max(1, int(math.ceil(n / target_ticks)))
    # snap to 1/2/5 * 10^k
    k = 10 ** int(math.floor(math.log10(raw)))
    m = raw / k
    if m <= 1:
        s = 1
    elif m <= 2:
        s = 2
    elif m <= 5:
        s = 5
    else:
        s = 10
    return int(s * k)


def _write_jpg_lineplot(
    out_path: str,
    title: str,
    ylabel: str,
    xs: list[int],
    ys: list[float],
    y_min: float,
    y_max: float,
):
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: pillow\n"
            "Install it with:\n"
            "  pip install pillow\n"
        ) from e

    width, height = 1400, 600
    pad_l, pad_r, pad_t, pad_b = 120, 40, 70, 90
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Filter points (skip NaNs)
    pts = [(x, y) for x, y in zip(xs, ys) if (y == y and not math.isinf(y))]
    if not pts:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        draw.text((20, 20), "No valid points", fill=(0, 0, 0), font=font)
        img.save(out_path, quality=95)
        return

    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        x1 = x0 + 1

    def tx(x: float) -> float:
        return pad_l + (x - x0) * plot_w / (x1 - x0)

    def ty(y: float) -> float:
        y = min(max(y, y_min), y_max)
        return pad_t + (y_max - y) * plot_h / (y_max - y_min)

    # Title
    if title:
        draw.text((pad_l, 24), title, fill=(17, 17, 17), font=font)

    # Axes
    axis_color = (34, 34, 34)
    draw.line([(pad_l, pad_t), (pad_l, pad_t + plot_h)], fill=axis_color, width=2)
    draw.line([(pad_l, pad_t + plot_h), (pad_l + plot_w, pad_t + plot_h)], fill=axis_color, width=2)

    # Y label (left side)
    # Draw rotated text by rendering to a separate image.
    ylabel_img = Image.new("RGBA", (200, 40), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylabel_img)
    yd.text((0, 0), ylabel, fill=(17, 17, 17), font=font)
    ylabel_img = ylabel_img.rotate(90, expand=True)
    img.paste(ylabel_img, (18, int(height / 2 - ylabel_img.size[1] / 2)), ylabel_img)

    # X label
    draw.text((int(width / 2 - 40), height - 40), "关键帧个数", fill=(17, 17, 17), font=font)

    # Y ticks (5)
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5.0
        yy = ty(yv)
        draw.line([(pad_l - 8, yy), (pad_l, yy)], fill=axis_color, width=1)
        draw.text((pad_l - 55, yy - 6), _fmt2(yv), fill=(17, 17, 17), font=font)
        # light gridline
        draw.line([(pad_l, yy), (pad_l + plot_w, yy)], fill=(0, 0, 0, 18), width=1)

    # X ticks (mark keyframe counts)
    n = len(xs)
    step = _nice_step(n, target_ticks=12)
    ticks = list(range(1, n + 1, step))
    if ticks[-1] != n:
        ticks.append(n)
    for xv in ticks:
        xx = tx(xv)
        draw.line([(xx, pad_t + plot_h), (xx, pad_t + plot_h + 8)], fill=axis_color, width=1)
        draw.text((xx - 6, pad_t + plot_h + 12), str(xv), fill=(17, 17, 17), font=font)

    # Line
    pts_sorted = sorted(pts, key=lambda p: p[0])
    line_xy = [(tx(x), ty(y)) for x, y in pts_sorted]
    draw.line(line_xy, fill=(31, 119, 180), width=2)

    # Points + labels (value above)
    for x, y in pts_sorted:
        cx, cy = tx(x), ty(y)
        r = 5
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(214, 39, 40), outline=None)
        draw.text((cx - 10, cy - 18), _fmt2(y), fill=(17, 17, 17), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, quality=95)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = _read_rows(args.metrics_csv, args.only)
    if not rows:
        raise SystemExit(f"No rows found in {args.metrics_csv}")

    # X axis is keyframe count (1..N), as requested.
    xs = list(range(1, len(rows) + 1))

    # Output 3 JPGs
    _write_jpg_lineplot(
        out_path=os.path.join(args.out_dir, f"PSNR_{args.only}.jpg"),
        title=args.title,
        ylabel="PSNR",
        xs=xs,
        ys=[r["psnr"] for r in rows],
        y_min=10.0,
        y_max=30.0,
    )
    _write_jpg_lineplot(
        out_path=os.path.join(args.out_dir, f"SSIM_{args.only}.jpg"),
        title=args.title,
        ylabel="SSIM",
        xs=xs,
        ys=[r["ssim"] for r in rows],
        y_min=0.0,
        y_max=1.0,
    )
    _write_jpg_lineplot(
        out_path=os.path.join(args.out_dir, f"LPIPS_{args.only}.jpg"),
        title=args.title,
        ylabel="LPIPS",
        xs=xs,
        ys=[r["lpips"] for r in rows],
        y_min=0.0,
        y_max=1.0,
    )

    # Tiny summary
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"title={args.title}\n")
        f.write(f"points={len(rows)}\n")
        f.write(f"filter={args.only}\n")


if __name__ == "__main__":
    main()

