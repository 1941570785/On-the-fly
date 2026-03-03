import argparse
import csv
import os
import runpy
import sys
import time
from dataclasses import dataclass


@dataclass
class ExportConfig:
    enabled: bool
    out_dir: str
    csv_path: str
    pyr_lvl: int
    export_every_kf: int
    include_test: bool
    include_train: bool
    disable_after_inference: bool


def _write_image(path: str, rgb_uint8):
    # rgb_uint8: HxWx3 RGB uint8
    try:
        import cv2  # type: ignore

        os.makedirs(os.path.dirname(path), exist_ok=True)
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        return
    except ModuleNotFoundError:
        pass
    except Exception:
        pass

    try:
        import imageio.v2 as imageio  # type: ignore

        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imwrite(path, rgb_uint8)
        return
    except ModuleNotFoundError:
        pass
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(rgb_uint8, mode="RGB").save(path)
        return
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "No image writer backend available. Install one of:\n"
            "  pip install opencv-python\n"
            "  pip install imageio\n"
            "  pip install pillow\n"
        ) from e


def _append_csv(csv_path: str, header: list[str], row: list):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def _parse_wrapper_args():
    p = argparse.ArgumentParser(
        description=(
            "Run train.py WITHOUT modifying it, but export one rendered image + metrics "
            "(PSNR/SSIM/LPIPS) per registered keyframe via runtime monkeypatch."
        ),
        add_help=True,
    )
    p.add_argument("--export_keyframes", action="store_true")
    p.add_argument("--export_dir", type=str, default="")
    p.add_argument("--metrics_csv", type=str, default="")
    p.add_argument("--export_pyr_lvl", type=int, default=0)
    p.add_argument("--export_every_kf", type=int, default=1)
    p.add_argument("--include_test", action="store_true", help="Also export keyframes marked is_test=True")
    p.add_argument("--include_train", action="store_true", help="Also export keyframes marked is_test=False")
    p.add_argument(
        "--disable_after_inference",
        action="store_true",
        help="Stop exporting after train.py switches scene_model to inference mode.",
    )
    p.add_argument(
        "--train_entry",
        type=str,
        default="train.py",
        help="Path to the training entry script (default: train.py).",
    )
    args, train_argv = p.parse_known_args()

    # Default behavior: include both train and test unless user restricts.
    include_test = args.include_test or (not args.include_test and not args.include_train)
    include_train = args.include_train or (not args.include_test and not args.include_train)

    cfg = ExportConfig(
        enabled=bool(args.export_keyframes),
        out_dir=args.export_dir,
        csv_path=args.metrics_csv,
        pyr_lvl=int(args.export_pyr_lvl),
        export_every_kf=max(1, int(args.export_every_kf)),
        include_test=include_test,
        include_train=include_train,
        disable_after_inference=bool(args.disable_after_inference),
    )
    return cfg, args.train_entry, train_argv


def main():
    cfg, train_entry, train_argv = _parse_wrapper_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Capture model_path from train's get_args without modifying train.py
    import args as args_mod  # type: ignore

    orig_get_args = args_mod.get_args
    captured = {"model_path": None}

    def wrapped_get_args():
        a = orig_get_args()
        captured["model_path"] = getattr(a, "model_path", None)
        return a

    args_mod.get_args = wrapped_get_args  # type: ignore

    export_state = {
        "enabled": cfg.enabled,
        "exported_kfs": set(),
        "last_export_time": 0.0,
    }

    if cfg.enabled:
        print("[wrapper] Keyframe export is ENABLED")
    else:
        print("[wrapper] Keyframe export is DISABLED (pass --export_keyframes to enable)")

    # Monkeypatch SceneModel methods
    from scene.scene_model import SceneModel  # type: ignore
    from fused_ssim import fused_ssim  # type: ignore
    from utils import psnr  # type: ignore

    orig_opt_loop = SceneModel.optimization_loop
    orig_enable_infer = getattr(SceneModel, "enable_inference_mode", None)

    def _should_export_kf(kf):
        is_test = bool(kf.info.get("is_test", False))
        return (is_test and cfg.include_test) or ((not is_test) and cfg.include_train)

    def _export_new_keyframes(scene_model: SceneModel):
        if not export_state["enabled"]:
            return
        model_path = captured["model_path"] or ""
        if model_path == "":
            # If we couldn't capture it, fall back to cwd.
            model_path = os.getcwd()

        out_dir = cfg.out_dir if cfg.out_dir != "" else os.path.join(model_path, "keyframe_renders")
        csv_path = cfg.csv_path if cfg.csv_path != "" else os.path.join(model_path, "keyframe_metrics.csv")

        # Export any keyframes we haven't exported yet, respecting export_every_kf
        for kid, kf in enumerate(scene_model.keyframes):
            if kid in export_state["exported_kfs"]:
                continue
            if kid % cfg.export_every_kf != 0:
                export_state["exported_kfs"].add(kid)
                continue
            if not _should_export_kf(kf):
                export_state["exported_kfs"].add(kid)
                continue

            # Render and compute metrics
            render_pkg = scene_model.render_from_id(kid, pyr_lvl=cfg.pyr_lvl)
            pred = render_pkg["render"].clamp(0, 1)

            gt = kf.image_pyr[cfg.pyr_lvl].cuda().clamp(0, 1)
            if kf.mask_pyr is not None:
                mask = kf.mask_pyr[cfg.pyr_lvl].cuda()
            else:
                mask = (gt[:1] > 0).to(gt.device)
            mask = mask.expand_as(pred)

            pred_m = pred * mask
            gt_m = gt * mask

            psnr_val = psnr(pred_m[mask], gt_m[mask]) if mask.any() else float("nan")
            ssim_val = fused_ssim(pred_m[None], gt_m[None], train=False).item()

            lpips_val = float("nan")
            if getattr(scene_model, "lpips", None) is not None:
                try:
                    lpips_val = scene_model.lpips(pred_m[None], gt_m[None]).item()
                except Exception:
                    lpips_val = float("nan")

            img = pred.mul(255).permute(1, 2, 0).byte().detach().cpu().numpy()
            img_path = os.path.join(out_dir, f"{kid:06d}.png")
            _write_image(img_path, img)

            header = [
                "timestamp",
                "keyframe_id",
                "image_name",
                "is_test",
                "pyr_lvl",
                "psnr",
                "ssim",
                "lpips",
                "num_keyframes",
                "num_anchors",
                "num_gaussians",
            ]
            row = [
                time.time(),
                int(kid),
                kf.info.get("name", ""),
                bool(kf.info.get("is_test", False)),
                int(cfg.pyr_lvl),
                float(psnr_val),
                float(ssim_val),
                float(lpips_val),
                int(len(scene_model.keyframes)),
                int(len(scene_model.anchors)),
                int(scene_model.n_active_gaussians),
            ]
            _append_csv(csv_path, header, row)

            export_state["exported_kfs"].add(kid)

        export_state["last_export_time"] = time.time()

    def patched_optimization_loop(self: SceneModel, *a, **kw):
        out = orig_opt_loop(self, *a, **kw)
        try:
            _export_new_keyframes(self)
        except Exception as e:
            # Do not break training due to export issues
            print(f"[wrapper] export failed: {e}")
        return out

    SceneModel.optimization_loop = patched_optimization_loop  # type: ignore

    if orig_enable_infer is not None and cfg.disable_after_inference:

        def patched_enable_inference_mode(self: SceneModel, *a, **kw):
            out = orig_enable_infer(self, *a, **kw)
            export_state["enabled"] = False
            return out

        SceneModel.enable_inference_mode = patched_enable_inference_mode  # type: ignore

    # Normalize pass-through argv:
    # - Support users adding a leading "--" separator (argparse passthrough)
    # - Support "python train.py ..." prefixes (strip them)
    while train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]
    if train_argv and train_argv[0] == "python":
        train_argv = train_argv[1:]
    if train_argv and os.path.basename(train_argv[0]) == os.path.basename(train_entry):
        train_argv = train_argv[1:]

    # Run train.py with original arguments (excluding wrapper args)
    sys.argv = [train_entry] + train_argv
    runpy.run_path(os.path.join(repo_root, train_entry), run_name="__main__")


if __name__ == "__main__":
    main()

