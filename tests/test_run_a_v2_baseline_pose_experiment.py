import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from tools.run_a_v2_baseline_pose_experiment import (
    SCENES,
    VARIANTS,
    _summary,
    _write_manifest,
    build_command,
    build_specs,
    parse_args,
    preflight,
    validate_single_gpu,
)


class AV2BaselinePoseRunnerTests(unittest.TestCase):
    def test_a_v222_locks_pose_safe_v18_without_other_pose_variants(self):
        spec = build_specs(
            output_root=Path("/tmp/results"),
            variants=["a_v222_pose_safe_v18"],
            scene_names=["forest1"],
            repeats=[1],
        )[0]
        command = build_command(spec)

        retry_index = command.index("--pose_direct_retry_mode")
        self.assertEqual(command[retry_index + 1], "pose_safe_v18")
        self.assertNotIn("--pose_verification_reference_geometry_mode", command)
        self.assertNotIn("--pose_verification_registration_solver_mode", command)
        self.assertNotIn("--pose_verification_registration_sampling_mode", command)
        self.assertEqual(
            VARIANTS["a_v222_pose_safe_v18"],
            "observe_v1",
        )

    def test_allow_dirty_is_explicit_and_recorded_by_preflight(self):
        args = parse_args(["--allow_dirty"])
        self.assertTrue(args.allow_dirty)

        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                output_root=Path(directory),
                variants=["a_v21"],
                scene_names=["bonsai"],
                repeats=[1],
            )
            inventory = [
                {
                    "index": "1",
                    "uuid": "gpu-1",
                    "name": "test",
                    "driver_version": "test",
                }
            ]
            with (
                mock.patch(
                    "tools.run_a_v2_baseline_pose_experiment._git_dirty",
                    return_value=True,
                ),
                mock.patch(
                    "tools.run_a_v2_baseline_pose_experiment.gpu_inventory",
                    return_value=inventory,
                ),
            ):
                context = preflight(
                    specs,
                    Path(sys.executable),
                    "1",
                    allow_dirty=True,
                )

        self.assertTrue(context["repo_dirty"])

    def test_manifest_preserves_jobs_from_previous_batched_invocations(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            _write_manifest(
                output_root,
                [{"job_id": "repeat01:bonsai:a_off", "returncode": 0}],
            )
            _write_manifest(
                output_root,
                [{"job_id": "repeat02:bonsai:a_off", "returncode": 0}],
            )
            rows = json.loads((output_root / "manifest.json").read_text())

        self.assertEqual(
            [row["job_id"] for row in rows],
            ["repeat01:bonsai:a_off", "repeat02:bonsai:a_off"],
        )

    def test_summary_reads_the_trace_verification_accepted_key(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v2"],
                scene_names=["bonsai"],
                repeats=[1],
            )[0]
            spec.model_dir.mkdir(parents=True)
            (spec.model_dir / "pose_initialization_risk_trace.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "verification_attempts": 37,
                            "verification_accepted": 3,
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = _summary(spec, gpu="6", returncode=0)

        self.assertEqual(summary["a_attempts"], 37)
        self.assertEqual(summary["a_accepts"], 3)

    def test_direct_script_entrypoint_can_resolve_repo_modules(self):
        root = Path(__file__).resolve().parents[1]
        process = subprocess.run(
            [
                sys.executable,
                str(root / "tools" / "run_a_v2_baseline_pose_experiment.py"),
                "--help",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--gpu", process.stdout)

    def test_requires_exactly_one_physical_gpu_identifier(self):
        self.assertEqual(validate_single_gpu("7"), "7")
        with self.assertRaises(ValueError):
            validate_single_gpu("")
        with self.assertRaises(ValueError):
            validate_single_gpu("0,1")

    def test_default_matrix_contains_all_variants_five_repeats_and_nine_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                output_root=Path(directory),
                variants=list(VARIANTS),
                scene_names=list(SCENES),
                repeats=[1, 2, 3, 4, 5],
            )

        self.assertEqual(len(specs), len(VARIANTS) * 5 * 9)
        self.assertEqual({spec.repeat for spec in specs}, {1, 2, 3, 4, 5})
        self.assertEqual({spec.variant for spec in specs}, set(VARIANTS))
        self.assertEqual({spec.scene.name for spec in specs}, set(SCENES))
        self.assertTrue(
            all(f"repeat_{spec.repeat:02d}" in str(spec.run_dir) for spec in specs)
        )

    def test_legacy_variants_only_change_a_mode_and_keep_final_k16_profile(self):
        legacy_variants = ["a_off", "a_current", "a_v2"]
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                output_root=Path(directory),
                variants=legacy_variants,
                scene_names=["bonsai"],
                repeats=[1],
            )
        commands = {spec.variant: build_command(spec) for spec in specs}

        for variant in legacy_variants:
            mode = VARIANTS[variant]
            command = commands[variant]
            mode_index = command.index("--pose_initialization_risk_mode") + 1
            self.assertEqual(command[mode_index], mode)
            self.assertIn("baseline_render_lock_intra_frame_v31", command)
            self.assertIn("--pose_risk_utility_admission_mode", command)
            v2_threshold_index = command.index(
                "--pose_verification_v2_min_improvement"
            ) + 1
            self.assertEqual(command[v2_threshold_index], "0.0")
            self.assertIn("--paper_aligned_pose_render_extra_optimization_max_extra", command)
            budget_index = command.index(
                "--paper_aligned_pose_render_extra_optimization_max_extra"
            ) + 1
            self.assertEqual(command[budget_index], "16")

        normalized = []
        for command in commands.values():
            values = list(command)
            values[values.index("-m") + 1] = "MODEL_DIR"
            values[values.index("--pose_initialization_risk_mode") + 1] = "A_MODE"
            normalized.append(values)
        self.assertTrue(all(command == normalized[0] for command in normalized[1:]))

    def test_a_v21_reuses_verify_v2_and_enables_balanced_step_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v21"],
                scene_names=["bonsai"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        mode_index = command.index("--pose_initialization_risk_mode") + 1
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        self.assertEqual(command[mode_index], "verify_v2")
        self.assertEqual(command[candidate_index], "balanced_step_v21")
        self.assertIn("--pose_verification_v2_min_improvement", command)
        self.assertIn("--paper_aligned_pose_render_extra_optimization_max_extra", command)

    def test_a_v21_strict_restores_the_two_percent_acceptance_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v21_strict"],
                scene_names=["bonsai"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        mode_index = command.index("--pose_initialization_risk_mode") + 1
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        threshold_index = command.index(
            "--pose_verification_v2_min_improvement"
        ) + 1
        self.assertEqual(command[mode_index], "verify_v2")
        self.assertEqual(command[candidate_index], "balanced_step_v21")
        self.assertEqual(command[threshold_index], "0.02")

    def test_a_v22_uses_epipolar_validation_with_the_strict_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v22_epipolar"],
                scene_names=["bonsai"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        threshold_index = command.index(
            "--pose_verification_v2_min_improvement"
        ) + 1
        self.assertEqual(command[candidate_index], "balanced_epipolar_v22")
        self.assertEqual(command[threshold_index], "0.02")

    def test_a_v23_uses_multihypothesis_candidates_with_strict_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v23_multihypothesis"],
                scene_names=["bonsai"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        threshold_index = command.index(
            "--pose_verification_v2_min_improvement"
        ) + 1
        self.assertEqual(command[candidate_index], "multihypothesis_v23")
        self.assertEqual(command[threshold_index], "0.02")

    def test_a_v24_uses_multiview_relative_candidates_with_strict_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v24_multiview"],
                scene_names=["bonsai"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        threshold_index = command.index(
            "--pose_verification_v2_min_improvement"
        ) + 1
        temporal_index = command.index(
            "--pose_verification_max_temporal_score_ratio"
        ) + 1
        self.assertEqual(command[candidate_index], "multiview_relative_v24")
        self.assertEqual(command[threshold_index], "0.02")
        self.assertEqual(command[temporal_index], "0.90")

    def test_a_v25_freezes_joint_pose_at_the_verified_geometry_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v25_anchor_freeze"],
                scene_names=["bonsai"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        threshold_index = command.index(
            "--pose_verification_v2_min_improvement"
        ) + 1
        anchor_index = command.index(
            "--pose_verification_geometry_anchor_mode"
        ) + 1
        self.assertEqual(command[candidate_index], "balanced_step_v21")
        self.assertEqual(command[threshold_index], "0.02")
        self.assertEqual(command[anchor_index], "freeze_v1")

    def test_a_v26_uses_frozen_sparse_reference_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v26_frozen_sparse"],
                scene_names=["counter"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        snapshot_index = command.index(
            "--pose_verification_reference_geometry_mode"
        ) + 1
        self.assertEqual(command[candidate_index], "balanced_step_v21")
        self.assertEqual(command[snapshot_index], "frozen_first_valid_v1")

    def test_a_v217_keeps_sparse_points_and_reference_poses_in_one_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v217_coherent_frozen_geometry"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        anchor_index = command.index(
            "--pose_verification_geometry_anchor_mode"
        ) + 1
        snapshot_index = command.index(
            "--pose_verification_reference_geometry_mode"
        ) + 1
        self.assertEqual(command[candidate_index], "balanced_step_v21")
        self.assertEqual(command[anchor_index], "freeze_v1")
        self.assertEqual(command[snapshot_index], "frozen_first_valid_v1")

    def test_a_v218_uses_support_guarded_frozen_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v218_guarded_frozen_geometry"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        snapshot_index = command.index(
            "--pose_verification_reference_geometry_mode"
        ) + 1
        support_index = command.index(
            "--pose_verification_frozen_min_match_support"
        ) + 1
        ratio_index = command.index(
            "--pose_verification_frozen_min_live_ratio"
        ) + 1
        self.assertEqual(command[snapshot_index], "guarded_frozen_v3")
        self.assertEqual(command[support_index], "24")
        self.assertEqual(command[ratio_index], "0.50")
        self.assertNotIn(
            "--pose_verification_geometry_anchor_mode",
            command,
        )

    def test_a_v219_guards_frozen_global_points_without_freezing_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v219_guarded_frozen_live_pose"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        snapshot_index = command.index(
            "--pose_verification_reference_geometry_mode"
        ) + 1
        support_index = command.index(
            "--pose_verification_frozen_min_match_support"
        ) + 1
        ratio_index = command.index(
            "--pose_verification_frozen_min_live_ratio"
        ) + 1
        self.assertEqual(
            command[snapshot_index],
            "guarded_frozen_live_pose_v4",
        )
        self.assertEqual(command[support_index], "24")
        self.assertEqual(command[ratio_index], "0.50")
        self.assertNotIn(
            "--pose_verification_geometry_anchor_mode",
            command,
        )

    def test_a_v220_uses_frame_homogeneous_frozen_reference_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v220_homogeneous_frozen_geometry"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        snapshot_index = command.index(
            "--pose_verification_reference_geometry_mode"
        ) + 1
        min_refs_index = command.index(
            "--pose_verification_frozen_min_reference_count"
        ) + 1
        total_support_index = command.index(
            "--pose_verification_frozen_min_total_support"
        ) + 1
        self.assertEqual(
            command[snapshot_index],
            "guarded_frozen_homogeneous_v5",
        )
        self.assertEqual(command[min_refs_index], "2")
        self.assertEqual(command[total_support_index], "48")

    def test_a_v221_uses_all_frozen_references_with_global_support_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v221_frozen_global_support_guard"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        snapshot_index = command.index(
            "--pose_verification_reference_geometry_mode"
        ) + 1
        total_support_index = command.index(
            "--pose_verification_frozen_min_total_support"
        ) + 1
        self.assertEqual(
            command[snapshot_index],
            "frozen_global_support_guard_v6",
        )
        self.assertEqual(command[total_support_index], "24")

    def test_a_v27_freezes_only_a_verification_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v27_frozen_verification"],
                scene_names=["counter"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        snapshot_index = command.index(
            "--pose_verification_reference_geometry_mode"
        ) + 1
        self.assertEqual(command[candidate_index], "balanced_step_v21")
        self.assertEqual(
            command[snapshot_index],
            "frozen_verification_only_v2",
        )

    def test_a_v28_uses_stable_anchor_reference_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v28_stable_anchor"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        anchor_mode_index = command.index(
            "--pose_verification_anchor_reference_mode"
        ) + 1
        scope_index = command.index(
            "--pose_verification_anchor_candidate_scope"
        ) + 1
        snapshot_index = command.index(
            "--pose_verification_reference_geometry_mode"
        ) + 1
        pre_error_index = command.index(
            "--pose_verification_anchor_pre_error_scale"
        ) + 1
        self.assertEqual(command[candidate_index], "balanced_step_v21")
        self.assertEqual(command[anchor_mode_index], "stable_anchor_v1")
        self.assertEqual(command[scope_index], "anchor_only_v1")
        self.assertEqual(
            command[snapshot_index],
            "frozen_verification_only_v2",
        )
        self.assertEqual(command[pre_error_index], "4.0")

    def test_a_v29_adds_only_the_conservative_reference_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v29_reference_guard"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        policy_index = command.index(
            "--pose_verification_reference_policy"
        ) + 1
        threshold_index = command.index(
            "--pose_verification_reference_risk_threshold"
        ) + 1
        cooldown_index = command.index(
            "--pose_verification_reference_cooldown_frames"
        ) + 1
        self.assertEqual(
            command[policy_index],
            "conservative_quarantine_v1",
        )
        self.assertEqual(command[threshold_index], "0.12")
        self.assertEqual(command[cooldown_index], "20")
        self.assertIn(
            "--pose_verification_anchor_reference_mode",
            command,
        )
        self.assertIn(
            "--pose_verification_anchor_candidate_scope",
            command,
        )

    def test_a_v210_guards_high_risk_references_even_when_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v210_high_risk_reference_guard"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        policy_index = command.index(
            "--pose_verification_reference_policy"
        ) + 1
        self.assertEqual(
            command[policy_index],
            "conservative_high_risk_v2",
        )
        self.assertIn(
            "--pose_verification_anchor_reference_mode",
            command,
        )

    def test_a_v211_reuses_the_a_activation_floor_for_reference_guarding(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v211_trigger_aligned_reference_guard"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        threshold_index = command.index(
            "--pose_verification_reference_risk_threshold"
        ) + 1
        self.assertEqual(command[threshold_index], "0.10")
        policy_index = command.index(
            "--pose_verification_reference_policy"
        ) + 1
        self.assertEqual(
            command[policy_index],
            "conservative_high_risk_v2",
        )

    def test_a_v212_rejects_candidates_that_worsen_temporal_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v212_temporal_safe_reference_guard"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        ratio_index = command.index(
            "--pose_verification_max_temporal_score_ratio"
        ) + 1
        self.assertEqual(command[ratio_index], "1.0")
        threshold_index = command.index(
            "--pose_verification_reference_risk_threshold"
        ) + 1
        self.assertEqual(command[threshold_index], "0.10")

    def test_a_v213_uses_independent_deterministic_pose_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v213_deterministic_pose_sampling"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        mode_index = command.index(
            "--pose_verification_registration_sampling_mode"
        ) + 1
        self.assertEqual(command[mode_index], "frame_deterministic_v1")
        threshold_index = command.index(
            "--pose_verification_reference_risk_threshold"
        ) + 1
        self.assertEqual(command[threshold_index], "0.10")

    def test_a_v214_uses_deterministic_opencv_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v214_deterministic_opencv_registration"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        sampling_index = command.index(
            "--pose_verification_registration_sampling_mode"
        ) + 1
        self.assertEqual(command[sampling_index], "frame_deterministic_v1")
        solver_index = command.index(
            "--pose_verification_registration_solver_mode"
        ) + 1
        self.assertEqual(command[solver_index], "deterministic_opencv_v2")
        threshold_index = command.index(
            "--pose_verification_reference_risk_threshold"
        ) + 1
        self.assertEqual(command[threshold_index], "0.10")

    def test_a_v215_protects_pose_after_fixed_async_joint_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v215_fixed_async_pose_budget"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        mode_index = command.index(
            "--pose_verification_async_pose_protection_mode"
        ) + 1
        self.assertEqual(command[mode_index], "fixed_joint_budget_v1")
        self.assertNotIn(
            "--pose_verification_registration_sampling_mode",
            command,
        )
        threshold_index = command.index(
            "--pose_verification_reference_risk_threshold"
        ) + 1
        self.assertEqual(command[threshold_index], "0.10")

    def test_a_v216_combines_recent_and_stable_anchor_pose_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v216_combined_reference_support"],
                scene_names=["forest1"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        scope_index = command.index(
            "--pose_verification_anchor_candidate_scope"
        ) + 1
        self.assertEqual(command[scope_index], "combined_v1")
        self.assertNotIn(
            "--pose_verification_registration_sampling_mode",
            command,
        )
        self.assertNotIn(
            "--pose_verification_async_pose_protection_mode",
            command,
        )
        threshold_index = command.index(
            "--pose_verification_reference_risk_threshold"
        ) + 1
        self.assertEqual(command[threshold_index], "0.10")

    def test_a_v21_sensitive_only_lowers_the_adaptive_risk_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v21_sensitive"],
                scene_names=["bonsai"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        sigma_indices = [
            index
            for index, value in enumerate(command)
            if value == "--pose_initialization_risk_adaptive_sigma"
        ]
        threshold_index = command.index(
            "--pose_verification_v2_min_improvement"
        ) + 1
        candidate_index = command.index(
            "--pose_verification_candidate_mode"
        ) + 1
        self.assertEqual(command[sigma_indices[-1] + 1], "0.0")
        self.assertEqual(command[threshold_index], "0.02")
        self.assertEqual(command[candidate_index], "balanced_step_v21")

    def test_a_v21_sensitive_g005_uses_the_screened_half_percent_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = build_specs(
                output_root=Path(directory),
                variants=["a_v21_sensitive_g005"],
                scene_names=["counter"],
                repeats=[1],
            )[0]

        command = build_command(spec)
        threshold_index = command.index(
            "--pose_verification_v2_min_improvement"
        ) + 1
        sigma_indices = [
            index
            for index, value in enumerate(command)
            if value == "--pose_initialization_risk_adaptive_sigma"
        ]
        self.assertEqual(command[threshold_index], "0.005")
        self.assertEqual(command[sigma_indices[-1] + 1], "0.0")

    def test_each_scene_keeps_all_variants_adjacent_with_rotated_repeat_order(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_specs(
                output_root=Path(directory),
                variants=list(VARIANTS),
                scene_names=["bonsai"],
                repeats=[1, 2, 3],
            )

        by_repeat = {
            repeat: [spec.variant for spec in specs if spec.repeat == repeat]
            for repeat in (1, 2, 3)
        }
        variants = list(VARIANTS)
        self.assertEqual(by_repeat[1], variants)
        self.assertEqual(by_repeat[2], variants[1:] + variants[:1])
        self.assertEqual(by_repeat[3], variants[2:] + variants[:2])


if __name__ == "__main__":
    unittest.main()
