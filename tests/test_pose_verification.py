import math
import unittest
from types import SimpleNamespace

import torch

from poses.pose_verification import (
    async_pose_update_enabled,
    choose_verified_pose,
    compute_epipolar_sampson_errors,
    compute_reprojection_errors,
    decide_failed_verification_reference_quarantine,
    decide_pose_refinement,
    deterministic_pose_sampling_seed,
    estimate_multireference_relative_pose_candidates,
    estimate_opencv_pnp_candidates,
    geometry_anchor_pose,
    interpolate_world_to_camera_pose,
    pose_correction_magnitude,
    predict_constant_velocity_pose,
    resolve_stable_pose_anchor_probe_references,
    rank_stable_pose_anchor_records,
    sample_stable_pose_anchor_candidates,
    select_pose_candidate_by_reprojection,
    select_balanced_correspondence_indices,
    select_robust_correspondences,
    split_pose_verification_evidence,
    summarize_reprojection_errors,
    validation_candidate_rank,
)


def _pose(*, tx: float = 0.0, rz_deg: float = 0.0) -> torch.Tensor:
    angle = math.radians(rz_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [
            [c, -s, 0.0, tx],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


class PoseVerificationGeometryTests(unittest.TestCase):
    def test_async_pose_protection_limits_pose_updates_to_base_budget(self):
        self.assertTrue(
            async_pose_update_enabled(
                "fixed_joint_budget_v1",
                run_until_interrupt=True,
                iteration=9,
                base_iterations=10,
            )
        )
        self.assertFalse(
            async_pose_update_enabled(
                "fixed_joint_budget_v1",
                run_until_interrupt=True,
                iteration=10,
                base_iterations=10,
            )
        )
        self.assertTrue(
            async_pose_update_enabled(
                "fixed_joint_budget_v1",
                run_until_interrupt=False,
                iteration=10,
                base_iterations=10,
            )
        )
        self.assertTrue(
            async_pose_update_enabled(
                "off",
                run_until_interrupt=True,
                iteration=100,
                base_iterations=10,
            )
        )

    def test_pose_candidate_ranking_prefers_stronger_reprojection_support(self):
        points = torch.tensor(
            [
                [-1.0, -1.0, 4.0],
                [1.0, -1.0, 4.0],
                [-1.0, 1.0, 4.0],
                [1.0, 1.0, 4.0],
                [0.0, -0.5, 3.0],
                [0.5, 0.0, 5.0],
            ],
            dtype=torch.float32,
        )
        focal = torch.tensor([100.0])
        centre = torch.tensor([50.0, 50.0])
        observations = torch.stack(
            [
                focal[0] * points[:, 0] / points[:, 2] + centre[0],
                focal[0] * points[:, 1] / points[:, 2] + centre[1],
            ],
            dim=1,
        )
        wrong = torch.eye(4)
        wrong[0, 3] = 1.0

        selected, debug = select_pose_candidate_by_reprojection(
            [
                {"name": "wrong", "pose": wrong},
                {"name": "correct", "pose": torch.eye(4)},
            ],
            points,
            observations,
            focal=focal,
            centre=centre,
            max_reprojection_error=2.0,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["name"], "correct")
        self.assertEqual(int(selected["inlier_mask"].sum().item()), len(points))
        self.assertEqual(debug["selected_name"], "correct")

    def test_deterministic_pose_sampling_seed_is_frame_specific(self):
        first = deterministic_pose_sampling_seed(119)
        repeated = deterministic_pose_sampling_seed(119)
        next_frame = deterministic_pose_sampling_seed(120)
        verification_stream = deterministic_pose_sampling_seed(119, stream=1)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_frame)
        self.assertNotEqual(first, verification_stream)
        self.assertGreaterEqual(first, 0)

    def test_failed_verification_reference_quarantine_is_conservative(self):
        risk_event = {
            "eligible": True,
            "warmed_up": True,
            "verification_trigger": True,
            "verification_accepted": False,
            "risk_score": 0.15,
            "adaptive_threshold": 0.08,
            "multi_signal_risk": True,
            "severe_pose_risk": False,
        }

        accepted = decide_failed_verification_reference_quarantine(
            {**risk_event, "verification_accepted": True},
            policy="conservative_quarantine_v1",
            risk_threshold=0.12,
            frame_id=100,
            last_quarantine_frame_id=-1,
            cooldown_frames=20,
        )
        below_threshold = decide_failed_verification_reference_quarantine(
            {**risk_event, "risk_score": 0.119},
            policy="conservative_quarantine_v1",
            risk_threshold=0.12,
            frame_id=100,
            last_quarantine_frame_id=-1,
            cooldown_frames=20,
        )
        quarantined = decide_failed_verification_reference_quarantine(
            risk_event,
            policy="conservative_quarantine_v1",
            risk_threshold=0.12,
            frame_id=100,
            last_quarantine_frame_id=-1,
            cooldown_frames=20,
        )
        cooldown = decide_failed_verification_reference_quarantine(
            risk_event,
            policy="conservative_quarantine_v1",
            risk_threshold=0.12,
            frame_id=115,
            last_quarantine_frame_id=100,
            cooldown_frames=20,
        )
        disabled = decide_failed_verification_reference_quarantine(
            risk_event,
            policy="off",
            risk_threshold=0.12,
            frame_id=100,
            last_quarantine_frame_id=-1,
            cooldown_frames=20,
        )

        self.assertFalse(accepted["quarantine"])
        self.assertEqual(accepted["reason"], "candidate_accepted")
        self.assertFalse(below_threshold["quarantine"])
        self.assertEqual(below_threshold["reason"], "risk_below_threshold")
        self.assertTrue(quarantined["quarantine"])
        self.assertEqual(quarantined["reason"], "failed_verification_high_risk")
        self.assertFalse(cooldown["quarantine"])
        self.assertTrue(cooldown["cooldown_active"])
        self.assertEqual(cooldown["reason"], "cooldown")
        self.assertFalse(disabled["quarantine"])
        self.assertEqual(disabled["reason"], "policy_off")

    def test_high_risk_reference_quarantine_is_independent_of_acceptance(self):
        risk_event = {
            "eligible": True,
            "warmed_up": True,
            "verification_trigger": True,
            "risk_score": 0.15,
            "adaptive_threshold": 0.08,
            "multi_signal_risk": True,
            "severe_pose_risk": False,
        }

        accepted = decide_failed_verification_reference_quarantine(
            {**risk_event, "verification_accepted": True},
            policy="conservative_high_risk_v2",
            risk_threshold=0.12,
            frame_id=100,
            last_quarantine_frame_id=-1,
            cooldown_frames=20,
        )
        rejected = decide_failed_verification_reference_quarantine(
            {**risk_event, "verification_accepted": False},
            policy="conservative_high_risk_v2",
            risk_threshold=0.12,
            frame_id=100,
            last_quarantine_frame_id=-1,
            cooldown_frames=20,
        )

        self.assertTrue(accepted["quarantine"])
        self.assertEqual(accepted["reason"], "verified_high_risk_reference")
        self.assertTrue(rejected["quarantine"])
        self.assertEqual(rejected["reason"], "failed_high_risk_reference")

    def test_stable_anchor_pool_filters_unsafe_and_base_references(self):
        records = [
            {
                "keyframe_id": 1,
                "source_frame_id": 10,
                "support_count": 900,
                "risk_score": 0.01,
            },
            {
                "keyframe_id": 2,
                "source_frame_id": 20,
                "support_count": 900,
                "risk_score": 0.01,
            },
            {
                "keyframe_id": 3,
                "source_frame_id": 30,
                "support_count": 900,
                "risk_score": 0.20,
            },
            {
                "keyframe_id": 4,
                "source_frame_id": 40,
                "support_count": 100,
                "risk_score": 0.01,
            },
            {
                "keyframe_id": 5,
                "source_frame_id": 50,
                "support_count": 900,
                "risk_score": 0.01,
                "isolated": True,
            },
            {
                "keyframe_id": 6,
                "source_frame_id": 60,
                "support_count": 900,
                "risk_score": 0.01,
            },
            {
                "keyframe_id": 7,
                "source_frame_id": 70,
                "support_count": 900,
                "risk_score": 0.01,
            },
        ]

        pool, debug = sample_stable_pose_anchor_candidates(
            records,
            base_reference_ids={1},
            current_frame_id=100,
            min_age_frames=30,
            min_support_count=400,
            max_risk_score=0.08,
            max_pool_size=8,
        )

        self.assertEqual([record["keyframe_id"] for record in pool], [2, 6, 7])
        self.assertEqual(debug["eligible_count"], 3)
        self.assertEqual(debug["sampled_count"], 3)

    def test_stable_anchor_ranking_enforces_temporal_diversity(self):
        records = [
            {"keyframe_id": 10, "source_frame_id": 100, "match_score": 500.0},
            {"keyframe_id": 11, "source_frame_id": 104, "match_score": 490.0},
            {"keyframe_id": 12, "source_frame_id": 150, "match_score": 430.0},
            {"keyframe_id": 13, "source_frame_id": 210, "match_score": 420.0},
            {"keyframe_id": 14, "source_frame_id": 280, "match_score": 100.0},
        ]

        selected, debug = rank_stable_pose_anchor_records(
            records,
            max_references=3,
            min_match_score=180.0,
            min_source_separation=24,
        )

        self.assertEqual(
            [record["keyframe_id"] for record in selected],
            [10, 12, 13],
        )
        self.assertEqual(debug["above_match_threshold_count"], 4)
        self.assertEqual(debug["selected_count"], 3)
        self.assertEqual(debug["diversity_rejected_count"], 1)

    def test_anchor_only_probe_requires_two_stable_references(self):
        base = [SimpleNamespace(index=1), SimpleNamespace(index=2)]
        anchors = [SimpleNamespace(index=10), SimpleNamespace(index=20)]

        selected, debug = resolve_stable_pose_anchor_probe_references(
            base,
            anchors,
            candidate_scope="anchor_only_v1",
        )
        fallback, fallback_debug = (
            resolve_stable_pose_anchor_probe_references(
                base,
                anchors[:1],
                candidate_scope="anchor_only_v1",
            )
        )

        self.assertEqual([reference.index for reference in selected], [10, 20])
        self.assertEqual(debug["resolved_scope"], "anchor_only_v1")
        self.assertEqual(
            [reference.index for reference in fallback],
            [1, 2, 10],
        )
        self.assertEqual(
            fallback_debug["resolved_scope"],
            "combined_fallback",
        )

    def test_geometry_anchor_pose_is_opt_in_and_exact(self):
        current = _pose(tx=0.20, rz_deg=8.0)
        anchor = _pose(tx=0.05, rz_deg=2.0)

        disabled = geometry_anchor_pose(
            current,
            {
                "_pose_verification_geometry_anchor_mode": "off",
                "_pose_verification_geometry_anchor_Rt": anchor.tolist(),
            },
        )
        enabled = geometry_anchor_pose(
            current,
            {
                "_pose_verification_geometry_anchor_mode": "freeze_v1",
                "_pose_verification_geometry_anchor_Rt": anchor.tolist(),
            },
        )

        self.assertTrue(torch.equal(disabled, current))
        self.assertNotEqual(disabled.data_ptr(), current.data_ptr())
        self.assertTrue(torch.equal(enabled, anchor))

    def test_constant_velocity_prediction_respects_frame_gap(self):
        first = _pose(rz_deg=0.0)
        first_centre = torch.tensor([-0.30, 0.0, 0.0])
        first[:3, 3] = -(first[:3, :3] @ first_centre)
        second = _pose(rz_deg=2.0)
        second_centre = torch.tensor([-0.20, 0.0, 0.0])
        second[:3, 3] = -(second[:3, :3] @ second_centre)

        prediction, debug = predict_constant_velocity_pose(
            [(4, first), (6, second)],
            current_frame_id=10,
            like_pose=second,
        )

        self.assertIsNotNone(prediction)
        self.assertEqual(debug["history_frame_gap"], 2)
        self.assertEqual(debug["current_frame_gap"], 4)
        self.assertAlmostEqual(float(prediction[0, 3]), 0.0, places=4)
        correction = pose_correction_magnitude(_pose(rz_deg=6.0), prediction)
        self.assertLess(correction["rotation_deg"], 1e-3)

    def test_multireference_relative_pose_recovers_camera_geometry(self):
        generator = torch.Generator().manual_seed(31)
        focal = torch.tensor([500.0])
        centre = torch.tensor([320.0, 240.0])
        points = torch.rand((100, 3), generator=generator)
        points[:, :2] = (points[:, :2] - 0.5) * 2.0
        points[:, 2] = 3.0 + 3.0 * points[:, 2]

        references = []
        for camera_centre in (
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([0.45, 0.0, 0.0]),
            torch.tensor([0.0, 0.35, 0.0]),
        ):
            pose = _pose()
            pose[:3, 3] = -camera_centre
            references.append(pose)
        true_pose = _pose(rz_deg=6.0)
        true_centre = torch.tensor([0.20, 0.14, 0.08])
        true_pose[:3, 3] = -(true_pose[:3, :3] @ true_centre)
        initial_pose = _pose(rz_deg=1.0)
        initial_centre = torch.tensor([0.05, 0.03, 0.0])
        initial_pose[:3, 3] = -(
            initial_pose[:3, :3] @ initial_centre
        )

        def project(pose: torch.Tensor) -> torch.Tensor:
            camera = (
                points @ pose[:3, :3].transpose(0, 1)
                + pose[:3, 3]
            )
            return camera[:, :2] / camera[:, 2:] * focal + centre

        current_uv_parts = []
        reference_uv_parts = []
        reference_pose_parts = []
        reference_id_parts = []
        current_uv = project(true_pose)
        for ref_id, reference_pose in enumerate(references):
            noisy_current = current_uv + 0.10 * torch.randn(
                current_uv.shape, generator=generator
            )
            noisy_reference = project(reference_pose) + 0.10 * torch.randn(
                current_uv.shape, generator=generator
            )
            noisy_current[:8] = torch.rand(
                (8, 2), generator=generator
            ) * torch.tensor([640.0, 480.0])
            current_uv_parts.append(noisy_current)
            reference_uv_parts.append(noisy_reference)
            reference_pose_parts.append(
                reference_pose.unsqueeze(0).repeat(len(points), 1, 1)
            )
            reference_id_parts.append(
                torch.full((len(points),), ref_id, dtype=torch.long)
            )

        candidates, debug = estimate_multireference_relative_pose_candidates(
            torch.cat(current_uv_parts),
            torch.cat(reference_uv_parts),
            torch.cat(reference_pose_parts),
            torch.cat(reference_id_parts),
            focal=focal,
            centre=centre,
            initial_pose=initial_pose,
            max_epipolar_error=1.0,
        )

        self.assertGreaterEqual(debug["usable_references"], 2)
        self.assertGreaterEqual(len(candidates), 1)
        best = min(
            candidates,
            key=lambda item: (
                pose_correction_magnitude(true_pose, item["pose"])[
                    "translation"
                ]
                + math.radians(
                    pose_correction_magnitude(true_pose, item["pose"])[
                        "rotation_deg"
                    ]
                )
            ),
        )
        correction = pose_correction_magnitude(true_pose, best["pose"])
        self.assertLess(correction["translation"], 0.08)
        self.assertLess(correction["rotation_deg"], 1.0)

    def test_validation_rank_rejects_median_only_tail_overfit(self):
        pre = {
            "valid_count": 100,
            "mean": 10.0,
            "median": 10.0,
            "p90": 10.0,
            "max": 12.0,
        }
        balanced = {
            "valid_count": 100,
            "mean": 9.80,
            "median": 9.70,
            "p90": 9.80,
            "max": 12.0,
        }
        median_only = {
            "valid_count": 100,
            "mean": 9.99,
            "median": 9.50,
            "p90": 10.05,
            "max": 12.0,
        }

        self.assertLess(
            validation_candidate_rank(pre, balanced),
            validation_candidate_rank(pre, median_only),
        )

    def test_opencv_pnp_candidates_recover_pose_with_outliers(self):
        generator = torch.Generator().manual_seed(17)
        points = torch.rand((120, 3), generator=generator)
        points[:, :2] = (points[:, :2] - 0.5) * 2.0
        points[:, 2] = 2.0 + 3.0 * points[:, 2]
        true_pose = _pose(tx=0.18, rz_deg=7.0)
        initial_pose = _pose(tx=0.05, rz_deg=1.0)
        focal = torch.tensor([420.0])
        centre = torch.tensor([320.0, 240.0])

        camera = (
            points @ true_pose[:3, :3].transpose(0, 1)
            + true_pose[:3, 3]
        )
        uv = camera[:, :2] / camera[:, 2:] * focal + centre
        uv += 0.2 * torch.randn(uv.shape, generator=generator)
        uv[:24] = torch.rand((24, 2), generator=generator) * torch.tensor(
            [640.0, 480.0]
        )

        candidates, debug = estimate_opencv_pnp_candidates(
            points,
            uv,
            focal=focal,
            centre=centre,
            initial_pose=initial_pose,
            max_reprojection_error=2.0,
        )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertGreaterEqual(debug["successful_candidates"], 1)
        best = min(
            candidates,
            key=lambda item: (
                pose_correction_magnitude(true_pose, item["pose"])[
                    "translation"
                ]
                + math.radians(
                    pose_correction_magnitude(true_pose, item["pose"])[
                        "rotation_deg"
                    ]
                )
            ),
        )
        correction = pose_correction_magnitude(true_pose, best["pose"])
        self.assertLess(correction["translation"], 0.02)
        self.assertLess(correction["rotation_deg"], 0.5)

    def test_epipolar_sampson_error_prefers_the_geometry_consistent_pose(self):
        points = torch.tensor(
            [
                [-0.2, -0.1, 2.0],
                [0.1, -0.2, 2.5],
                [0.3, 0.1, 3.0],
                [-0.1, 0.3, 2.2],
                [0.4, -0.3, 3.5],
                [-0.3, 0.2, 2.8],
            ],
            dtype=torch.float32,
        )
        reference_pose = _pose()
        true_pose = _pose(tx=0.2, rz_deg=5.0)
        wrong_pose = _pose(tx=0.2, rz_deg=0.0)
        focal = torch.tensor([100.0])
        centre = torch.tensor([50.0, 50.0])

        def project(pose: torch.Tensor) -> torch.Tensor:
            camera = points @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]
            return camera[:, :2] / camera[:, 2:] * focal + centre

        reference_uv = project(reference_pose)
        current_uv = project(true_pose)
        reference_poses = reference_pose.unsqueeze(0).repeat(len(points), 1, 1)

        true_error, true_valid = compute_epipolar_sampson_errors(
            true_pose,
            current_uv,
            reference_uv,
            reference_poses,
            focal=focal,
            centre=centre,
        )
        wrong_error, wrong_valid = compute_epipolar_sampson_errors(
            wrong_pose,
            current_uv,
            reference_uv,
            reference_poses,
            focal=focal,
            centre=centre,
        )

        self.assertTrue(bool(true_valid.all()))
        self.assertTrue(bool(wrong_valid.all()))
        self.assertLess(float(true_error.max()), 1e-4)
        self.assertGreater(float(wrong_error.median()), 0.5)

    def test_balanced_selection_is_deterministic_and_caps_dominant_reference(self):
        errors = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.1, 1.2, 1.3]
        )
        confidence = torch.ones(12)
        uv = torch.tensor(
            [
                [5.0, 5.0],
                [25.0, 5.0],
                [45.0, 5.0],
                [65.0, 5.0],
                [5.0, 45.0],
                [25.0, 45.0],
                [45.0, 45.0],
                [65.0, 45.0],
                [8.0, 8.0],
                [28.0, 8.0],
                [48.0, 48.0],
                [68.0, 48.0],
            ]
        )
        ref_ids = torch.tensor([1] * 8 + [2] * 4)
        valid = torch.ones(12, dtype=torch.bool)

        first, debug = select_balanced_correspondence_indices(
            errors,
            confidence,
            uv,
            ref_ids,
            valid,
            width=80,
            height=80,
            max_points=6,
            grid_rows=2,
            grid_cols=4,
            max_reference_fraction=0.5,
        )
        second, _ = select_balanced_correspondence_indices(
            errors,
            confidence,
            uv,
            ref_ids,
            valid,
            width=80,
            height=80,
            max_points=6,
            grid_rows=2,
            grid_cols=4,
            max_reference_fraction=0.5,
        )

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(first), 6)
        self.assertLessEqual(int((ref_ids[first] == 1).sum()), 3)
        self.assertLessEqual(int((ref_ids[first] == 2).sum()), 3)
        self.assertEqual(debug["selected_reference_count"], 2)
        self.assertGreaterEqual(debug["selected_cell_count"], 4)

    def test_pose_interpolation_uses_rotation_geodesic_and_camera_centres(self):
        initial = _pose()
        candidate = _pose(rz_deg=20.0)
        candidate_centre = torch.tensor([2.0, 0.0, 0.0])
        candidate[:3, 3] = -(candidate[:3, :3] @ candidate_centre)

        midpoint = interpolate_world_to_camera_pose(initial, candidate, 0.5)
        midpoint_centre = -(midpoint[:3, :3].transpose(0, 1) @ midpoint[:3, 3])
        correction = pose_correction_magnitude(initial, midpoint)

        self.assertAlmostEqual(correction["rotation_deg"], 10.0, places=3)
        self.assertTrue(
            torch.allclose(midpoint_centre, torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                interpolate_world_to_camera_pose(initial, candidate, 0.0),
                initial,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                interpolate_world_to_camera_pose(initial, candidate, 1.0),
                candidate,
                atol=1e-6,
            )
        )

    def test_reference_holdout_keeps_complete_reference_groups_disjoint(self):
        ref_ids = torch.tensor([10, 10, 10, 10, 20, 20, 20, 30, 30, 30])
        uv = torch.stack(
            [torch.arange(10, dtype=torch.float32), torch.zeros(10)], dim=-1
        )

        solve, validation, debug = split_pose_verification_evidence(
            ref_ids,
            uv,
            torch.ones(10, dtype=torch.bool),
            width=100,
            height=80,
            min_solve_support=4,
            min_validation_support=3,
        )

        self.assertEqual(debug["strategy"], "reference_holdout")
        self.assertTrue(debug["valid"])
        self.assertFalse(bool((solve & validation).any()))
        self.assertTrue(torch.equal(solve | validation, torch.ones(10, dtype=torch.bool)))
        for ref_id in torch.unique(ref_ids):
            group = ref_ids == ref_id
            self.assertTrue(bool((solve[group].all() or validation[group].all())))
        self.assertTrue(
            set(debug["solve_reference_ids"]).isdisjoint(
                set(debug["validation_reference_ids"])
            )
        )

    def test_single_reference_uses_explicit_spatial_holdout(self):
        uv = torch.tensor(
            [
                [10.0, 10.0],
                [30.0, 10.0],
                [50.0, 10.0],
                [70.0, 10.0],
                [10.0, 50.0],
                [30.0, 50.0],
                [50.0, 50.0],
                [70.0, 50.0],
            ]
        )

        solve, validation, debug = split_pose_verification_evidence(
            torch.full((8,), 7, dtype=torch.long),
            uv,
            torch.ones(8, dtype=torch.bool),
            width=80,
            height=80,
            min_solve_support=3,
            min_validation_support=3,
        )

        self.assertEqual(debug["strategy"], "single_reference_spatial_holdout")
        self.assertTrue(debug["valid"])
        self.assertGreaterEqual(int(solve.sum()), 3)
        self.assertGreaterEqual(int(validation.sum()), 3)
        self.assertFalse(bool((solve & validation).any()))
        self.assertTrue(torch.equal(solve | validation, torch.ones(8, dtype=torch.bool)))

    def test_reference_holdout_reports_invalid_when_validation_is_too_small(self):
        ref_ids = torch.tensor([10, 10, 10, 10, 10, 20])
        uv = torch.stack(
            [torch.arange(6, dtype=torch.float32), torch.zeros(6)], dim=-1
        )

        solve, validation, debug = split_pose_verification_evidence(
            ref_ids,
            uv,
            torch.ones(6, dtype=torch.bool),
            width=100,
            height=80,
            min_solve_support=4,
            min_validation_support=2,
        )

        self.assertFalse(debug["valid"])
        self.assertEqual(debug["reason"], "insufficient_independent_support")
        self.assertFalse(bool((solve & validation).any()))

    def test_reprojection_uses_world_to_camera_pose_and_rejects_negative_depth(self):
        xyz = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
        )
        observed = torch.tensor(
            [[50.0, 50.0], [101.0, 50.0], [50.0, 50.0]],
            dtype=torch.float32,
        )

        errors, valid = compute_reprojection_errors(
            _pose(),
            xyz,
            observed,
            focal=torch.tensor([100.0]),
            centre=torch.tensor([50.0, 50.0]),
        )

        self.assertTrue(torch.equal(valid, torch.tensor([True, True, False])))
        self.assertAlmostEqual(float(errors[0]), 0.0, places=5)
        self.assertAlmostEqual(float(errors[1]), 1.0, places=5)
        self.assertTrue(torch.isinf(errors[2]))

    def test_mad_filter_removes_local_outlier(self):
        errors = torch.tensor([0.2, 0.3, 0.4, 0.5, 12.0])
        uv = torch.tensor(
            [[5.0, 5.0], [8.0, 6.0], [12.0, 8.0], [14.0, 10.0], [10.0, 10.0]]
        )

        mask, debug = select_robust_correspondences(
            errors,
            uv,
            torch.ones(5, dtype=torch.bool),
            width=100,
            height=80,
            mad_scale=2.5,
            max_cutoff=10.0,
            min_support=4,
            grid_rows=4,
            grid_cols=5,
        )

        self.assertTrue(torch.equal(mask, torch.tensor([True, True, True, True, False])))
        self.assertEqual(debug["selected_count"], 4)
        self.assertGreater(debug["rejected_count"], 0)

    def test_filter_preserves_best_correspondence_in_each_occupied_cell(self):
        errors = torch.tensor([0.1, 0.2, 8.0])
        uv = torch.tensor([[5.0, 5.0], [8.0, 8.0], [95.0, 75.0]])

        mask, debug = select_robust_correspondences(
            errors,
            uv,
            torch.ones(3, dtype=torch.bool),
            width=100,
            height=80,
            mad_scale=1.0,
            max_cutoff=10.0,
            min_support=2,
            grid_rows=4,
            grid_cols=5,
        )

        self.assertTrue(mask[2])
        self.assertEqual(debug["occupied_cells"], 2)
        self.assertEqual(debug["selected_cells"], 2)

    def test_filter_restores_lowest_errors_to_minimum_support(self):
        errors = torch.arange(1.0, 11.0)
        uv = torch.stack([torch.arange(10.0), torch.zeros(10)], dim=-1)

        mask, debug = select_robust_correspondences(
            errors,
            uv,
            torch.ones(10, dtype=torch.bool),
            width=100,
            height=80,
            mad_scale=0.0,
            max_cutoff=10.0,
            min_support=7,
            grid_rows=1,
            grid_cols=1,
        )

        self.assertEqual(int(mask.sum()), 7)
        self.assertTrue(torch.equal(torch.where(mask)[0], torch.arange(7)))
        self.assertEqual(debug["restored_for_min_support"], 2)

    def test_summary_reports_robust_distribution(self):
        stats = summarize_reprojection_errors(
            torch.tensor([1.0, 2.0, 3.0, float("inf")]),
            torch.tensor([True, True, True, False]),
        )

        self.assertEqual(stats["valid_count"], 3)
        self.assertAlmostEqual(stats["median"], 2.0)
        self.assertGreater(stats["p90"], 2.0)


class PoseVerificationAcceptanceTests(unittest.TestCase):
    def test_accepts_lower_residual_with_bounded_pose_correction(self):
        initial = _pose()
        refined = _pose(tx=0.01, rz_deg=1.0)
        correction = pose_correction_magnitude(initial, refined)

        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.5, "p90": 3.5, "valid_count": 96},
            correction=correction,
            max_rotation_deg=3.0,
            max_translation=0.05,
            min_relative_median_improvement=0.05,
            min_support_ratio=0.80,
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "accepted")
        self.assertGreater(decision["relative_median_improvement"], 0.20)

    def test_default_accepts_two_percent_median_gain_when_tail_also_improves(self):
        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.94, "p90": 3.95, "valid_count": 100},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        self.assertTrue(decision["accepted"])

    def test_accepts_small_p90_tolerance_when_mean_and_median_improve(self):
        decision = decide_pose_refinement(
            pre={"mean": 2.2, "median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"mean": 2.0, "median": 1.9, "p90": 4.02, "valid_count": 100},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        self.assertTrue(decision["accepted"])

    def test_rejects_mean_degradation_even_when_median_improves(self):
        decision = decide_pose_refinement(
            pre={"mean": 2.0, "median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"mean": 2.1, "median": 1.8, "p90": 4.0, "valid_count": 100},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "mean_degraded")

    def test_rejects_tail_degradation_even_when_median_improves(self):
        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.0, "p90": 4.5, "valid_count": 100},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "p90_degraded")

    def test_rejects_support_collapse(self):
        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.0, "p90": 3.0, "valid_count": 40},
            correction={"rotation_deg": 1.0, "translation": 0.01},
            max_rotation_deg=3.0,
            max_translation=0.05,
            min_support_ratio=0.80,
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "support_collapsed")

    def test_rejects_implausible_rotation_and_returns_exact_initial_pose(self):
        initial = _pose(tx=0.2)
        refined = _pose(tx=0.2, rz_deg=8.0)
        decision = decide_pose_refinement(
            pre={"median": 2.0, "p90": 4.0, "valid_count": 100},
            post={"median": 1.0, "p90": 3.0, "valid_count": 100},
            correction=pose_correction_magnitude(initial, refined),
            max_rotation_deg=3.0,
            max_translation=0.05,
        )

        selected = choose_verified_pose(initial, refined, decision)

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "rotation_correction_too_large")
        self.assertTrue(torch.equal(selected, initial))
        self.assertNotEqual(selected.data_ptr(), initial.data_ptr())


if __name__ == "__main__":
    unittest.main()
