import unittest
from types import SimpleNamespace

import torch

from scene.keyframe import Keyframe


class _Desc:
    def __init__(self):
        self.kpts = torch.zeros(3, 2, device="cuda")
        self.pts3d = torch.arange(
            9, dtype=torch.float32, device="cuda"
        ).view(3, 3)
        self.pts_conf = torch.tensor(
            [0.2, 0.6, 0.9], dtype=torch.float32, device="cuda"
        )
        self.has_pt3d = torch.tensor(
            [True, False, True], dtype=torch.bool, device="cuda"
        )

    def update_3D_pts(self, *_args, **_kwargs):
        pass

    def to(self, *_args, **_kwargs):
        pass


class _Ref:
    def __init__(self, index):
        self.index = index
        self.info = {}

    def get_Rt(self):
        rt = torch.eye(4, device="cuda")
        rt[0, 3] = float(self.index)
        return rt


class _Triangulator:
    n_cams = 3

    def __init__(self, chosen_kfs_ids=None):
        self.chosen_kfs_ids = chosen_kfs_ids or [1, 1, 0]
        self.rts_others = None
        self.uvs_others = None

    def prepare_matches(self, _desc):
        uv = torch.zeros(3, 2, device="cuda")
        uvs_others = torch.arange(18, device="cuda", dtype=torch.float32).view(3, 3, 2)
        return uv, uvs_others, list(self.chosen_kfs_ids)

    def __call__(self, _uv, uvs_others, _rt, rts_others, _f, _centre):
        self.uvs_others = uvs_others.detach().clone()
        self.rts_others = rts_others.detach().clone()
        new_pts = torch.zeros(3, 3, device="cuda")
        depth = torch.ones(3, device="cuda")
        best_dis = torch.zeros(3, device="cuda")
        valid = torch.zeros(3, device="cuda", dtype=torch.bool)
        return new_pts, depth, best_dis, valid


class KeyframeBaselineResolutionTests(unittest.TestCase):
    def test_frozen_pose_geometry_snapshot_is_opt_in_and_immutable(self):
        keyframe = Keyframe.__new__(Keyframe)
        keyframe.desc_kpts = _Desc()
        keyframe._pose_verification_geometry_snapshot_mode = "frozen_first_valid_v1"
        keyframe._pose_verification_geometry_snapshot = None

        captured = keyframe.capture_pose_verification_geometry_snapshot()
        original_points = keyframe.desc_kpts.pts3d.clone()
        original_confidence = keyframe.desc_kpts.pts_conf.clone()
        original_mask = keyframe.desc_kpts.has_pt3d.clone()
        keyframe.desc_kpts.pts3d.add_(100.0)
        keyframe.desc_kpts.pts_conf.zero_()
        keyframe.desc_kpts.has_pt3d.logical_not_()

        points, confidence, mask = keyframe.get_pose_verification_geometry()

        self.assertTrue(captured)
        self.assertTrue(torch.equal(points, original_points))
        self.assertTrue(torch.equal(confidence, original_confidence))
        self.assertTrue(torch.equal(mask, original_mask))

    def test_disabled_pose_geometry_snapshot_reads_live_points(self):
        keyframe = Keyframe.__new__(Keyframe)
        keyframe.desc_kpts = _Desc()
        keyframe._pose_verification_geometry_snapshot_mode = "off"
        keyframe._pose_verification_geometry_snapshot = None

        self.assertFalse(keyframe.capture_pose_verification_geometry_snapshot())
        keyframe.desc_kpts.pts3d.add_(7.0)
        points, confidence, mask = keyframe.get_pose_verification_geometry()

        self.assertIs(points, keyframe.desc_kpts.pts3d)
        self.assertIs(confidence, keyframe.desc_kpts.pts_conf)
        self.assertIs(mask, keyframe.desc_kpts.has_pt3d)

    def test_guarded_snapshot_captures_the_same_geometry_pose(self):
        keyframe = Keyframe.__new__(Keyframe)
        keyframe.desc_kpts = _Desc()
        keyframe._pose_verification_geometry_snapshot_mode = (
            "guarded_frozen_v3"
        )
        keyframe._pose_verification_geometry_snapshot = None
        keyframe._pose_verification_geometry_snapshot_Rt = None
        initial_rt = torch.eye(4, device="cuda")
        initial_rt[1, 3] = 0.25
        expected_rt = initial_rt.clone()
        keyframe._pose_verification_geometry_initial_Rt = initial_rt

        self.assertTrue(keyframe.capture_pose_verification_geometry_snapshot())
        captured_rt = keyframe.get_pose_verification_geometry_Rt()
        keyframe._pose_verification_geometry_initial_Rt[1, 3] = 9.0

        self.assertTrue(torch.equal(captured_rt, expected_rt))
        self.assertAlmostEqual(float(captured_rt[1, 3]), 0.25)

    def test_guarded_live_pose_mode_still_captures_immutable_global_points(self):
        keyframe = Keyframe.__new__(Keyframe)
        keyframe.desc_kpts = _Desc()
        keyframe._pose_verification_geometry_snapshot_mode = (
            "guarded_frozen_live_pose_v4"
        )
        keyframe._pose_verification_geometry_snapshot = None
        keyframe._pose_verification_geometry_snapshot_Rt = None
        keyframe._pose_verification_geometry_initial_Rt = torch.eye(
            4,
            device="cuda",
        )

        self.assertTrue(keyframe.capture_pose_verification_geometry_snapshot())
        frozen_points = keyframe.get_pose_verification_geometry()[0]
        keyframe.desc_kpts.pts3d.add_(50.0)

        self.assertFalse(torch.equal(frozen_points, keyframe.desc_kpts.pts3d))

    def test_homogeneous_guard_mode_captures_immutable_global_points(self):
        keyframe = Keyframe.__new__(Keyframe)
        keyframe.desc_kpts = _Desc()
        keyframe._pose_verification_geometry_snapshot_mode = (
            "guarded_frozen_homogeneous_v5"
        )
        keyframe._pose_verification_geometry_snapshot = None
        keyframe._pose_verification_geometry_snapshot_Rt = None
        keyframe._pose_verification_geometry_initial_Rt = torch.eye(
            4,
            device="cuda",
        )

        self.assertTrue(keyframe.capture_pose_verification_geometry_snapshot())
        self.assertIsNotNone(keyframe.get_pose_verification_geometry())

    def test_global_support_guard_captures_immutable_global_points(self):
        keyframe = Keyframe.__new__(Keyframe)
        keyframe.desc_kpts = _Desc()
        keyframe._pose_verification_geometry_snapshot_mode = (
            "frozen_global_support_guard_v6"
        )
        keyframe._pose_verification_geometry_snapshot = None
        keyframe._pose_verification_geometry_snapshot_Rt = None
        keyframe._pose_verification_geometry_initial_Rt = torch.eye(
            4,
            device="cuda",
        )

        self.assertTrue(keyframe.capture_pose_verification_geometry_snapshot())
        self.assertIsNotNone(keyframe.get_pose_verification_geometry())

    def test_baseline_resolution_preserves_duplicate_chosen_keyframe_order(self):
        triangulator = _Triangulator()
        keyframe = Keyframe.__new__(Keyframe)
        keyframe.desc_kpts = _Desc()
        keyframe.latest_invdepth = None
        keyframe.triangulator = triangulator
        keyframe.f = 100.0
        keyframe.centre = torch.zeros(2, device="cuda")
        keyframe.info = {"image_name": "current"}
        keyframe.last_chosen_kfs_resolution = {}
        keyframe.get_Rt = lambda: torch.eye(4, device="cuda")

        refs = [_Ref(0), _Ref(1)]

        keyframe.update_3dpts(refs, resolution_mode="baseline")

        self.assertIsNotNone(triangulator.rts_others)
        self.assertEqual(
            [float(x) for x in triangulator.rts_others[:, 0, 3].detach().cpu()],
            [1.0, 1.0, 0.0],
        )

    def test_baseline_resolution_accepts_keyframe_ids_without_list_indexing(self):
        triangulator = _Triangulator(chosen_kfs_ids=[41, 40, 41])
        keyframe = Keyframe.__new__(Keyframe)
        keyframe.desc_kpts = _Desc()
        keyframe.latest_invdepth = None
        keyframe.triangulator = triangulator
        keyframe.f = 100.0
        keyframe.centre = torch.zeros(2, device="cuda")
        keyframe.info = {"image_name": "current"}
        keyframe.last_chosen_kfs_resolution = {}
        keyframe.get_Rt = lambda: torch.eye(4, device="cuda")

        refs = [_Ref(40), _Ref(41)]

        keyframe.update_3dpts(refs, resolution_mode="baseline")

        self.assertIsNotNone(triangulator.rts_others)
        self.assertEqual(
            [float(x) for x in triangulator.rts_others[:, 0, 3].detach().cpu()],
            [41.0, 40.0, 41.0],
        )


if __name__ == "__main__":
    unittest.main()
