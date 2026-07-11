import unittest
from types import SimpleNamespace

import torch

from scene.keyframe import Keyframe


class _Desc:
    def __init__(self):
        self.kpts = torch.zeros(3, 2, device="cuda")

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
