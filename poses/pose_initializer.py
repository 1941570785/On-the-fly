# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

# 位姿初始化器，用于初始化关键帧的位姿
# 参考自：https://github.com/verlab/accelerated_features


import torch
import math

from poses.feature_detector import DescribedKeypoints
from poses.mini_ba import MiniBA
from poses.triangulator import matches_to_points
from utils import fov2focal, depth2points, sixD2mtx
from scene.keyframe import Keyframe
from poses.ransac import RANSACEstimator, EstimatorType

class PoseInitializer():
    """
    【位姿估计模块】位姿初始化器
    
    负责两种姿态初始化模式：
    1. Bootstrap模式：同时估计多个关键帧的初始位姿和焦距
    2. 增量模式：使用PnP-RANSAC和Mini-BA估计新关键帧的位姿
    
    使用Mini-BA（小规模Bundle Adjustment）进行快速优化。
    """
    def __init__(self, width, height, triangulator, matcher, max_pnp_error, args):
        """
        【位姿估计模块】初始化位姿初始化器
        
        Args:
            width: 图像宽度
            height: 图像高度
            triangulator: 三角化器
            matcher: 特征匹配器
            max_pnp_error: PnP-RANSAC的最大误差
            args: 训练参数
        """
        # 相机尺寸与模块引用
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher

        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')
        self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        self.num_kpts = args.num_kpts

        self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.num_pts_miniba_incr
        self.min_num_inliers = args.min_num_inliers

        # Initialize the focal length
        # 选择初始焦距：优先用户给定，其次 FOV，最后默认 0.7*width
        if args.init_focal > 0:
            self.f_init = args.init_focal
        elif args.init_fov > 0:
            self.f_init = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            self.f_init = 0.7 * width

        # Initialize MiniBA models
        self.miniba_bootstrap = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  not args.fix_focal, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniba_rebooting = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  False, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniBA_incr = MiniBA(
            1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,
            make_cuda_graph=True, iters=args.iters_miniba_incr)
        
        self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)
        self.last_incremental_debug: dict[str, object] = {}
        self.last_recovery_pose_outcome_fix: dict[str, object] = {}
        self.last_recovery_2d3d_support: dict[str, object] = {}
        self.last_recovery_pnp_consensus: dict[str, object] = {}
        self.last_recovery_ref_subset: list[dict[str, object]] = []
        self.recovery_defer_source_frame_id: int = -1
        self._last_pnp_Rt: torch.Tensor | None = None
        self.recovery_miniba_retry_min_2d3d = 500
        self.recovery_miniba_retry_min_pnp_inliers = 20
        self.recovery_consensus_target_refs = 10
        self.recovery_consensus_max_refs = 12
        self.recovery_consensus_min_refs = 6
        self.recovery_consensus_min_total_valid_2d3d = 120
        self.recovery_probe_min_inlier_ratio = 0.03

    def build_problem(self,
                      desc_kpts_list: list[DescribedKeypoints],
                      npts: int,
                      n_cams: int,
                      n_primary_cam: int,
                      min_n_matches: int,
                      kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras."""
        # 将多视角匹配组织成 miniBA 所需的 uvs / xyz_indices
        npts_per_primary_cam = npts // n_primary_cam
        uvs = torch.zeros(npts, n_cams, 2, device='cuda') - 1
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int64, device='cuda') - 1
        unused_kpts_mask = torch.ones((n_cams, desc_kpts_list[0].kpts.shape[0]), device='cuda', dtype=torch.bool)
        for k in range(n_primary_cam):
            # 统计当前主视角与其他视角的匹配出现次数
            idx_occurrences = torch.zeros(self.num_kpts, device="cuda", dtype=torch.int)
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= unused_kpts_mask[k]
            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue
            idx_occurrences = idx_occurrences > 0
            selected_indices = torch.multinomial(idx_occurrences.float(), npts_per_primary_cam, replacement=False)

            selected_mask = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
            selected_mask[selected_indices] = True
            aligned_ids = torch.arange(npts_per_primary_cam, device="cuda")
            all_aligned_ids = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
            all_aligned_ids[selected_indices] = aligned_ids

            uvs_k = uvs[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam, :, :]
            xyz_indices_k = xyz_indices[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam]
            for l in range(n_cams):
                if l == k:
                    # 主视角自身的关键点坐标直接填充
                    uvs_k[:, l, :] = desc_kpts_list[l].kpts[selected_indices]
                    xyz_indices_k[:, l] = selected_indices
                else:
                    lId = kfId_list[l]
                    if lId in desc_kpts_list[k].matches:
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        mask = selected_mask[idxk] 
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        # 将主视角关键点与其他视角对齐到同一 3D 点槽位
                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l, idxl] = False
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts[idxl]
                        xyz_indices_k[set_idx, l] = idxl

                        selected_indices_l = idxl.clone()
                        selected_mask_l = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
                        selected_mask_l[selected_indices_l] = True
                        all_aligned_ids_l = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
                        all_aligned_ids_l[selected_indices_l] = set_idx.clone()

                        for m in range(l + 1, n_cams):
                            mId = kfId_list[m]
                            if mId in desc_kpts_list[l].matches:
                                idxl = desc_kpts_list[l].matches[mId].idx
                                idxm = desc_kpts_list[l].matches[mId].idx_other

                                mask = selected_mask_l[idxl] 
                                idxl = idxl[mask]
                                idxm = idxm[mask]

                                set_idx = all_aligned_ids_l[idxl]
                                set_mask = uvs_k[set_idx, m, 0] == -1
                                # 仅填充未被占用的槽位
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[idxm[set_mask]]

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        # 若某个 3D 点有效匹配过少则剔除
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        return uvs, xyz_indices

    @torch.no_grad()
    def initialize_bootstrap(self, desc_kpts_list: list[DescribedKeypoints], rebooting=False):
        """
        【位姿估计模块】Bootstrap位姿初始化
        
        同时估计多个关键帧的初始位姿和焦距。
        使用Mini-BA进行联合优化，确保所有位姿和焦距的一致性。
        
        Args:
            desc_kpts_list: 关键帧的描述关键点列表
            rebooting: 是否为重启模式（重启时不优化焦距）
        
        Returns:
            Rts: 估计的位姿矩阵列表 [N, 4, 4]
            f: 估计的焦距
            final_residual: 最终残差（用于验证收敛性）
        """
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        ## Exhaustive matching
        # 全连接匹配，以获得稳定的多视角约束
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                _ = self.matcher(desc_kpts_list[i], desc_kpts_list[j], remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
        
        ## Build the problem by organizing matches
        uvs, xyz_indices = self.build_problem(desc_kpts_list, npts, n_cams, n_cams, 2, list(range(n_cams)))

        ## Initialize for miniBA (poses at identity, 3D points with rand depth)
        # 3D 点用单位深度回投影初始化，带随机缩放
        f_init = (torch.tensor([self.f_init], device="cuda"))
        Rs6D_init = torch.eye(3, 2, device="cuda")[None].repeat(n_cams, 1, 1)
        ts_init = torch.zeros(n_cams, 3, device="cuda")

        xyz_init = torch.zeros(npts, 3, device="cuda")
        for k in range(n_cams):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            xyz_init[mask] += depth2points(uvs[mask, k, :], 1, f_init, self.centre)
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()

        ## Run miniBA, estimating 3D points, camera focal and poses
        # rebooting 时不再优化焦距
        if rebooting:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_rebooting(Rs6D_init, ts_init, self.f, xyz_init, self.centre, uvs.view(-1))
        else:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_bootstrap(Rs6D_init, ts_init, f_init, xyz_init, self.centre, uvs.view(-1))
        final_residual = (r * mask).abs().sum()/mask.sum()

        self.f = f
        self.intrinsics = torch.cat([f, self.centre], dim=0)

        ## Scale to 0.1 average translation
        # 归一化尺度，避免尺度漂移
        rel_ts = ts[:-1] - ts[1:]
        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        ts *= scale
        xyz = scale * xyz.clone()
        Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts[:, :3, :3] = sixD2mtx(Rs6D)
        Rts[:, :3, 3] = ts

        return Rts, f, final_residual

    @torch.no_grad()
    def initialize_incremental(self, keyframes: list[Keyframe], curr_desc_kpts: DescribedKeypoints, index: int, is_test: bool, curr_img):
        """
        【位姿估计模块】增量位姿初始化
        
        使用历史关键帧估计新关键帧的位姿。
        流程：
        1. 匹配当前帧与历史关键帧
        2. 使用PnP-RANSAC估计初始位姿
        3. 使用Mini-BA优化位姿
        
        Args:
            keyframes: 历史关键帧列表
            curr_desc_kpts: 当前帧的描述关键点
            index: 当前帧索引
            is_test: 是否为测试帧
            curr_img: 当前图像（未使用，保留接口）
        
        Returns:
            Rt: 估计的位姿矩阵 [4, 4]，如果失败返回None
        """
        self.last_incremental_debug = {
            "failure_reason": "",
            "num_2d3d_correspondences": 0,
            "num_pnp_inliers": 0,
            "num_miniba_inliers": 0,
            "ref_keyframe_ids": [],
            "ref_source_frame_ids": [],
            "ref_commit_origin": [],
            "ref_is_recovery": [],
            "ref_is_seed": [],
            "ref_is_support_eligible": [],
            "match_count_by_ref": [],
            "match_count_total": 0,
            "match_count_to_seed_keyframes": 0,
            "best_match_keyframe_id": -1,
            "best_match_is_seed": False,
            "best_match_num_matches": 0,
            "pnp_ref_keyframe_ids": [],
            "pnp_ref_source_frame_ids": [],
            "pnp_ref_contains_seed": False,
            "miniba_ref_keyframe_ids": [],
            "miniba_ref_source_frame_ids": [],
            "miniba_ref_contains_seed": False,
        }

        # Match the current frame with previous keyframes
        # 收集可用于 PnP 的 2D-3D 对应
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        corr_ref_ids = []
        for keyframe in keyframes:
            # 匹配当前帧与历史关键帧并过滤外点
            matches = self.matcher(curr_desc_kpts, keyframe.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=index, kID_other=keyframe.index)

            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            valid_count = int(mask.sum().item())
            source_frame_id = int(keyframe.info.get("_paper_aligned_source_frame_id", keyframe.index))
            commit_origin = str(keyframe.info.get("_paper_aligned_commit_origin", "unknown"))
            is_recovery = commit_origin in {"true_recovery_commit", "early_seed_recovery_commit"}
            is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
            self.last_incremental_debug["ref_keyframe_ids"].append(int(keyframe.index))
            self.last_incremental_debug["ref_source_frame_ids"].append(source_frame_id)
            self.last_incremental_debug["ref_commit_origin"].append(commit_origin)
            self.last_incremental_debug["ref_is_recovery"].append(is_recovery)
            self.last_incremental_debug["ref_is_seed"].append(is_seed)
            self.last_incremental_debug["ref_is_support_eligible"].append(
                bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
            )
            self.last_incremental_debug["match_count_by_ref"].append(valid_count)
            self.last_incremental_debug["match_count_total"] += valid_count
            if is_seed:
                self.last_incremental_debug["match_count_to_seed_keyframes"] += valid_count
            if valid_count > int(self.last_incremental_debug["best_match_num_matches"]):
                self.last_incremental_debug["best_match_num_matches"] = valid_count
                self.last_incremental_debug["best_match_keyframe_id"] = int(keyframe.index)
                self.last_incremental_debug["best_match_is_seed"] = is_seed
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])
            corr_ref_ids.append(torch.full((valid_count,), int(keyframe.index), device="cuda", dtype=torch.long))

        if len(xyz) == 0:
            self.last_incremental_debug["failure_reason"] = "no_2d3d_correspondences"
            return None
        xyz = torch.cat(xyz, dim=0)
        uvs = torch.cat(uvs, dim=0)
        confs = torch.cat(confs, dim=0)
        match_indices = torch.cat(match_indices, dim=0)
        corr_ref_ids = torch.cat(corr_ref_ids, dim=0)
        self.last_incremental_debug["num_2d3d_correspondences"] = int(len(xyz))

        # Subsample the points if there are too many
        # 先按置信度采样控制 PnP 输入规模
        if len(xyz) > self.num_pts_pnpransac:
            # 按置信度随机下采样，避免单帧点过多
            selected_indices = torch.multinomial(confs, self.num_pts_miniba_incr, replacement=False)
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]
            corr_ref_ids = corr_ref_ids[selected_indices]

        # Estimate an initial camera pose and inliers using PnP RANSAC
        # 使用上一关键帧作为初始位姿
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        if len(xyz) < 4:
            self.last_incremental_debug["failure_reason"] = "insufficient_correspondences_for_pnp"
            return None
        try:
            Rt, inliers = self.PnPRANSAC(uvs, xyz, self.f, self.centre, Rs6D_init, ts_init, confs)
        except Exception:
            self.last_incremental_debug["failure_reason"] = "pnp_ransac_exception"
            return None

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]
        corr_ref_ids = corr_ref_ids[inliers]
        self.last_incremental_debug["num_pnp_inliers"] = int(len(xyz))
        pnp_ref_ids = sorted({int(x) for x in corr_ref_ids.detach().cpu().tolist()})
        self.last_incremental_debug["pnp_ref_keyframe_ids"] = pnp_ref_ids
        self.last_incremental_debug["pnp_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        ]
        self.last_incremental_debug["pnp_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        )
        if len(xyz) < 4:
            self.last_incremental_debug["failure_reason"] = "pnp_inliers_too_few"
            return None

        # Subsample the points if there are too many
        # 为 miniBA 填充固定数量的点
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(torch.rand_like(xyz[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False)[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
            miniba_ref_ids_tensor = corr_ref_ids[selected_indices]
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)
            miniba_ref_ids_tensor = corr_ref_ids
        miniba_ref_ids = sorted({int(x) for x in miniba_ref_ids_tensor.detach().cpu().tolist()})
        self.last_incremental_debug["miniba_ref_keyframe_ids"] = miniba_ref_ids
        self.last_incremental_debug["miniba_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        ]
        self.last_incremental_debug["miniba_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        )

        # Run the initialization
        # 以 PnP 结果为初始化，执行小规模 BA 微调
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        self.last_incremental_debug["num_miniba_inliers"] = int(mask.sum().item())
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]

        # Check if we have sufficiently many inliers
        # 训练阶段要求足够内点以避免错误注册
        if is_test or mask.sum() > self.min_num_inliers:
            # Return the pose of the current frame
            self.last_incremental_debug["failure_reason"] = ""
            return Rt
        else:
            print("Too few inliers for pose initialization")
            self.last_incremental_debug["failure_reason"] = "miniba_inliers_too_few"
            # Remove matches as we prevent the current frame from being registered
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None

    def _keyframe_has_pt3d_count(self, keyframe: Keyframe) -> int:
        return int(keyframe.desc_kpts.has_pt3d.sum().item())

    def _sort_recovery_reference_keyframes(
        self, keyframes: list[Keyframe], curr_desc_kpts: DescribedKeypoints, index: int
    ) -> list[Keyframe]:
        scored: list[tuple[int, int, int, int, Keyframe]] = []
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                kID=index,
                kID_other=keyframe.index,
            )
            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            valid_count = int(mask.sum().item())
            raw_count = int(len(matches.kpts))
            has_pt3d_total = self._keyframe_has_pt3d_count(keyframe)
            is_seed = int(bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False)))
            is_support = int(
                bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
            )
            scored.append((has_pt3d_total, valid_count, is_support, is_seed, raw_count, keyframe))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4]))
        return [item[5] for item in scored]

    def _ref_conf_weight_from_probe_stat(self, stat: dict[str, object]) -> float:
        ratio = float(stat.get("ref_pnp_inlier_ratio", 0.0) or 0.0)
        valid = float(stat.get("ref_valid_2d3d_count", 0) or 0.0)
        score = float(stat.get("ref_consensus_score", 0.0) or 0.0)
        return float(
            max(
                0.12,
                min(
                    1.0,
                    0.20 + 0.35 * min(score, 2.0) / 2.0 + 0.30 * min(ratio * 12.0, 1.0) + 0.15 * min(valid / 200.0, 1.0),
                ),
            )
        )

    def _per_ref_correspondence_cap(self, stat: dict[str, object]) -> int | None:
        valid = int(stat.get("ref_valid_2d3d_count", 0) or 0)
        if valid <= 0:
            return 0
        ratio = float(stat.get("ref_pnp_inlier_ratio", 0.0) or 0.0)
        has_pt3d = int(stat.get("ref_has_pt3d_count", 0) or 0)
        if has_pt3d > 5000 and ratio < 0.01:
            return max(16, int(valid * 0.08))
        if ratio < 0.02:
            return max(24, int(valid * min(0.35, 0.10 + ratio * 10.0)))
        return None

    def _collect_recovery_correspondences(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        Rt_guess_for_triangulation: torch.Tensor | None,
        ref_conf_weights: dict[int, float] | None = None,
        ref_correspondence_caps: dict[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        support_debug: dict[str, object] = {
            "raw_2d2d_match_count": 0,
            "verified_2d2d_match_count": 0,
            "has_pt3d_match_count": 0,
            "temporary_3d_support_used": False,
            "temporary_3d_support_count": 0,
            "valid_2d3d_from_direct_refs": 0,
            "valid_2d3d_from_recovery_refs": 0,
            "valid_2d3d_from_seed_refs": 0,
            "per_ref_3d_bearing_counts": [],
            "per_ref_raw_match_counts": [],
            "selected_refs_by_3d_support": [int(kf.index) for kf in keyframes],
        }
        xyz: list[torch.Tensor] = []
        uvs: list[torch.Tensor] = []
        confs: list[torch.Tensor] = []
        corr_ref_ids: list[torch.Tensor] = []
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                kID=index,
                kID_other=keyframe.index,
            )
            raw_count = int(len(matches.kpts))
            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            valid_count = int(mask.sum().item())
            commit_origin = str(keyframe.info.get("_paper_aligned_commit_origin", "unknown"))
            is_recovery = commit_origin in {"true_recovery_commit", "early_seed_recovery_commit"}
            is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
            support_debug["raw_2d2d_match_count"] = int(support_debug["raw_2d2d_match_count"]) + raw_count
            support_debug["has_pt3d_match_count"] = int(support_debug["has_pt3d_match_count"]) + valid_count
            support_debug["per_ref_3d_bearing_counts"].append(valid_count)
            support_debug["per_ref_raw_match_counts"].append(raw_count)
            if is_recovery:
                support_debug["valid_2d3d_from_recovery_refs"] = int(
                    support_debug["valid_2d3d_from_recovery_refs"]
                ) + valid_count
            elif is_seed:
                support_debug["valid_2d3d_from_seed_refs"] = int(support_debug["valid_2d3d_from_seed_refs"]) + valid_count
            else:
                support_debug["valid_2d3d_from_direct_refs"] = int(
                    support_debug["valid_2d3d_from_direct_refs"]
                ) + valid_count
            if valid_count == 0:
                continue
            ref_pts_conf = keyframe.desc_kpts.pts_conf[matches.idx_other[mask]]
            ref_xyz = keyframe.desc_kpts.pts3d[matches.idx_other[mask]]
            ref_uvs = matches.kpts[mask]
            kid = int(keyframe.index)
            if ref_correspondence_caps is not None and kid in ref_correspondence_caps:
                cap = int(ref_correspondence_caps[kid])
                if cap <= 0:
                    continue
                if valid_count > cap:
                    top_idx = torch.topk(ref_pts_conf, cap, largest=True).indices
                    ref_xyz = ref_xyz[top_idx]
                    ref_uvs = ref_uvs[top_idx]
                    ref_pts_conf = ref_pts_conf[top_idx]
                    valid_count = cap
            if ref_conf_weights is not None:
                ref_pts_conf = ref_pts_conf * float(ref_conf_weights.get(kid, 0.5))
            xyz.append(ref_xyz)
            uvs.append(ref_uvs)
            confs.append(ref_pts_conf)
            corr_ref_ids.append(
                torch.full((valid_count,), kid, device="cuda", dtype=torch.long)
            )
        if Rt_guess_for_triangulation is not None:
            extra_xyz, extra_uvs, extra_confs = self._triangulate_recovery_correspondences(
                keyframes, curr_desc_kpts, index, Rt_guess_for_triangulation
            )
            if len(extra_xyz) > 0:
                support_debug["temporary_3d_support_used"] = True
                support_debug["temporary_3d_support_count"] = int(len(extra_xyz))
                xyz.append(extra_xyz)
                uvs.append(extra_uvs)
                confs.append(extra_confs)
                corr_ref_ids.append(
                    torch.full((len(extra_xyz),), int(keyframes[0].index), device="cuda", dtype=torch.long)
                )
        if not xyz:
            support_debug["verified_2d2d_match_count"] = 0
            return (
                torch.zeros(0, 3, device="cuda"),
                torch.zeros(0, 2, device="cuda"),
                torch.zeros(0, device="cuda"),
                torch.zeros(0, device="cuda", dtype=torch.long),
                support_debug,
            )
        xyz_cat = torch.cat(xyz, dim=0)
        uvs_cat = torch.cat(uvs, dim=0)
        confs_cat = torch.cat(confs, dim=0)
        corr_ref_ids_cat = torch.cat(corr_ref_ids, dim=0)
        support_debug["verified_2d2d_match_count"] = int(len(xyz_cat))
        return xyz_cat, uvs_cat, confs_cat, corr_ref_ids_cat, support_debug

    def _run_pose_from_correspondences(
        self,
        keyframes: list[Keyframe],
        xyz_cat: torch.Tensor,
        uvs_cat: torch.Tensor,
        confs_cat: torch.Tensor,
        corr_ref_ids_cat: torch.Tensor,
        index: int,
        is_test: bool,
    ) -> torch.Tensor | None:
        self.last_incremental_debug["num_2d3d_correspondences"] = int(len(xyz_cat))
        if len(xyz_cat) < 4:
            self.last_incremental_debug["failure_reason"] = "insufficient_correspondences_for_pnp"
            return None
        if len(xyz_cat) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(
                confs_cat, min(self.num_pts_pnpransac, len(xyz_cat)), replacement=False
            )
            xyz_cat = xyz_cat[selected_indices]
            uvs_cat = uvs_cat[selected_indices]
            confs_cat = confs_cat[selected_indices]
            corr_ref_ids_cat = corr_ref_ids_cat[selected_indices]
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        try:
            Rt, inliers = self.PnPRANSAC(uvs_cat, xyz_cat, self.f, self.centre, Rs6D_init, ts_init, confs_cat)
        except Exception:
            self.last_incremental_debug["failure_reason"] = "pnp_ransac_exception"
            return None
        xyz_cat = xyz_cat[inliers]
        uvs_cat = uvs_cat[inliers]
        confs_cat = confs_cat[inliers]
        corr_ref_ids_cat = corr_ref_ids_cat[inliers]
        self.last_incremental_debug["num_pnp_inliers"] = int(len(xyz_cat))
        pnp_ref_ids = sorted({int(x) for x in corr_ref_ids_cat.detach().cpu().tolist()})
        self.last_incremental_debug["pnp_ref_keyframe_ids"] = pnp_ref_ids
        self.last_incremental_debug["pnp_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        ]
        self.last_incremental_debug["pnp_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        )
        if len(xyz_cat) < 4:
            self.last_incremental_debug["failure_reason"] = "pnp_inliers_too_few"
            self._last_pnp_Rt = None
            return None
        self._last_pnp_Rt = Rt.clone()
        if len(xyz_cat) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(
                torch.rand_like(xyz_cat[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False
            )[1]
            xyz_ba = xyz_cat[selected_indices]
            uvs_ba = uvs_cat[selected_indices]
            miniba_ref_ids_tensor = corr_ref_ids_cat[selected_indices]
        else:
            xyz_ba = torch.cat(
                [xyz_cat, torch.zeros(self.num_pts_miniba_incr - len(xyz_cat), 3, device="cuda")], dim=0
            )
            uvs_ba = torch.cat(
                [uvs_cat, -torch.ones(self.num_pts_miniba_incr - len(uvs_cat), 2, device="cuda")], dim=0
            )
            miniba_ref_ids_tensor = corr_ref_ids_cat
        miniba_ref_ids = sorted({int(x) for x in miniba_ref_ids_tensor.detach().cpu().tolist()})
        self.last_incremental_debug["miniba_ref_keyframe_ids"] = miniba_ref_ids
        self.last_incremental_debug["miniba_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        ]
        self.last_incremental_debug["miniba_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        )
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(
            Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1)
        )
        self.last_incremental_debug["num_miniba_inliers"] = int(mask.sum().item())
        Rt_out = torch.eye(4, device="cuda")
        Rt_out[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt_out[:3, 3] = ts[0]
        if is_test or mask.sum() > self.min_num_inliers:
            self.last_incremental_debug["failure_reason"] = ""
            return Rt_out
        self.last_incremental_debug["failure_reason"] = "miniba_inliers_too_few"
        for keyframe in keyframes:
            keyframe.desc_kpts.matches.pop(index, None)
        return None

    def _triangulate_recovery_correspondences(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        Rt_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        extra_xyz: list[torch.Tensor] = []
        extra_uvs: list[torch.Tensor] = []
        extra_confs: list[torch.Tensor] = []
        max_error = float(self.triangulator.max_error.item())
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                kID=index,
                kID_other=keyframe.index,
            )
            if len(matches.kpts) == 0:
                continue
            missing = ~keyframe.desc_kpts.has_pt3d[matches.idx_other]
            if int(missing.sum().item()) == 0:
                continue
            uv_src = matches.kpts[missing]
            uv_ref = matches.kpts_other[missing]
            Rt_ref = keyframe.get_Rt()
            rel_Rt = Rt_source @ torch.linalg.inv_ex(Rt_ref)[0]
            pts3d, dis, err = matches_to_points(
                uv_src, uv_ref, rel_Rt[:3, :3], rel_Rt[:3, 3], self.f, self.centre
            )
            valid = (
                (pts3d[:, 2] > 1e-6)
                & (dis > 0.0)
                & (err < max_error)
                & (~pts3d.isnan().any(dim=-1))
            )
            if int(valid.sum().item()) == 0:
                continue
            extra_xyz.append(pts3d[valid])
            extra_uvs.append(uv_src[valid])
            extra_confs.append(torch.ones(int(valid.sum().item()), device="cuda"))
        if not extra_xyz:
            return (
                torch.zeros(0, 3, device="cuda"),
                torch.zeros(0, 2, device="cuda"),
                torch.zeros(0, device="cuda"),
            )
        return torch.cat(extra_xyz, dim=0), torch.cat(extra_uvs, dim=0), torch.cat(extra_confs, dim=0)

    def _run_incremental_pose_core(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
    ) -> torch.Tensor | None:
        self.last_incremental_debug = {
            "failure_reason": "",
            "num_2d3d_correspondences": 0,
            "num_pnp_inliers": 0,
            "num_miniba_inliers": 0,
            "ref_keyframe_ids": [],
            "ref_source_frame_ids": [],
            "ref_commit_origin": [],
            "ref_is_recovery": [],
            "ref_is_seed": [],
            "ref_is_support_eligible": [],
            "match_count_by_ref": [],
            "match_count_total": 0,
            "match_count_to_seed_keyframes": 0,
            "best_match_keyframe_id": -1,
            "best_match_is_seed": False,
            "best_match_num_matches": 0,
            "pnp_ref_keyframe_ids": [],
            "pnp_ref_source_frame_ids": [],
            "pnp_ref_contains_seed": False,
            "miniba_ref_keyframe_ids": [],
            "miniba_ref_source_frame_ids": [],
            "miniba_ref_contains_seed": False,
        }
        xyz: list[torch.Tensor] = []
        uvs: list[torch.Tensor] = []
        confs: list[torch.Tensor] = []
        match_indices: list[torch.Tensor] = []
        corr_ref_ids: list[torch.Tensor] = []
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all",
                kID=index,
                kID_other=keyframe.index,
            )
            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            valid_count = int(mask.sum().item())
            source_frame_id = int(keyframe.info.get("_paper_aligned_source_frame_id", keyframe.index))
            commit_origin = str(keyframe.info.get("_paper_aligned_commit_origin", "unknown"))
            is_recovery = commit_origin in {"true_recovery_commit", "early_seed_recovery_commit"}
            is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
            self.last_incremental_debug["ref_keyframe_ids"].append(int(keyframe.index))
            self.last_incremental_debug["ref_source_frame_ids"].append(source_frame_id)
            self.last_incremental_debug["ref_commit_origin"].append(commit_origin)
            self.last_incremental_debug["ref_is_recovery"].append(is_recovery)
            self.last_incremental_debug["ref_is_seed"].append(is_seed)
            self.last_incremental_debug["ref_is_support_eligible"].append(
                bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
            )
            self.last_incremental_debug["match_count_by_ref"].append(valid_count)
            self.last_incremental_debug["match_count_total"] += valid_count
            if is_seed:
                self.last_incremental_debug["match_count_to_seed_keyframes"] += valid_count
            if valid_count > int(self.last_incremental_debug["best_match_num_matches"]):
                self.last_incremental_debug["best_match_num_matches"] = valid_count
                self.last_incremental_debug["best_match_keyframe_id"] = int(keyframe.index)
                self.last_incremental_debug["best_match_is_seed"] = is_seed
            if valid_count == 0:
                continue
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])
            corr_ref_ids.append(
                torch.full((valid_count,), int(keyframe.index), device="cuda", dtype=torch.long)
            )

        if len(xyz) == 0:
            self.last_incremental_debug["failure_reason"] = "no_2d3d_correspondences"
            return None
        xyz_cat = torch.cat(xyz, dim=0)
        uvs_cat = torch.cat(uvs, dim=0)
        confs_cat = torch.cat(confs, dim=0)
        match_indices_cat = torch.cat(match_indices, dim=0)
        corr_ref_ids_cat = torch.cat(corr_ref_ids, dim=0)
        self.last_incremental_debug["num_2d3d_correspondences"] = int(len(xyz_cat))

        if len(xyz_cat) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(confs_cat, self.num_pts_miniba_incr, replacement=False)
            xyz_cat = xyz_cat[selected_indices]
            uvs_cat = uvs_cat[selected_indices]
            confs_cat = confs_cat[selected_indices]
            match_indices_cat = match_indices_cat[selected_indices]
            corr_ref_ids_cat = corr_ref_ids_cat[selected_indices]

        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        if len(xyz_cat) < 4:
            self.last_incremental_debug["failure_reason"] = "insufficient_correspondences_for_pnp"
            return None
        try:
            Rt, inliers = self.PnPRANSAC(uvs_cat, xyz_cat, self.f, self.centre, Rs6D_init, ts_init, confs_cat)
        except Exception:
            self.last_incremental_debug["failure_reason"] = "pnp_ransac_exception"
            return None

        xyz_cat = xyz_cat[inliers]
        uvs_cat = uvs_cat[inliers]
        confs_cat = confs_cat[inliers]
        match_indices_cat = match_indices_cat[inliers]
        corr_ref_ids_cat = corr_ref_ids_cat[inliers]
        self.last_incremental_debug["num_pnp_inliers"] = int(len(xyz_cat))
        pnp_ref_ids = sorted({int(x) for x in corr_ref_ids_cat.detach().cpu().tolist()})
        self.last_incremental_debug["pnp_ref_keyframe_ids"] = pnp_ref_ids
        self.last_incremental_debug["pnp_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        ]
        self.last_incremental_debug["pnp_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in pnp_ref_ids
        )
        if len(xyz_cat) < 4:
            self.last_incremental_debug["failure_reason"] = "pnp_inliers_too_few"
            self._last_pnp_Rt = None
            return None

        self._last_pnp_Rt = Rt.clone()

        if len(xyz_cat) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(
                torch.rand_like(xyz_cat[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False
            )[1]
            xyz_ba = xyz_cat[selected_indices]
            uvs_ba = uvs_cat[selected_indices]
            miniba_ref_ids_tensor = corr_ref_ids_cat[selected_indices]
        else:
            xyz_ba = torch.cat(
                [xyz_cat, torch.zeros(self.num_pts_miniba_incr - len(xyz_cat), 3, device="cuda")], dim=0
            )
            uvs_ba = torch.cat(
                [uvs_cat, -torch.ones(self.num_pts_miniba_incr - len(uvs_cat), 2, device="cuda")], dim=0
            )
            miniba_ref_ids_tensor = corr_ref_ids_cat
        miniba_ref_ids = sorted({int(x) for x in miniba_ref_ids_tensor.detach().cpu().tolist()})
        self.last_incremental_debug["miniba_ref_keyframe_ids"] = miniba_ref_ids
        self.last_incremental_debug["miniba_ref_source_frame_ids"] = [
            int(kf.info.get("_paper_aligned_source_frame_id", kf.index))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        ]
        self.last_incremental_debug["miniba_ref_contains_seed"] = any(
            bool(kf.info.get("_paper_aligned_is_v7_early_seed", False))
            for kf in keyframes
            if int(kf.index) in miniba_ref_ids
        )

        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        self.last_incremental_debug["num_miniba_inliers"] = int(mask.sum().item())
        Rt_out = torch.eye(4, device="cuda")
        Rt_out[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt_out[:3, 3] = ts[0]
        if is_test or mask.sum() > self.min_num_inliers:
            self.last_incremental_debug["failure_reason"] = ""
            return Rt_out
        self.last_incremental_debug["failure_reason"] = "miniba_inliers_too_few"
        for keyframe in keyframes:
            keyframe.desc_kpts.matches.pop(index, None)
        return None

    def _ref_source_frame_id(self, keyframe: Keyframe) -> int:
        return int(keyframe.info.get("_paper_aligned_source_frame_id", keyframe.index))

    def _ref_anchor_id(self, keyframe: Keyframe) -> int:
        return int(keyframe.info.get("_paper_aligned_anchor_id", -1))

    @torch.no_grad()
    def _probe_ref_pnp_inliers(
        self,
        keyframe: Keyframe,
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        init_keyframe: Keyframe,
    ) -> tuple[int, int, float]:
        matches = self.matcher(
            curr_desc_kpts,
            keyframe.desc_kpts,
            remove_outliers=True,
            update_kpts_flag="all",
            kID=index,
            kID_other=keyframe.index,
        )
        mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
        valid_count = int(mask.sum().item())
        if valid_count < 4:
            return 0, valid_count, 0.0
        xyz = keyframe.desc_kpts.pts3d[matches.idx_other[mask]]
        uvs = matches.kpts[mask]
        confs = keyframe.desc_kpts.pts_conf[matches.idx_other[mask]]
        try:
            _, inliers = self.PnPRANSAC(
                uvs,
                xyz,
                self.f,
                self.centre,
                init_keyframe.rW2C,
                init_keyframe.tW2C,
                confs,
            )
            inlier_count = int(inliers.sum().item())
        except Exception:
            return 0, valid_count, 0.0
        ratio = float(inlier_count) / max(valid_count, 1)
        return inlier_count, valid_count, ratio

    def _recovery_ref_consensus_score(
        self,
        keyframe: Keyframe,
        valid_2d3d: int,
        pnp_inliers: int,
        pnp_ratio: float,
        defer_source_frame_id: int,
        anchor_mode_id: int,
    ) -> float:
        has_pt3d_total = self._keyframe_has_pt3d_count(keyframe)
        src_dist = abs(self._ref_source_frame_id(keyframe) - int(defer_source_frame_id))
        dist_weight = 1.0 / (1.0 + float(src_dist) / 50.0)
        anchor_match = 1.0 if self._ref_anchor_id(keyframe) == anchor_mode_id and anchor_mode_id >= 0 else 0.5
        is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
        is_support = bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False))
        seed_bonus = 0.05 if is_seed or is_support else 0.0
        high_pt3d_penalty = 0.25 if has_pt3d_total > 3000 and pnp_ratio < self.recovery_probe_min_inlier_ratio else 0.0
        return (
            0.35 * min(float(valid_2d3d) / 500.0, 2.0)
            + 0.45 * min(pnp_ratio * 10.0, 2.0)
            + 0.10 * dist_weight
            + 0.10 * anchor_match
            + seed_bonus
            - high_pt3d_penalty
        )

    def _select_coherent_ref_subset(
        self,
        keyframes: list[Keyframe],
        ref_stats: list[dict[str, object]],
        defer_source_frame_id: int,
    ) -> tuple[list[Keyframe], list[dict[str, object]], list[int], list[str], str]:
        if not keyframes:
            return [], [], [], [], "no_candidate_refs"
        stat_by_kid = {int(s.get("ref_keyframe_id", -1)): s for s in ref_stats}
        ranked = sorted(
            zip(keyframes, ref_stats),
            key=lambda item: -float(item[1].get("consensus_score", 0.0)),
        )
        selected: list[Keyframe] = []
        subset_trace: list[dict[str, object]] = []
        excluded_ids: list[int] = []
        excluded_reasons: list[str] = []
        has_direct_or_support = False

        def is_support_ref(keyframe: Keyframe) -> bool:
            return bool(keyframe.info.get("_paper_aligned_support_eligible_recovery_keyframe", False)) or str(
                keyframe.info.get("_paper_aligned_commit_origin", "")
            ) in {"direct_admit", "direct"}

        for keyframe, stat in ranked:
            kid = int(keyframe.index)
            ratio = float(stat.get("ref_pnp_inlier_ratio", 0.0) or 0.0)
            valid = int(stat.get("ref_valid_2d3d_count", 0) or 0)
            pnp_inl = int(stat.get("ref_pnp_inlier_count", 0) or 0)
            has_pt3d = int(stat.get("ref_has_pt3d_count", 0) or 0)
            is_seed = bool(keyframe.info.get("_paper_aligned_is_v7_early_seed", False))
            is_support = is_support_ref(keyframe)
            extreme_noise = has_pt3d > 6000 and ratio < 0.005 and pnp_inl == 0
            if extreme_noise:
                excluded_ids.append(kid)
                excluded_reasons.append("extreme_high_haspt3d_zero_inlier")
                subset_trace.append(
                    {**stat, "ref_selected": False, "ref_exclusion_reason": "extreme_high_haspt3d_zero_inlier"}
                )
                continue
            if len(selected) >= self.recovery_consensus_target_refs:
                excluded_ids.append(kid)
                excluded_reasons.append("beyond_target_ref_count")
                subset_trace.append(
                    {**stat, "ref_selected": False, "ref_exclusion_reason": "beyond_target_ref_count"}
                )
                continue
            soft_ok = valid >= 8 or pnp_inl >= 2 or is_support or is_seed
            if not soft_ok and valid < 4:
                excluded_ids.append(kid)
                excluded_reasons.append("insufficient_probe_support")
                subset_trace.append(
                    {**stat, "ref_selected": False, "ref_exclusion_reason": "insufficient_probe_support"}
                )
                continue
            selected.append(keyframe)
            if is_support or is_seed:
                has_direct_or_support = True
            subset_trace.append({**stat, "ref_selected": True, "ref_exclusion_reason": ""})

        if not has_direct_or_support:
            for keyframe, stat in ranked:
                if is_support_ref(keyframe):
                    if keyframe not in selected:
                        selected.insert(0, keyframe)
                        subset_trace.append({**stat, "ref_selected": True, "ref_exclusion_reason": "promoted_support_ref"})
                    break

        if len(selected) < self.recovery_consensus_min_refs:
            for keyframe, stat in ranked:
                if keyframe in selected:
                    continue
                selected.append(keyframe)
                subset_trace.append({**stat, "ref_selected": True, "ref_exclusion_reason": "filled_to_min_refs"})
                if len(selected) >= self.recovery_consensus_min_refs:
                    break

        selected = selected[: self.recovery_consensus_max_refs]
        selected_ids = {int(kf.index) for kf in selected}
        for keyframe, stat in ranked:
            kid = int(keyframe.index)
            if kid in selected_ids:
                continue
            if len(subset_trace) >= len(keyframes) * 2:
                break
        selection_reason = "top_scored_soft_filter_with_probe_weights"
        return selected, subset_trace, excluded_ids, excluded_reasons, selection_reason

    def _run_pnp_inlier_count_only(
        self,
        keyframes: list[Keyframe],
        xyz_cat: torch.Tensor,
        uvs_cat: torch.Tensor,
        confs_cat: torch.Tensor,
    ) -> int:
        if len(xyz_cat) < 4:
            return 0
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        try:
            _, inliers = self.PnPRANSAC(uvs_cat, xyz_cat, self.f, self.centre, Rs6D_init, ts_init, confs_cat)
            return int(inliers.sum().item())
        except Exception:
            return 0

    def _init_recovery_incremental_debug(self, keyframes: list[Keyframe]) -> None:
        self.last_incremental_debug = {
            "failure_reason": "",
            "num_2d3d_correspondences": 0,
            "num_pnp_inliers": 0,
            "num_miniba_inliers": 0,
            "ref_keyframe_ids": [int(kf.index) for kf in keyframes],
            "ref_source_frame_ids": [
                int(kf.info.get("_paper_aligned_source_frame_id", kf.index)) for kf in keyframes
            ],
            "ref_commit_origin": [str(kf.info.get("_paper_aligned_commit_origin", "unknown")) for kf in keyframes],
            "ref_is_recovery": [
                str(kf.info.get("_paper_aligned_commit_origin", ""))
                in {"true_recovery_commit", "early_seed_recovery_commit"}
                for kf in keyframes
            ],
            "ref_is_seed": [bool(kf.info.get("_paper_aligned_is_v7_early_seed", False)) for kf in keyframes],
            "ref_is_support_eligible": [
                bool(kf.info.get("_paper_aligned_support_eligible_recovery_keyframe", False)) for kf in keyframes
            ],
            "match_count_by_ref": list(self.last_recovery_2d3d_support.get("per_ref_3d_bearing_counts", []) or []),
            "match_count_total": int(self.last_recovery_2d3d_support.get("verified_2d2d_match_count", 0) or 0),
            "match_count_to_seed_keyframes": int(
                self.last_recovery_2d3d_support.get("valid_2d3d_from_seed_refs", 0) or 0
            ),
            "best_match_keyframe_id": int(keyframes[0].index) if keyframes else -1,
            "best_match_is_seed": bool(keyframes[0].info.get("_paper_aligned_is_v7_early_seed", False))
            if keyframes
            else False,
            "best_match_num_matches": int(
                max(self.last_recovery_2d3d_support.get("per_ref_3d_bearing_counts", [0]) or [0])
            ),
            "pnp_ref_keyframe_ids": [],
            "pnp_ref_source_frame_ids": [],
            "pnp_ref_contains_seed": False,
            "miniba_ref_keyframe_ids": [],
            "miniba_ref_source_frame_ids": [],
            "miniba_ref_contains_seed": False,
        }

    @torch.no_grad()
    def initialize_incremental_recovery(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
        curr_img,
    ):
        sorted_keyframes = self._sort_recovery_reference_keyframes(keyframes, curr_desc_kpts, index)
        defer_sid = int(getattr(self, "recovery_defer_source_frame_id", -1))
        init_kf = sorted_keyframes[0]
        anchor_kf = max(sorted_keyframes, key=self._keyframe_has_pt3d_count)
        anchor_mode_id = self._ref_anchor_id(anchor_kf)
        Rt_guess = anchor_kf.get_Rt().clone()

        xyz_all, uvs_all, confs_all, _, support_all = self._collect_recovery_correspondences(
            sorted_keyframes, curr_desc_kpts, index, Rt_guess
        )
        pnp_before = self._run_pnp_inlier_count_only(sorted_keyframes, xyz_all, uvs_all, confs_all)
        valid_before = int(len(xyz_all))
        ratio_before = float(pnp_before) / max(valid_before, 1)

        ref_stats: list[dict[str, object]] = []
        for keyframe in sorted_keyframes:
            pnp_inl, valid_2d3d, pnp_ratio = self._probe_ref_pnp_inliers(
                keyframe, curr_desc_kpts, index, init_kf
            )
            score = self._recovery_ref_consensus_score(
                keyframe, valid_2d3d, pnp_inl, pnp_ratio, defer_sid, anchor_mode_id
            )
            ref_stats.append(
                {
                    "ref_keyframe_id": int(keyframe.index),
                    "ref_source_frame_id": self._ref_source_frame_id(keyframe),
                    "ref_commit_origin": str(keyframe.info.get("_paper_aligned_commit_origin", "")),
                    "ref_anchor_id": self._ref_anchor_id(keyframe),
                    "ref_valid_2d3d_count": valid_2d3d,
                    "ref_pnp_inlier_count": pnp_inl,
                    "ref_pnp_inlier_ratio": round(pnp_ratio, 6),
                    "ref_consensus_score": round(score, 6),
                    "ref_has_pt3d_count": self._keyframe_has_pt3d_count(keyframe),
                    "ref_source_distance_to_defer": abs(self._ref_source_frame_id(keyframe) - defer_sid),
                }
            )

        selected, subset_trace, excluded_ids, excluded_reasons, subset_selection_reason = (
            self._select_coherent_ref_subset(sorted_keyframes, ref_stats, defer_sid)
        )
        self.last_recovery_ref_subset = list(subset_trace)
        stat_by_kid = {int(s.get("ref_keyframe_id", -1)): s for s in ref_stats}
        selected_stats = [stat_by_kid[int(kf.index)] for kf in selected if int(kf.index) in stat_by_kid]
        selected_valid_sum = sum(int(s.get("ref_valid_2d3d_count", 0) or 0) for s in selected_stats)
        selected_pnp_sum = sum(int(s.get("ref_pnp_inlier_count", 0) or 0) for s in selected_stats)
        max_selected_pnp = max((int(s.get("ref_pnp_inlier_count", 0) or 0) for s in selected_stats), default=0)
        coherent = bool(
            len(selected) >= self.recovery_consensus_min_refs
            and selected_valid_sum >= self.recovery_consensus_min_total_valid_2d3d
            and (selected_pnp_sum >= 8 or max_selected_pnp >= 4)
        )
        selection_reason = subset_selection_reason if coherent else "no_coherent_subset"

        self.last_recovery_pnp_consensus = {
            "candidate_ref_count": len(sorted_keyframes),
            "candidate_ref_ids": [int(kf.index) for kf in sorted_keyframes],
            "candidate_ref_valid_2d3d_counts": [int(s["ref_valid_2d3d_count"]) for s in ref_stats],
            "candidate_ref_pnp_inlier_counts": [int(s["ref_pnp_inlier_count"]) for s in ref_stats],
            "candidate_ref_pnp_inlier_ratios": [float(s["ref_pnp_inlier_ratio"]) for s in ref_stats],
            "candidate_ref_anchor_ids": [int(s["ref_anchor_id"]) for s in ref_stats],
            "candidate_ref_source_distances": [int(s["ref_source_distance_to_defer"]) for s in ref_stats],
            "selected_consensus_ref_ids": [int(kf.index) for kf in selected],
            "selected_consensus_ref_count": len(selected),
            "selection_reason": selection_reason,
            "excluded_ref_ids": excluded_ids,
            "excluded_ref_reasons": excluded_reasons,
            "coherent_subset_found": coherent,
            "pnp_inliers_before_consensus": pnp_before,
            "pnp_inlier_ratio_before_consensus": round(ratio_before, 6),
            "valid_2d3d_before_consensus": valid_before,
        }

        if not coherent:
            self.last_recovery_2d3d_support = dict(support_all)
            self._init_recovery_incremental_debug(sorted_keyframes)
            self.last_incremental_debug["failure_reason"] = "no_coherent_ref_subset"
            self.last_recovery_pnp_consensus["failure_reason_after_consensus"] = "no_coherent_ref_subset"
            self.last_recovery_pose_outcome_fix = {
                "recovery_pose_mode": True,
                "correspondence_flow_fix_applied": False,
                "recovery_miniba_retry_applied": False,
                "failure_reason_after_fix": "no_coherent_ref_subset",
            }
            return None

        ref_conf_weights = {
            int(kf.index): self._ref_conf_weight_from_probe_stat(stat_by_kid[int(kf.index)])
            for kf in selected
            if int(kf.index) in stat_by_kid
        }
        ref_correspondence_caps: dict[int, int] = {}
        for kf in selected:
            kid = int(kf.index)
            if kid not in stat_by_kid:
                continue
            cap = self._per_ref_correspondence_cap(stat_by_kid[kid])
            if cap is not None:
                ref_correspondence_caps[kid] = int(cap)
        xyz_cat, uvs_cat, confs_cat, corr_ref_ids_cat, support_debug = self._collect_recovery_correspondences(
            selected,
            curr_desc_kpts,
            index,
            Rt_guess,
            ref_conf_weights=ref_conf_weights,
            ref_correspondence_caps=ref_correspondence_caps or None,
        )
        self.last_recovery_2d3d_support = dict(support_debug)
        self._init_recovery_incremental_debug(selected)
        self.last_recovery_pose_outcome_fix = {
            "recovery_pose_mode": True,
            "handoff_fix_applied": True,
            "reference_consistency_fix_applied": True,
            "pnp_geometric_consensus_fix_applied": True,
            "correspondence_flow_fix_applied": bool(support_debug.get("temporary_3d_support_used", False)),
            "recovery_miniba_retry_applied": False,
            "triangulation_augmented_correspondence_count": int(
                support_debug.get("temporary_3d_support_count", 0) or 0
            ),
            "miniba_success_before_fix": False,
            "miniba_inliers_before_fix": 0,
            "miniba_success_after_fix": False,
            "miniba_inliers_after_fix": 0,
            "failure_reason_before_fix": "",
            "failure_reason_after_fix": "",
        }
        Rt = self._run_pose_from_correspondences(
            selected, xyz_cat, uvs_cat, confs_cat, corr_ref_ids_cat, index, is_test
        )
        after_debug = dict(self.last_incremental_debug)
        pnp_after = int(after_debug.get("num_pnp_inliers", 0) or 0)
        valid_after = int(after_debug.get("num_2d3d_correspondences", 0) or 0)
        ratio_after = float(pnp_after) / max(valid_after, 1)
        self.last_recovery_pnp_consensus.update(
            {
                "pnp_inliers_after_consensus": pnp_after,
                "pnp_inlier_ratio_after_consensus": round(ratio_after, 6),
                "miniba_inliers_before_consensus": 0,
                "miniba_inliers_after_consensus": int(after_debug.get("num_miniba_inliers", 0) or 0),
                "miniba_success_after_consensus": bool(Rt is not None),
                "recovery_success_after_consensus": bool(Rt is not None),
                "failure_reason_after_consensus": str(after_debug.get("failure_reason", "") or ""),
            }
        )
        self.last_recovery_pose_outcome_fix["miniba_success_before_fix"] = bool(Rt is not None)
        self.last_recovery_pose_outcome_fix["miniba_inliers_before_fix"] = int(
            after_debug.get("num_miniba_inliers", 0) or 0
        )
        if Rt is not None:
            self.last_recovery_pose_outcome_fix["miniba_success_after_fix"] = True
            self.last_recovery_pose_outcome_fix["miniba_inliers_after_fix"] = int(
                after_debug.get("num_miniba_inliers", 0) or 0
            )
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = ""
            return Rt

        failure_reason = str(after_debug.get("failure_reason", "") or "")
        if (
            failure_reason != "miniba_inliers_too_few"
            or pnp_after < self.recovery_miniba_retry_min_pnp_inliers
            or valid_after < self.recovery_miniba_retry_min_2d3d
        ):
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = failure_reason
            return None

        self.last_recovery_pose_outcome_fix["recovery_miniba_retry_applied"] = True
        Rt_seed = self._last_pnp_Rt
        if Rt_seed is None:
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = "pnp_pose_unavailable"
            return None
        xyz_retry, uvs_retry, confs_retry, corr_retry, retry_support = self._collect_recovery_correspondences(
            selected,
            curr_desc_kpts,
            index,
            Rt_seed,
            ref_conf_weights=ref_conf_weights,
            ref_correspondence_caps=ref_correspondence_caps or None,
        )
        if len(xyz_retry) < self.recovery_miniba_retry_min_2d3d or pnp_after < self.recovery_miniba_retry_min_pnp_inliers:
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = "recovery_retry_gates_not_met"
            return None
        self._init_recovery_incremental_debug(selected)
        Rt_retry = self._run_pose_from_correspondences(
            selected, xyz_retry, uvs_retry, confs_retry, corr_retry, index, is_test
        )
        if Rt_retry is not None:
            self.last_recovery_pose_outcome_fix["miniba_success_after_fix"] = True
            self.last_recovery_pose_outcome_fix["miniba_inliers_after_fix"] = int(
                self.last_incremental_debug.get("num_miniba_inliers", 0) or 0
            )
            self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = ""
            return Rt_retry
        self.last_recovery_pose_outcome_fix["failure_reason_after_fix"] = str(
            self.last_incremental_debug.get("failure_reason", "") or "miniba_inliers_too_few"
        )
        return None