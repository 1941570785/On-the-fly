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
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from poses.mini_ba import MiniBA
from utils import fov2focal, depth2points, sixD2mtx, mtx2sixD, make_torch_sampler
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
        self.min_pnp_inliers = args.min_pnp_inliers
        self.pose_refine_iters = args.pose_refine_iters
        self.pose_refine_lr = args.pose_refine_lr
        self.use_pose_reprojection_loss = args.use_pose_reprojection_loss
        self.pose_reprojection_weight = args.pose_reprojection_weight
        self.use_pose_photometric_refine = args.use_pose_photometric_refine
        self.pose_photometric_weight = args.pose_photometric_weight
        self.pose_refine_downsample = args.pose_refine_downsample
        self.use_correspondence_guided_pose_init = args.use_correspondence_guided_pose_init

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

    @torch.no_grad()
    def _collect_rendered_correspondences(
        self,
        ref_keyframe: Keyframe,
        curr_desc_kpts: DescribedKeypoints,
        scene_model,
        curr_index: int,
    ):
        """
        Build extra 2D-3D correspondences from rendered depth on the previous keyframe.
        """
        if scene_model is None:
            return None, None, None, 0

        matches = self.matcher(
            curr_desc_kpts,
            ref_keyframe.desc_kpts,
            remove_outliers=True,
            update_kpts_flag="all",
            kID=curr_index,
            kID_other=ref_keyframe.index,
        )
        if len(matches.idx) == 0:
            return None, None, None, 0

        render_pkg = scene_model.render_from_id(ref_keyframe.index)
        ref_depth = 1.0 / render_pkg["invdepth"][0].clamp_min(1e-6)

        ref_uv = matches.kpts_other
        sampler = make_torch_sampler(ref_uv.view(1, 1, -1, 2), self.width, self.height)
        sampled_depth = F.grid_sample(
            ref_depth[None, None], sampler, mode="bilinear", align_corners=True
        )[0, 0, 0]

        valid = sampled_depth > 1e-6
        if valid.sum() < 4:
            return None, None, None, 0

        ref_uv = ref_uv[valid]
        curr_uv = matches.kpts[valid]
        depth = sampled_depth[valid]
        pts_cam = depth2points(ref_uv, depth.unsqueeze(-1), self.f, self.centre)
        pts_world = (pts_cam - ref_keyframe.get_t()) @ ref_keyframe.get_R()

        conf = torch.ones_like(depth) * 0.6
        return pts_world, curr_uv, conf, int(valid.sum().item())

    def _refine_pose_with_renderer(self, Rt, xyz, uvs, curr_img, scene_model):
        """
        Lightweight joint refinement branch:
        - reprojection loss on correspondences
        - optional photometric loss using the current renderer
        """
        if self.pose_refine_iters <= 0:
            return Rt

        with torch.enable_grad():
            rW2C = torch.nn.Parameter(mtx2sixD(Rt[:3, :3]).clone())
            tW2C = torch.nn.Parameter(Rt[:3, 3].clone())
            optimizer = torch.optim.Adam([rW2C, tW2C], lr=self.pose_refine_lr)

            target = curr_img.cuda()
            ds = max(1, int(self.pose_refine_downsample))
            if ds > 1:
                target = F.avg_pool2d(target[None], ds)[0]

            for _ in range(self.pose_refine_iters):
                optimizer.zero_grad()
                R = sixD2mtx(rW2C)
                xyz_cam = xyz @ R.T + tW2C[None]
                proj = self.f * xyz_cam[:, :2] / xyz_cam[:, 2:3].clamp_min(1e-6) + self.centre
                reproj = (proj - uvs).abs().mean()
                loss = self.pose_reprojection_weight * reproj

                if self.use_pose_photometric_refine and scene_model is not None:
                    Rt_tmp = torch.eye(4, device="cuda")
                    Rt_tmp[:3, :3] = R
                    Rt_tmp[:3, 3] = tW2C
                    view_matrix = Rt_tmp.transpose(0, 1)
                    render_pkg = scene_model.render(
                        self.width // ds,
                        self.height // ds,
                        view_matrix,
                        scaling_modifier=1,
                        bg=torch.zeros(3, device="cuda"),
                    )
                    rendered = render_pkg["render"]
                    photo = (rendered - target).abs().mean()
                    loss = loss + self.pose_photometric_weight * photo

                loss.backward()
                optimizer.step()

        refined = torch.eye(4, device="cuda")
        refined[:3, :3] = sixD2mtx(rW2C)
        refined[:3, 3] = tW2C
        return refined

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
    def initialize_incremental(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
        curr_img,
        scene_model=None,
    ):
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
        
        # Match the current frame with previous keyframes
        # 收集可用于 PnP 的 2D-3D 对应
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        stats = {
            "n_correspondences": 0,
            "n_corr_guided": 0,
            "pnp_inliers": 0,
            "pnp_success": False,
        }
        for keyframe in keyframes:
            # 匹配当前帧与历史关键帧并过滤外点
            matches = self.matcher(curr_desc_kpts, keyframe.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=index, kID_other=keyframe.index)

            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])

        if self.use_correspondence_guided_pose_init and len(keyframes) > 0:
            pts_w, uv_curr, corr_conf, n_corr = self._collect_rendered_correspondences(
                keyframes[0], curr_desc_kpts, scene_model, index
            )
            if pts_w is not None:
                xyz.append(pts_w)
                uvs.append(uv_curr)
                confs.append(corr_conf)
                match_indices.append(torch.zeros_like(corr_conf, dtype=torch.long))
                stats["n_corr_guided"] = n_corr

        if len(xyz) == 0:
            return None, stats

        xyz = torch.cat(xyz, dim=0)
        uvs = torch.cat(uvs, dim=0)
        confs = torch.cat(confs, dim=0)
        match_indices = torch.cat(match_indices, dim=0)
        stats["n_correspondences"] = int(xyz.shape[0])

        # Subsample the points if there are too many
        # 先按置信度采样控制 PnP 输入规模
        if len(xyz) > self.num_pts_pnpransac:
            # 按置信度随机下采样，避免单帧点过多
            selected_indices = torch.multinomial(confs, self.num_pts_miniba_incr, replacement=False)
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]

        # Estimate an initial camera pose and inliers using PnP RANSAC
        # 使用上一关键帧作为初始位姿
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        Rt, inliers = self.PnPRANSAC(uvs, xyz, self.f, self.centre, Rs6D_init, ts_init, confs)
        stats["pnp_inliers"] = int(inliers.sum().item())

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]

        # Subsample the points if there are too many
        # 为 miniBA 填充固定数量的点
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(torch.rand_like(xyz[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False)[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)

        # Run the initialization
        # 以 PnP 结果为初始化，执行小规模 BA 微调
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]

        if self.use_pose_reprojection_loss or self.use_pose_photometric_refine:
            Rt = self._refine_pose_with_renderer(Rt, xyz, uvs, curr_img, scene_model)

        # Check if we have sufficiently many inliers
        # 训练阶段要求足够内点以避免错误注册
        if is_test or (mask.sum() > self.min_num_inliers and stats["pnp_inliers"] >= self.min_pnp_inliers):
            # Return the pose of the current frame
            stats["pnp_success"] = True
            return Rt, stats
        else:
            print("Too few inliers for pose initialization")
            # Remove matches as we prevent the current frame from being registered
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None, stats