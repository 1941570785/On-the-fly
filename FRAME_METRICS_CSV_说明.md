# frame_metrics.csv 字段说明

本文档说明 `frame_metrics.csv` 中各列的含义。该文件通常位于各场景的 `results/<dataset>/<scene>/frame_metrics.csv`，记录逐帧的渲染质量、位姿估计及评估指标。

---

## 字段总览表

| 列名 | 类型 | 含义 | 备注 |
|------|------|------|------|
| **dataset_name** | str | 数据集名称 | 如 `StaticHikes`、`MipNeRF360`、`TUM` |
| **scene_name** | str | 场景名称 | 如 `university2`、`garden`、`desk1` |
| **frame_idx** | int | 关键帧索引 | 在关键帧序列中的顺序（0, 1, 2, ...） |
| **original_frame_idx** | int | 原始帧索引 | 对应原始图像序列中的帧号（可能因跳帧而不连续） |
| **original_image_name** | str | 原始图像文件名 | 如 `1.jpg`、`11.jpg` |
| **render_image_name** | str | 渲染输出文件名 | 测试帧有值（如 `1.jpg`），训练帧通常为空 |
| **sequence_order** | int | 序列顺序 | 该帧在输入序列中的顺序位置 |
| **is_test_view** | bool | 是否为测试视角 | `True`：holdout 测试帧；`False`：训练帧 |
| **is_keyframe** | bool | 是否为关键帧 | 被选为关键帧的帧为 `True` |
| **is_registered** | bool | 是否已注册 | 位姿是否已成功估计并注册到场景 |
| **registration_status** | str | 注册状态 | 如 `registered`，表示位姿已成功注册 |
| **pose_stage** | str | 位姿阶段 | 如 `final`，表示最终优化后的位姿 |
| **pose_format** | str | 位姿格式 | `w2c`：世界到相机（World-to-Camera）变换矩阵 |
| **split** | str | 数据划分 | `test` 或 `train`，与 `is_test_view` 对应 |
| **est_r00** ~ **est_r22** | float | 估计旋转矩阵元素 | 3×3 旋转矩阵 R 的 9 个元素（行优先） |
| **est_tx** | float | 估计平移 X | 相机位姿平移向量的 x 分量 |
| **est_ty** | float | 估计平移 Y | 相机位姿平移向量的 y 分量 |
| **est_tz** | float | 估计平移 Z | 相机位姿平移向量的 z 分量 |
| **psnr** | float | 峰值信噪比 | 渲染质量指标，越高越好；仅测试帧有值 |
| **ssim** | float | 结构相似性 | 渲染质量指标，范围 [0,1]，越高越好；仅测试帧有值 |
| **lpips** | float | 感知损失 | 渲染质量指标，越低越好；仅测试帧有值 |
| **abs_trans_error** | float | 绝对平移误差 | 与 GT 位姿的平移误差（cm），需 COLMAP GT |
| **abs_rot_error_deg** | float | 绝对旋转误差（度） | 与 GT 位姿的旋转误差（°），需 COLMAP GT |
| **rel_trans_error** | float | 相对平移误差 | 与前一帧的相对平移误差 |
| **rel_rot_error_deg** | float | 相对旋转误差（度） | 与前一帧的相对旋转误差（°） |
| **output_dir** | str | 输出目录 | 该场景结果保存路径 |

---

## 分组说明

### 1. 元信息与标识

| 列名 | 含义 |
|------|------|
| dataset_name, scene_name | 数据集与场景标识 |
| frame_idx, original_frame_idx | 帧在关键帧序列与原始序列中的索引 |
| original_image_name, render_image_name | 输入图像与渲染输出文件名 |
| sequence_order | 在输入序列中的顺序 |
| is_test_view, is_keyframe, is_registered | 布尔标识 |
| registration_status, pose_stage, pose_format, split | 状态与格式描述 |

### 2. 位姿（6DoF）

| 列名 | 含义 |
|------|------|
| est_r00 ~ est_r22 | 旋转矩阵 R（3×3） |
| est_tx, est_ty, est_tz | 平移向量 t（3×1） |
| 合起来 | 4×4 外参矩阵 [R \| t; 0 0 0 1]，格式为 w2c |

### 3. 渲染质量（仅测试帧）

| 列名 | 含义 | 方向 |
|------|------|------|
| psnr | 峰值信噪比 (dB) | ↑ 越高越好 |
| ssim | 结构相似性 | ↑ 越高越好 |
| lpips | 学习感知图像块相似度 | ↓ 越低越好 |

### 4. 位姿误差（需 COLMAP GT 或前一帧）

| 列名 | 含义 |
|------|------|
| abs_trans_error | 相对 GT 的绝对平移误差（cm） |
| abs_rot_error_deg | 相对 GT 的绝对旋转误差（°） |
| rel_trans_error | 相对前一帧的平移误差 |
| rel_rot_error_deg | 相对前一帧的旋转误差（°） |

---

## 注意事项

1. **psnr / ssim / lpips**：只有 `is_test_view=True` 的帧有值，训练帧为空。
2. **abs_* 误差**：需要 `sparse/0` 下有 COLMAP 结果作为 GT 才会计算。
3. **rel_* 误差**：基于相邻帧位姿计算，不依赖 COLMAP。
4. **render_image_name**：测试帧对应 `test_images/` 中的文件名；训练帧通常为空。
