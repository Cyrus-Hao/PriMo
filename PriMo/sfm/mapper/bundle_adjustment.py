import numpy as np
import pyceres
import pycolmap
from pyceres import Problem
from pycolmap import LossFunctionType
from mpsfm.baseclass import BaseClass


def fit_robust_gaussian_mad(data):
    # 计算data的中位数 中位数对异常值没有均值这么敏感 更适合包含了离群点的数据
    mu = np.median(data)
    # 求数据点与中位数的偏差的绝对值 再进行取中位数 得到mad主要反映了数据的离群状况
    mad = np.median(np.abs(data - mu))
    # mad->高斯分布的标准差估计
    sigma = 1.4826 * mad
    # 中位数 标准差的输出
    return mu, sigma


class Optimizer(BaseClass):
    default_conf = {
        # 深度估计任务中的损失函数类型 log(1 + (x/scale)^2)
        "depth_loss_name": "cauchy",
        # 3D参考损失函数
        "ref3d_loss_name": "trivial",
        # 重投影损失函数 sqrt(x^2 + eps) - sqrt(eps) 对小误差表现类似 L2（平方损失），对大误差接近 L1（绝对值损失）
        "reproj_loss_name": "SOFT_L1",
        # 重投影损失缩放 调整 Soft L1 损失的平滑程度 较大会使损失对大误差更接近线性 较小的值会使损失更接近平方损失
        "reproj_loss_scale": 1.5,
        # 是否启用尺度过滤 True时仅保留满足 1/scale_filter_factor < div（obsdepths 观测深度 /projdepths 投影深度） < scale_filter_factor 的数据点 为了排除异常尺度
        "scale_filter": True,
        # 尺度过滤因子 表示允许深度比率在 [0.667, 1.5] 范围内
        "scale_filter_factor": 1.5,
        # 度量尺度过滤 确保深度图的尺度与真实世界单位一致
        "metric_scale_filter": True,
        # 鲁棒标准差倍数 暂时没看懂 下文应该有提
        "rob_std": 2,
        # 截断模式[quantile, mad] mad在上面 quantile是基于分位数截断 例如剔除超出 95% 分位数的数据
        "truncation_mode": "mad",
        # 是否启用额外机制处理严重异常值 为true时检查白化（数据除以标准差 使其标准化为均值 0、标准差 1 的分布）后的对数深度差 whitened < 3 剔除超出此阈值的数据点 当前不启用
        "gross_outliers": False,
        # 是否对所有图像应用单一尺度调整 当为true且当前图像（imid）不是参考图像（bundle["ref_id"]）即（imid != bundle["ref_id"]）时跳过尺度优化 直接使用参考图像的尺度 减少尺度漂移
        "single_rescale": True,
        # 最小截断倍数 在 update_truncation_multiplier 中 若 min_truncation_mult 不为 None 则 self.truncation_multiplier = max(self.truncation_multiplier, self.conf.min_truncation_mult) 当前为none 不强制最小截断阈值 依赖 MAD 或 quantile 计算的动态阈值
        "min_truncation_mult": None,
        # 详细输出级别
        "verbose": 0,
    }

    def _init(self, mpsfm_rec, correspondences):
        self.mpsfm_rec = mpsfm_rec  # 多视图立体重建记录存储到类属性
        self.correspondences = correspondences  # 对应点关系存储到类属性（特征点匹配）
        self.truncation_multiplier = 1  # 截断乘数初始化（异常值检测阈值的初始值）
        # 损失函数类型映射字典
        self.get_loss = {
            "trivial": LossFunctionType.TRIVIAL,  # L2 损失
            "cauchy": LossFunctionType.CAUCHY,  # log(1 + (x/scale)^2) scale主要是深度比率（投影深度/观测深度） projdepths / obsdepths
            "softl1": LossFunctionType.SOFT_L1,  # sqrt(x^2 + eps) - sqrt(eps) eps通常为较小的正数 例如1e-6 引入eps避免出现零和极小值导致的数值不稳定
        }

    # yield此生成器的方法为每帧待优化的图像（optim_ids）生成参数字典kwargs
    def __yield_problem_parameters(self, optim_ids, proj_depths=False):
        # 遍历优化图像ID
        for imid in optim_ids:
            # 获取当前图像对象 image包含图像的元数据，如特征点、深度图、相机姿态
            image = self.mpsfm_rec.images[imid]
            # 获取相机内参（焦距 主点等）
            camera = self.mpsfm_rec.rec.cameras[image.camera_id]
            # 获取与3D点关联的2D观测点索引 表示图像中具有有效2D-3D对应关系的观测点
            pt2D_ids = image.get_observation_point2D_idxs()
            # 获取pt2D_ids对应关键点的2D坐标（通常为 (x, y) 像素坐标） kps_with3D 是二维数组，形状为 (N,2)，N 为关键点数量
            kps_with3D = image.keypoint_coordinates(pt2D_ids)
            # 获取 pt2D_ids 对应的 3D 点 ID (p3d_ids是数组 长度与pt2D_ids相同)
            p3d_ids = image.point3D_ids(pt2D_ids)
            # 创建字典 kwargs （imid: 当前图像 ID |image: 图像对象| camera: 相机参数| pt3D_ids: 3D 点 ID 数组 | kps: 2D关键点坐标）
            kwargs = {"imid": imid, "image": image, "camera": camera, "pt3D_ids": p3d_ids, "kps": kps_with3D}
            # 从深度图获取关键点位置的观测深度
            kwargs["obsdepths"] = self.mpsfm_rec.images[imid].depth.data_prior_at_kps(kps_with3D)
            # 获取关键点位置的深度有效性掩码（valid）布尔数组 表示哪些关键点的深度值有效
            kwargs["valid"] = self.mpsfm_rec.images[imid].depth.valid_at_kps(kps_with3D)

            if proj_depths:
                # depth3d是3D点在相机坐标系中的Z值（理论深度）
                _, _, _, depth3d, _ = self.mpsfm_rec.project_image_3d_points(imid, kwargs["pt3D_ids"])
                # 先存储用于尺度优化或异常值过滤
                kwargs["projdepths"] = depth3d
            yield kwargs

    # 构建优化问题用于BA 优化相机姿态、3D 点、深度图和尺度  主要处理重投影误差和深度误差
    def __build_problem(
            self,
            bundle,  # 字典 包含optim_ids（待优化图像的 ID 列表） pts3D（待优化的 3D 点 ID 集合） ref_id（参考图像 ID，用于尺度一致性）
            fix_pose,  # 是否固定相机姿态，可以为 bool 或包含特定 imid 的 Iterable
            fix_scale,  # 是否固定深度图的尺度
            mode=None,  #
            depth_loss_name=None,  # 指定深度优化的损失函数类型 none为默认cauchy
            allow_scale_filter=False,  # 不开启尺度过滤（scale异常的点）
            param_multiplier=1,  # 调整异常值检测阈值的乘数
            depth_type="update",  # 使用更新后的深度（data_at_kps） 如果是别的例如"prior"就是使用初始深度（data_prior_at_kps）
            **kw,  # 允许额外参数传递
    ) -> tuple[
        Problem, dict, bool]:  # (bundler（pycolmap 的束调整器对象）, shift_scale（键为图像 ID，值为 [shift, scale] 数组）, success（优化是否成功）)

        conf = self.conf  # 获取default_conf配置 例如 depth_loss_name="cauchy", reproj_loss_name="SOFT_L1"
        optim_ids = list(bundle["optim_ids"])  # 从 bundle["optim_ids"] 获取待优化图像 ID 列表
        fixed_pose_ids: set[int] = set()
        if isinstance(fix_pose, bool):
            if fix_pose:
                fixed_pose_ids = set(optim_ids)
        elif fix_pose is not None:
            try:
                fixed_pose_ids = {int(i) for i in fix_pose}
            except TypeError:
                fixed_pose_ids = {int(fix_pose)}
        depth_loss_name = depth_loss_name or conf.depth_loss_name  # 未指定则用cauchy
        depth_loss_type = self.get_loss[
            depth_loss_name]  # 通过 self.get_loss 映射到 pycolmap.LossFunctionType（如 CAUCHY）算是自己写的函数与pycolmap库函数链接？
        shift_scale = {imid: np.array([0.0, 0.0]) for imid in optim_ids}  # 初始化每帧图像的平移和尺度参数为 [0.0, 0.0]
        # 创建 pycolmap.BundleAdjustmentConfig 对象，配置束调整参数
        ba_config = pycolmap.BundleAdjustmentConfig()
        # 迭代添加待优化图像
        for imid in optim_ids:
            ba_config.add_image(imid)
            # 若为局部优化，仅优化轨迹长度（track.length()）小于15的3D点 昨天看到的track length上限的约束原来在这
        if mode == "local":
            for p3Did in bundle["pts3D"]:
                if self.mpsfm_rec.points3D[p3Did].track.length() < 15:
                    ba_config.add_variable_point(p3Did)
        bundle_camids = [self.mpsfm_rec.images[c].camera_id for c in optim_ids]
        # 固定相机内参（如焦距、主点）不优化
        for camid in bundle_camids:
            ba_config.set_constant_cam_intrinsics(camid)
            # 计算所有图像关键点标准差的中位数（kp_std） 用于重投影误差的损失函数权重
        kp_std = np.median([self.mpsfm_rec.images[imid].kp_std for imid in optim_ids])
        # 配置束调整选项
        options = pycolmap.BundleAdjustmentOptions(
            # 重投影误差的权重 基于关键点标准差的倒数平方
            loss_function_magnitude=1 / kp_std ** 2,
            # 映射到SOFT_L1损失
            loss_function_type=pycolmap.LossFunctionType(self.conf.reproj_loss_name),
            # 损失函数尺度
            loss_function_scale=self.conf.reproj_loss_scale * kp_std,
        )
        # 创建束调整器，初始化优化问题problem
        bundler = pycolmap.create_default_bundle_adjuster(options, ba_config, self.mpsfm_rec.rec)
        problem = bundler.problem
        # 获取scale_filter_factor 用于尺度过滤
        scale_filter_factor = self.conf.scale_filter_factor
        # 获取 conf.gross_outliers=False，控制是否处理严重异常值
        gross_outliers = self.conf.gross_outliers
        # 调整异常值阈值 （由 update_truncation_multiplier更新)
        param_multiplier *= self.truncation_multiplier
        # 遍历图像并添加深度约束
        for ii, imid in enumerate(optim_ids):
            image = self.mpsfm_rec.images[imid]
            pose = image.cam_from_world
            # 处理深度数据（先添加残差，确保相关参数块已被注册）
            # 跳过未激活深度
            if not image.depth.activated:
                continue
            image = self.mpsfm_rec.images[imid]
            # p2Ds:2D 关键点索引
            p2Ds = np.array(image.get_observation_point2D_idxs())
            # kps_with3D：关键点坐标
            kps_with3D = image.keypoint_coordinates(p2Ds)
            # 深度有效性掩码 bool 上面代码所示那些点的深度值有效
            valid = image.depth.valid_at_kps(kps_with3D)
            kps_with3D = kps_with3D[valid]
            # 根据 depth_type，选择更新深度（data_at_kps）或初始深度（data_prior_at_kps）
            if depth_type == "update":
                depths = image.depth.data_at_kps(kps_with3D)
            else:
                depths = image.depth.data_prior_at_kps(kps_with3D)
            p2Ds = p2Ds[valid]
            # p3Ds: 对应的 3D 点 ID
            p3Ds = image.point3D_ids(p2Ds)
            # depth3d: 投影深度，从 3D 点投影到图像平面
            _, _, _, depth3d, _ = self.mpsfm_rec.project_image_3d_points(imid, p3Ds)
            # 异常值过滤
            # 初始掩码剔除非正深度
            mask = depths > 0
            # 同时剔除投影为负的 3D 深度（点在相机后方）
            mask *= depth3d > 0
            # 尺度过滤 上面代码参数所示 仅保留比率在 [1/1.5, 1.5]内的点
            if allow_scale_filter and self.conf.scale_filter:
                div = depths / depth3d
                mask *= (div < scale_filter_factor) * (div > (1 / scale_filter_factor))
            uncertainty_update = image.depth.uncertainty_update
            # 方差variances 从 uncertainty_update 获取 表示深度不确定性
            variances = np.array([uncertainty_update[pt2D_id] for pt2D_id in p2Ds])
            # 计算白化对数深度差 whitened 剔除超出 3 个标准差的点
            if gross_outliers and image.depth.activated:
                whitened = np.abs(np.log(depths).clip(1e-6, None) - np.log(depth3d).clip(1e-6, None)) / variances ** 0.5
                mask *= whitened < 3
            # 应用掩码
            # 若无有效点（np.sum(mask) == 0）跳过当前图像
            if np.sum(mask) == 0:
                print("No valid points for depth regularizing")
                continue
            # 仅保留 mask=True 的点
            depths = depths[mask]
            p2Ds = p2Ds[mask]
            variances = variances[mask]
            # 用于权重计算
            inv_uncert = 1 / variances.clip(1e-6, None)
            p3Ds = np.array(p3Ds)[mask]
            # 计算深度优化参数
            m = param_multiplier * self.conf.rob_std
            # 基于深度不确定性和深度值
            params = m * variances ** 0.5 / depths
            # 该权重反映深度误差的重要性
            magnitudes = depths ** 2 * inv_uncert
            # 添加深度束调整器 添加深度残差到优化问题
            pycolmap.create_depth_bundle_adjuster(
                problem,
                imid,
                p3Ds,
                depths,
                # 权重和鲁棒参数
                magnitudes,
                params,
                # 默认 CAUCHY
                depth_loss_type,
                # 平移和尺度参数
                shift_scale[imid],
                self.mpsfm_rec.rec,
                # 控制是否固定平移和尺度
                fix_shift=True,
                fix_scale=fix_scale,
                logloss=True,
            )
            # 仅在尺度为自由变量时设置流形（shift 固定、scale 自由）
            if not fix_scale:
                problem.set_manifold(shift_scale[imid], pyceres.SubsetManifold(2, [0]))
            # 在残差加入后再约束相机位姿（确保参数块已存在）
            should_fix_pose = (imid in fixed_pose_ids) or (ii == 0)
            if should_fix_pose:
                problem.set_parameter_block_constant(pose.rotation.quat)
                problem.set_parameter_block_constant(pose.translation)
            else:
                if ii == 1:
                    problem.set_manifold(pose.translation, pyceres.SubsetManifold(3, [0]))
                problem.set_manifold(pose.rotation.quat, pyceres.EigenQuaternionManifold())
        # 一次性求解：在完成所有图像的残差与流形设置后再统一求解
        self.solve(problem)
        return bundler, shift_scale

    # 用于构建深度图尺度和偏移优化问题
    def __build_shiftscale_problem(
            # bundle是一个字典 包含optim_ids（要优化的图像 ID 列表）
            # allow_scale_filter（控制是否启用基于深度比率的尺度过滤） allow_metric_scale_filter（控制是否启用跨图像一致性尺度过滤）
            self, bundle, allow_scale_filter=False, allow_metric_scale_filter=False
    ) -> tuple[Problem, dict, bool]:
        # 用来存储每帧图像的偏移和尺度参数
        shift_scale = {}
        # 是否启用尺度过滤
        scale_filter = self.conf.scale_filter
        # 尺度过滤的阈值因子
        scale_filter_factor = self.conf.scale_filter_factor
        # 是否启用度量尺度过滤
        metric_scale_factor = self.conf.metric_scale_filter
        # 是否对所有图像应用单一尺度调整
        single_rescale = self.conf.single_rescale
        # 简化策略：逐图像中位数估计 + 必要的保护打印

        for kwargs in self.__yield_problem_parameters(
                bundle["optim_ids"], proj_depths=scale_filter or metric_scale_factor
        ):
            pose = kwargs["image"].cam_from_world
            imid, p3dids = kwargs["imid"], kwargs["pt3D_ids"]
            base_valid = kwargs["valid"].copy()
            if (scale_filter_factor or metric_scale_factor) and (
                    "ref_id" in bundle and imid != bundle["ref_id"] and single_rescale
            ):
                continue
            # 尺度过滤
            if (
                    allow_metric_scale_filter
                    and metric_scale_factor  # 启用了度量尺度过滤
                    and ((imid == bundle["ref_id"]) or (not single_rescale))  # 当前图像是参考图像或未启用单一尺度调整
            ):
                # 计算其他优化图像的平均尺度map_scale；若没有“其他图像”，则跳过度量尺度过滤
                other_scales = [self.mpsfm_rec.images[id].depth.scale for id in bundle["optim_ids"] if id != imid]
                if len(other_scales) > 0:
                    # 计算scale（投影深度与观测深度的比率） 避免除零
                    scale = kwargs["projdepths"] / (kwargs["obsdepths"].clip(1e-6, None))
                    # 获取当前图像的深度尺度
                    im_scale = self.mpsfm_rec.images[imid].depth.scale
                    # 计算提议尺度
                    proposed_scale = scale * im_scale
                    # 计算其他优化图像的平均尺度map_scale
                    map_scale = np.mean(other_scales)
                    # 计算尺度比率
                    proposed_scale = np.clip(proposed_scale, 1e-6, None)
                    div = map_scale / proposed_scale
                    div[~np.isfinite(div)] = np.inf
                    # 标记尺度比率在合理范围内的点为valid
                    valid_metric = (div < 1.5) & (div > (1 / 1.5))
                    filtered_valid = base_valid & valid_metric
                    # 如果全部被过滤，回退到原始有效集，避免后续没有尺度约束
                    if filtered_valid.sum() == 0:
                        self.log(
                            f"Metric scale filter dropped all points for imid={imid}; falling back to unfiltered set",
                            level=2,
                        )
                        kwargs["valid"] = base_valid
                    else:
                        removed = base_valid.sum() - filtered_valid.sum()
                        kwargs["valid"] = filtered_valid
                        self.log(
                            f"Setting {removed} points as outliers for metric scale optim, out of {base_valid.sum()}",
                            level=3,
                        )
            # 如果启用了尺度过滤且未启用度量尺度过滤
            if allow_scale_filter and scale_filter and not allow_metric_scale_filter:
                div = kwargs["obsdepths"] / kwargs["projdepths"]
                # 标记比率在 1/scale_filter_factor 到 scale_filter_factor 范围内的点为有效点
                presum = kwargs["valid"].sum()
                kwargs["valid"] *= (div < scale_filter_factor) * (div > (1 / scale_filter_factor))
                # 打印被标记为异常点的数量
                print(f"Setting {presum - kwargs['valid'].sum()} points as outliers for scale optim")
            # 获取三维点坐标
            p3d = self.mpsfm_rec.point3D_coordinates(p3dids)
            # 计算三维点在相机坐标系中的深度（z 坐标） pose 是相机从世界坐标系的变换矩阵
            z_full = (pose * p3d)[:, -1]
            # 同时要求观测有效且投影深度为正
            valid_mask = kwargs["valid"] * (z_full > 0)
            # 提取有效点的投影深度 z 和观测深度 odepth
            z = z_full[valid_mask]
            odepth = kwargs["obsdepths"][valid_mask]
            # 计算对数尺度样本与调试打印
            log_ratios = np.log(((z / odepth)).clip(1e-6, None))
            if log_ratios.shape[0] == 0:
                continue
            proposed = np.median(log_ratios)
            # 偏移设为 0，尺度为对数形式
            shift_scale[imid] = np.array([0.0, proposed])

        return shift_scale, True

    # 该段代码主要是用于计算给定bundle中 3D 点的协方差
    # bundle包含optim_ids（需要优化的图像 ID 列表） pts3D（需要计算协方差的 3D 点 ID 列表）
    def calculate_point_covs(self, bundle):
        # 配置光束法平差
        ba_config = pycolmap.BundleAdjustmentConfig()
        for imid in bundle["optim_ids"]:
            ba_config.add_image(imid)
        for pt3D_id in bundle["pts3D"]:
            ba_config.add_variable_point(pt3D_id)
        # 固定相机内参
        bundle_camids = [self.mpsfm_rec.images[c].camera_id for c in bundle["optim_ids"]]
        # 对每个相机ID调用set_constant_cam_intrinsics 作用是将相机内参（焦距、主点等）设为固定值
        for camid in bundle_camids:
            ba_config.set_constant_cam_intrinsics(camid)
        # 设置每张图像关键点的标准差 主要反映了2D特征点检测中的不确定性（像角点检测的像素误差）
        kp_std = np.median([self.mpsfm_rec.images[imid].kp_std for imid in bundle["optim_ids"]])
        # 配置光束法平差
        options = pycolmap.BundleAdjustmentOptions(loss_function_magnitude=1 / kp_std ** 2)
        # 创建光束法平差器（优化器） 主要为了优化相机位姿（外参）和 3D 点位置 最小化重投影误差
        bundler = pycolmap.create_default_bundle_adjuster(options, ba_config, self.mpsfm_rec.rec)
        # 估计3D点协方差 用优化后的bundler（光束法平差结果）和重建数据计算
        options = pycolmap.BACovarianceOptions({"params": pycolmap.BACovarianceOptionsParams.POINTS})
        ba_cov = pycolmap.estimate_ba_covariance(options, self.mpsfm_rec.rec, bundler)
        # 遍历3D点ID存储协方差矩阵
        for p3Did in bundle["pts3D"]:
            self.mpsfm_rec.point_covs.data[p3Did] = ba_cov.get_point_cov(p3Did)

    # 对整个重建进行光束法平差 优化每帧的相机位姿（外参）和 3D 点位置
    def ba(self, bundle, mode, **kwargs) -> tuple[Problem, bool]:
        # 防御性处理：避免在调用方通过 **kwargs 同时传入 fix_pose/fix_scale/mode 而造成重复关键字
        fix_pose = kwargs.pop("fix_pose", False)
        fix_scale = kwargs.pop("fix_scale", True)
        mode_val = kwargs.pop("mode", mode)
        # 允许优化相机位姿（旋转和平移）并在默认情况下固定深度图的尺度（可被调用方覆盖）
        problem, _ = self.__build_problem(
            bundle,
            fix_pose=fix_pose,
            fix_scale=fix_scale,
            mode=mode_val,
            **kwargs,
        )
        return problem, True

    # 优化每帧深度图的尺度和偏移参数 并将深度图标记为“激活”
    def optimize_prior_shiftscale(self, bundle, **kwargs) -> tuple[dict, bool]:
        shift_scale, success = self.__build_shiftscale_problem(bundle, **kwargs)
        if not success:
            return None, False
        # 对数尺度(scale)转化为实际尺度 shift是偏移
        shift_scale = {imid: (shift, np.exp(scale)) for imid, (shift, scale) in shift_scale.items()}
        return shift_scale, True

    # 固定相机位姿 仅优化三角化的 3D 点位置 结合深度图先验（如果已激活）或 仅最小化重投影误差
    def refine_3d_points(self, bundle, **kwargs) -> tuple[Problem, bool]:
        # 防御性处理：调用方若携带 fix_pose/fix_scale，避免与本函数的显式指定重复
        kwargs.pop("fix_pose", None)
        kwargs.pop("fix_scale", None)
        problem, _ = self.__build_problem(
            # 固定相机位姿和深度图尺度
            bundle, fix_pose=True, fix_scale=True, depth_loss_name=self.conf.ref3d_loss_name, **kwargs
        )
        return problem, True

    # 使用 Ceres Solver 求解一个非线性优化问题（如光束法平差或深度尺度优化）
    def solve(self, problem):
        # 先创建 Ceres Solver 的配置对象
        options = pyceres.SolverOptions()
        # 配置线性求解器
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.minimizer_progress_to_stdout = bool(self.conf.verbose > 3)
        # 使用所有可用 CPU 线程
        options.num_threads = -1
        # 创建对象记录优化结果（如迭代次数、最终损失、收敛状态）
        summary = pyceres.SolverSummary()
        # 更新 problem 中的参数（如相机位姿、3D 点、深度尺度）
        pyceres.solve(options, problem, summary)
        self.log(summary.BriefReport(), level=2)

    def update_truncation_multiplier(self, imids):
        D3d, D, dstds, D3dunscaled = [], [], [], []
        Dunscaled = []
        # 前面流程和之前初始化各项参数万变不离其宗 根据函数名字理解即可
        for imid in imids:
            image = self.mpsfm_rec.images[imid]
            # 获取与 3D 点关联的 2D 关键点索引
            p2Ds = np.array(image.get_observation_point2D_idxs())
            kps_with3D = image.keypoint_coordinates(p2Ds)
            valid = image.depth.valid_at_kps(kps_with3D)
            kps_with3D = kps_with3D[valid]
            depths = image.depth.data_at_kps(kps_with3D)
            p2Ds = p2Ds[valid]
            # 获取与 2D 关键点对应的 3D 点 ID
            p3Ds = np.array(image.point3D_ids(p2Ds))
            mask = depths > 0
            _, _, _, depth3d, _ = self.mpsfm_rec.project_image_3d_points(imid, p3Ds[mask])
            depths = depths[mask]
            # 存储每个 2D 关键点的不确定性 可能是方差？
            uncertainty_update = image.depth.uncertainty_update
            # 提取方差
            variances = np.array([uncertainty_update[pt2D_id] for pt2D_id in p2Ds[mask]])
            D.append(depths)
            # 去尺度化的观测深度
            Dunscaled.append(depths / image.depth.scale)
            D3d.append(depth3d)
            # 去尺度化的投影深度
            D3dunscaled.append(depth3d / image.depth.scale)
            # 深度不确定性的标准差
            dstds.append(variances ** 0.5)
        # 将所有图像的观测深度、投影深度和标准差合并为单一数组
        depths = np.concatenate(D)
        depth3ds = np.concatenate(D3d)
        dstds = np.concatenate(dstds)
        # 计算归一化对数差异
        log_stds = dstds / depths  # 标准差归一化为相对标准差
        log_stds = np.clip(log_stds, 1e-6, None)
        log_distances = np.log(depths) - np.log(depth3ds)  # 深度值的相对尺度误差
        witened_log_distances = log_distances / log_stds  # 白化 使误差服从标准正态分布
        # 上面提到过sigma用于设置截断倍数 超过 sigma 的点被视为异常点 是白化后对数深度差异的标准差
        _, sigma = fit_robust_gaussian_mad(witened_log_distances)
        self.truncation_multiplier = sigma
        # 如果配置中指定了min_truncation_mult（最小截断倍数） 确保 truncation_multiplier 不低于此值
        if self.conf.min_truncation_mult is not None:
            self.truncation_multiplier = max(self.truncation_multiplier, self.conf.min_truncation_mult)