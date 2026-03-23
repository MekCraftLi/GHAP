#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# from gaussian_splatting_sampler.utils import  RobustAutoSamplingTrigger
# from gaussian_splatting_sampler.gmm_sampler import gaussian_model_reduction
from gmm_sampler import gaussian_model_reduction
import json
import time
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, compaction=None):
    if compaction.flag:
        print('With Compaction!')
    else:
        print('No Compaction!')

    start_time = time.time()
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        ######################
        # TODO: gaussians.update_learning_rate(iteration, False)
        # if iteration == 30000:
        #     gaussians.update_learning_rate(iteration, True)
        # else:
        #     gaussians.update_learning_rate(iteration, True)
        gaussians.update_learning_rate(iteration, False)
        ######################
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()
        torch.cuda.synchronize()
        elapsed_time = iter_start.elapsed_time(iter_end)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                # TODO: subsampling in training
            if compaction.flag:
                if iteration in compaction.iter:
                    index = compaction.iter.index(iteration)
                    gaussians = subsampling(gaussians, compaction.ratio[index], 42, compaction.method)
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    # 将高斯模型返回
    return gaussians

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('gaussians_num', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    # if tb_writer and (idx < 5):
                    if tb_writer:
                    # TODO: CHANGE
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# TODO: For subsampling in training
def subsampling(gaussians, ratio, random_seed=42, method='GMR'):
    """
    Foveated Compression:
    - Core(0~30%): 保留 70% 名额
    - Mid(30%~70%): 保留 25% 名额
    - Far(70%~100%): 保留 5% 名额
    """
    if method != '2GMR':
        # 保留原随机分支逻辑（作为回退）
        import numpy as np
        n_total = gaussians.get_xyz.shape[0]
        if n_total == 0:
            return gaussians

        downsample_num = int(ratio) if ratio > 1 else int(n_total * ratio)
        downsample_num = max(1, min(n_total, downsample_num))

        np.random.seed(random_seed)
        keep_indices = np.random.choice(n_total, size=downsample_num, replace=False)
        keep_indices = torch.from_numpy(keep_indices).to(gaussians.get_xyz.device)

        with torch.no_grad():
            new_xyz = gaussians._xyz[keep_indices]
            new_features_dc = gaussians._features_dc[keep_indices]
            new_features_rest = gaussians._features_rest[keep_indices]
            new_scaling = gaussians._scaling[keep_indices]
            new_rotation = gaussians._rotation[keep_indices]
            new_opacity = gaussians._opacity[keep_indices]

        xyz_tensors = gaussians.replace_tensor_to_optimizer(new_xyz, "xyz")
        gaussians._xyz = xyz_tensors["xyz"]
        f_dc_tensors = gaussians.replace_tensor_to_optimizer(new_features_dc, "f_dc")
        gaussians._features_dc = f_dc_tensors["f_dc"]
        f_rest_tensors = gaussians.replace_tensor_to_optimizer(new_features_rest, "f_rest")
        gaussians._features_rest = f_rest_tensors["f_rest"]
        scaling_tensors = gaussians.replace_tensor_to_optimizer(new_scaling, "scaling")
        gaussians._scaling = scaling_tensors["scaling"]
        rotation_tensors = gaussians.replace_tensor_to_optimizer(new_rotation, "rotation")
        gaussians._rotation = rotation_tensors["rotation"]
        opacity_tensors = gaussians.replace_tensor_to_optimizer(new_opacity, "opacity")
        gaussians._opacity = opacity_tensors["opacity"]

        new_n = new_xyz.shape[0]
        device = gaussians.get_xyz.device
        gaussians.xyz_gradient_accum = torch.zeros((new_n, 1), device=device)
        gaussians.denom = torch.zeros((new_n, 1), device=device)
        gaussians.max_radii2D = torch.zeros((new_n,), device=device)
        return gaussians

    # -----------------------------
    # 1) 计算中心与距离
    # -----------------------------
    xyz = gaussians.get_xyz
    device = xyz.device
    n_total = xyz.shape[0]
    if n_total == 0:
        return gaussians

    target_n = int(ratio) if ratio > 1 else int(n_total * ratio)
    target_n = max(1, min(n_total, target_n))

    center = xyz.mean(dim=0, keepdim=True)  # (1, 3)
    distances = torch.linalg.norm(xyz - center, dim=1)  # (N,)

    # -----------------------------
    # 2) 三分区掩码 (Core/Mid/Far)
    # -----------------------------
    q30 = torch.quantile(distances, 0.30)
    q70 = torch.quantile(distances, 0.70)

    core_idx = torch.where(distances <= q30)[0]
    mid_idx = torch.where((distances > q30) & (distances <= q70))[0]
    far_idx = torch.where(distances > q70)[0]

    zone_indices = [core_idx, mid_idx, far_idx]
    zone_weights = [0.70, 0.25, 0.05]  # 不公平配额
    zone_caps = [int(z.numel()) for z in zone_indices]

    # -----------------------------
    # 3) 名额分配 + 边界兜底
    # -----------------------------
    raw_keep = [target_n * w for w in zone_weights]
    keep = [min(zone_caps[i], int(raw_keep[i])) for i in range(3)]

    # 若 core 非空且总目标 > 0，尽量保证 core 至少 1 个（强调 ROI）
    if zone_caps[0] > 0 and target_n > 0 and keep[0] == 0:
        keep[0] = 1

    # 如果超过目标，优先从 Far -> Mid -> Core 回收
    excess = sum(keep) - target_n
    if excess > 0:
        for i in [2, 1, 0]:
            # core 的最低保留线：若 core 非空且 target_n>0，保留至少 1
            floor_i = 1 if (i == 0 and zone_caps[0] > 0 and target_n > 0) else 0
            removable = max(0, keep[i] - floor_i)
            delta = min(removable, excess)
            keep[i] -= delta
            excess -= delta
            if excess == 0:
                break

    # 如果不足目标，优先向 Core -> Mid -> Far 填充
    residual = target_n - sum(keep)
    while residual > 0:
        progressed = False
        for i in [0, 1, 2]:
            if keep[i] < zone_caps[i]:
                keep[i] += 1
                residual -= 1
                progressed = True
                if residual == 0:
                    break
        if not progressed:
            break  # 所有分区都满了，无法继续补

    # -----------------------------
    # 4) 分区切片 + 分别 GMR
    # -----------------------------
    def _build_sub_gaussian(src_g, idx):
        """按索引构建局部 GaussianModel，供局部 GMR 使用。"""
        sub_g = GaussianModel(src_g.max_sh_degree, src_g.optimizer_type)
        sub_g.active_sh_degree = src_g.active_sh_degree
        with torch.no_grad():
            sub_g._xyz = torch.nn.Parameter(src_g._xyz[idx].detach().clone().requires_grad_(True))
            sub_g._features_dc = torch.nn.Parameter(src_g._features_dc[idx].detach().clone().requires_grad_(True))
            sub_g._features_rest = torch.nn.Parameter(src_g._features_rest[idx].detach().clone().requires_grad_(True))
            sub_g._scaling = torch.nn.Parameter(src_g._scaling[idx].detach().clone().requires_grad_(True))
            sub_g._rotation = torch.nn.Parameter(src_g._rotation[idx].detach().clone().requires_grad_(True))
            sub_g._opacity = torch.nn.Parameter(src_g._opacity[idx].detach().clone().requires_grad_(True))
        return sub_g

    xyz_parts = []
    fdc_parts = []
    frest_parts = []
    scaling_parts = []
    rotation_parts = []
    opacity_parts = []

    # 固定随机种子并给不同分区做轻微偏移，避免三个分区完全同随机流
    zone_seed_offsets = [0, 101, 202]

    for z_i, idx in enumerate(zone_indices):
        zone_n = int(idx.numel())
        keep_n = int(keep[z_i])
        if zone_n == 0 or keep_n <= 0:
            continue

        # 不需要压缩：直接保留该分区全部
        if keep_n >= zone_n:
            xyz_parts.append(gaussians._xyz[idx].detach().clone())
            fdc_parts.append(gaussians._features_dc[idx].detach().clone())
            frest_parts.append(gaussians._features_rest[idx].detach().clone())
            scaling_parts.append(gaussians._scaling[idx].detach().clone())
            rotation_parts.append(gaussians._rotation[idx].detach().clone())
            opacity_parts.append(gaussians._opacity[idx].detach().clone())
            continue

        local_ratio = float(keep_n) / float(zone_n)
        local_model = _build_sub_gaussian(gaussians, idx)

        reduced = gaussian_model_reduction(
            local_model,
            ratio=local_ratio,
            random_seed=int(random_seed + zone_seed_offsets[z_i])
        )

        xyz_parts.append(reduced._xyz.detach())
        fdc_parts.append(reduced._features_dc.detach())
        frest_parts.append(reduced._features_rest.detach())
        scaling_parts.append(reduced._scaling.detach())
        rotation_parts.append(reduced._rotation.detach())
        opacity_parts.append(reduced._opacity.detach())

    # 极端兜底：若由于边界导致没有任何分区写入，至少保留一个 core 点
    if len(xyz_parts) == 0:
        fallback_idx = core_idx[:1] if core_idx.numel() > 0 else torch.tensor([0], device=device, dtype=torch.long)
        xyz_parts = [gaussians._xyz[fallback_idx].detach().clone()]
        fdc_parts = [gaussians._features_dc[fallback_idx].detach().clone()]
        frest_parts = [gaussians._features_rest[fallback_idx].detach().clone()]
        scaling_parts = [gaussians._scaling[fallback_idx].detach().clone()]
        rotation_parts = [gaussians._rotation[fallback_idx].detach().clone()]
        opacity_parts = [gaussians._opacity[fallback_idx].detach().clone()]

    # -----------------------------
    # 5) 拼接回全局并替换优化器参数
    # -----------------------------
    new_xyz = torch.cat(xyz_parts, dim=0).to(device)
    new_fdc = torch.cat(fdc_parts, dim=0).to(device)
    new_frest = torch.cat(frest_parts, dim=0).to(device)
    new_scaling = torch.cat(scaling_parts, dim=0).to(device)
    new_rotation = torch.cat(rotation_parts, dim=0).to(device)
    new_opacity = torch.cat(opacity_parts, dim=0).to(device)

    # 使用原工程的 optimizer 替换接口，内部会包装 nn.Parameter(requires_grad=True)
    xyz_tensors = gaussians.replace_tensor_to_optimizer(new_xyz, "xyz")
    gaussians._xyz = xyz_tensors["xyz"]
    f_dc_tensors = gaussians.replace_tensor_to_optimizer(new_fdc, "f_dc")
    gaussians._features_dc = f_dc_tensors["f_dc"]
    f_rest_tensors = gaussians.replace_tensor_to_optimizer(new_frest, "f_rest")
    gaussians._features_rest = f_rest_tensors["f_rest"]
    scaling_tensors = gaussians.replace_tensor_to_optimizer(new_scaling, "scaling")
    gaussians._scaling = scaling_tensors["scaling"]
    rotation_tensors = gaussians.replace_tensor_to_optimizer(new_rotation, "rotation")
    gaussians._rotation = rotation_tensors["rotation"]
    opacity_tensors = gaussians.replace_tensor_to_optimizer(new_opacity, "opacity")
    gaussians._opacity = opacity_tensors["opacity"]

    # 重置优化器相关统计量
    new_n = gaussians.get_xyz.shape[0]
    gaussians.xyz_gradient_accum = torch.zeros((new_n, 1), device=device)
    gaussians.denom = torch.zeros((new_n, 1), device=device)
    gaussians.max_radii2D = torch.zeros((new_n,), device=device)

    return gaussians


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--sampling_iter", type=int, nargs='+', default=[15000])
    parser.add_argument("--sampling_ratio", type=float, nargs='+', default=[0.05])
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument("--block_num", type=int, default=3000)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    class Compact:
        def __init__(self, compact, sampling_iter, sampling_ratio, random):
            self.flag = compact
            self.iter = sampling_iter
            self.ratio = sampling_ratio
            if random == True:
                self.method = 'random'
            else:
                self.method = 'GMR'
    # Start GUI server, configure and run training
    compaction = Compact(args.compact, args.sampling_iter, args.sampling_ratio, args.random)
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gaussians = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, compaction)

    #_features_rest是所有的反光特征，形状(N, 15, 3)
    rest_features = gaussians._features_rest.data

    #计算每个高斯体的45个高阶特征的绝对值总和
    sh_magnitude = torch.norm(rest_features, dim=(1, 2));

    # sh_magnitude的形状是(N,)，其中每个元素表示对应高斯体的45个高阶特征的绝对值总和

    # 2. 设定阈值，找出最不反光的70%的高斯体
    threshold = torch.quantile(sh_magnitude, 0.7)

    # 3. 生成布尔掩码：哪些高斯体是"漫反射"的？
    is_matte_mask = sh_magnitude < threshold

    print(f"总共有 {rest_features.shape[0]} 个点。")
    print(f"检测到 {is_matte_mask.sum().item()} 个漫反射点，准备裁剪它们的高阶 SH！")

    # 直接把这 70% 漫反射点的高阶 SH 特征强制设为 0.0！
    gaussians._features_rest.data[is_matte_mask] = 0.0
    import os
    import zipfile

    print("\n[保存] 正在将 SH 裁剪后的模型保存至硬盘...")

    # 1. 确定原始模型的路径 (用于对比)
    original_ply_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iterations}")
    original_ply_path = os.path.join(original_ply_dir, "point_cloud.ply")

    # 2. 定义新模型的保存路径 (加了 _pruned 后缀)
    pruned_ply_dir = os.path.join(lp._model_path, "point_cloud", f"iteration_{args.iterations}_pruned")
    os.makedirs(pruned_ply_dir, exist_ok=True)
    pruned_ply_path = os.path.join(pruned_ply_dir, "point_cloud.ply")

    # 调用模型自带的保存函数
    gaussians.save_ply(pruned_ply_path)

    print(f"压缩模型保存在了{pruned_ply_path}")

    # 3. 定义 ZIP 压缩辅助函数
    def zip_file(input_file_path, output_zip_path):
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # arcname 防止把整个绝对路径都压缩进 zip 里
            zipf.write(input_file_path, arcname="point_cloud.ply")
        return os.path.getsize(output_zip_path) / (1024 * 1024)  # 返回 MB


    print("\n[测试] 正在进行 ZIP 字典极限压缩测试 (这可能需要十几秒，请稍候)...")

    if os.path.exists(original_ply_path):
        # 压缩原始文件
        original_zip_path = os.path.join(original_ply_dir, "point_cloud.zip")
        orig_zip_size = zip_file(original_ply_path, original_zip_path)
        orig_ply_size = os.path.getsize(original_ply_path) / (1024 * 1024)

        # 压缩裁剪后的文件
        pruned_zip_path = os.path.join(pruned_ply_dir, "point_cloud_pruned.zip")
        pruned_zip_size = zip_file(pruned_ply_path, pruned_zip_path)
        pruned_ply_size = os.path.getsize(pruned_ply_path) / (1024 * 1024)

        # 4. 打印战报
        print("\n" + "=" * 55)
        print("📊 【自适应 SH 降维压缩战报 (Adaptive SH Pruning)】")
        print("=" * 55)
        print(f"-> [Baseline] 原版模型 PLY 占用: {orig_ply_size:.2f} MB")
        print(f"-> [Baseline] 原版模型 ZIP 占用: {orig_zip_size:.2f} MB")
        print("-" * 55)
        print(f"-> [Ours] 裁剪后模型 PLY 占用: {pruned_ply_size:.2f} MB")
        print(f"-> [Ours] 裁剪后模型 ZIP 占用: {pruned_zip_size:.2f} MB")
        print("=" * 55)

        reduction = (1 - pruned_zip_size / orig_zip_size) * 100
        print(f"🎉 你的策略让传输体积在 GHAP 的基础上，再次额外暴降了: {reduction:.2f}%！")
        print("=" * 55)
    else:
        print(f"⚠️ 未找到原始 PLY 文件 ({original_ply_path})，可能是没有开启保存或迭代次数未到。")

    print("\nTraining & Pruning complete.")

    # All done
    print("\nTraining complete.")
