#!/usr/bin/env python3
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
# Modified by Skylar for depth-boosted-splatting with benchmarking
#

import os
import time
import uuid
import sys
import torch
from random import randint
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state, get_expon_lr_func
from utils.benchmarks import BenchmarkLogger

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

from tqdm import tqdm
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


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("SparseAdam requested but not installed; please install 3dgs_accel.")

    # 1) set up logging & scene
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        start_iter = first_iter + 1
    else:
        start_iter = 1

    # 2) background color tensor
    bg_color = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 3) prepare BenchmarkLogger and timer
    start_time = time.time()
    bench_logger = BenchmarkLogger(scene.model_path, tb_writer)

    # 4) prepare CUDA timers
    use_cuda_timer = torch.cuda.is_available()
    if use_cuda_timer:
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end   = torch.cuda.Event(enable_timing=True)
    else:
        iter_start = iter_end = None

    use_sparse_adam = (opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack   = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss = ema_depth_loss = 0.0

    progress_bar = tqdm(range(start_iter, opt.iterations + 1), desc="Training progress")

    for iteration in progress_bar:
        # --- network GUI hook (optional) ---
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_bytes = None
                (custom_cam, do_train, pipe.convert_SHs_python,
                 pipe.compute_cov3D_python, keep_alive, scaling_mod) = network_gui.receive()
                if custom_cam is not None:
                    net_img = render(custom_cam, gaussians, pipe, background,
                                     scaling_modifier=scaling_mod,
                                     use_trained_exp=dataset.train_test_exp,
                                     separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_bytes = memoryview((net_img.clamp(0,1)*255).byte()
                                           .permute(1,2,0).cpu().numpy())
                network_gui.send(net_bytes, dataset.source_path)
                if do_train and ((iteration < opt.iterations) or not keep_alive):
                    break
            except:
                network_gui.conn = None

        # --- record start time ---
        if use_cuda_timer:
            iter_start.record()

        # --- LR scheduling & SH growth ---
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # --- pick a random training view ---
        if not viewpoint_stack:
            viewpoint_stack   = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        idx = randint(0, len(viewpoint_indices)-1)
        cam = viewpoint_stack.pop(idx)
        vind = viewpoint_indices.pop(idx)

        # --- render ---
        if (iteration-1) == debug_from:
            pipe.debug = True
        bg = (torch.rand(3, device="cuda") if opt.random_background else background)
        pkg = render(cam, gaussians, pipe, bg,
                     use_trained_exp=dataset.train_test_exp,
                     separate_sh=SPARSE_ADAM_AVAILABLE)
        image = pkg["render"]
        vpts  = pkg["viewspace_points"]
        vis_f = pkg["visibility_filter"]
        radii = pkg["radii"]

        if cam.alpha_mask is not None:
            image *= cam.alpha_mask.cuda()

        # --- compute loss ---
        gt    = cam.original_image.cuda()
        Ll1   = l1_loss(image, gt)
        ssim_v = fused_ssim(image.unsqueeze(0), gt.unsqueeze(0)) \
            if FUSED_SSIM_AVAILABLE else ssim(image, gt)
        loss = (1.0 - opt.lambda_dssim)*Ll1 + opt.lambda_dssim*(1.0 - ssim_v)

        # depth‐L1 regularization
        depth_term = 0.0
        if depth_l1_weight(iteration)>0 and cam.depth_reliable:
            pred_d = pkg["depth"]
            mono_d = cam.invdepthmap.cuda()
            mask   = cam.depth_mask.cuda()
            depth_term = depth_l1_weight(iteration)*torch.abs((pred_d - mono_d)*mask).mean()
            loss += depth_term
            depth_term = depth_term.item()

        loss.backward()

        # --- record end time ---
        if use_cuda_timer:
            iter_end.record()

        with torch.no_grad():
            # update EMA for display
            ema_loss        = 0.4*loss.item() + 0.6*ema_loss
            ema_depth_loss  = 0.4*depth_term    + 0.6*ema_depth_loss

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "L1": f"{ema_loss:.7f}",
                    "DL1": f"{ema_depth_loss:.7f}"
                })

            # --- evaluation & logging ---
            if iteration in testing_iterations:
                # TensorBoard & CSV detailed test report
                training_report(
                    tb_writer, iteration, Ll1, loss, l1_loss,
                    (iter_start.elapsed_time(iter_end) if use_cuda_timer else 0.0),
                    testing_iterations, scene, render,
                    (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)
                )

                # --- BenchmarkLogger call ---
                # wrap render so that it matches measure_training API
                def _bench_render(c):
                    return render(c, gaussians, pipe, background,
                                  use_trained_exp=dataset.train_test_exp,
                                  separate_sh=SPARSE_ADAM_AVAILABLE)

                metrics = bench_logger.measure_training(
                    start_time, scene, _bench_render, scene.getTestCameras()
                )
                extra_meta = {
                    "dataset_name": os.path.basename(dataset.source_path),
                    "loss_lambda_dssim": opt.lambda_dssim,
                    "depth_l1_weight_init": opt.depth_l1_weight_init,
                    "depth_l1_weight_final": opt.depth_l1_weight_final,
                    "optimizer_type": opt.optimizer_type,
                }
                bench_logger.log("benchmark", iteration, metrics, extra_meta)

            # --- saving & densification ---
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[vis_f] = torch.max(gaussians.max_radii2D[vis_f], radii[vis_f])
                gaussians.add_densification_stats(vpts, vis_f)
                if iteration>opt.densify_from_iter and iteration%opt.densification_interval==0:
                    threshold = 20 if iteration>opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, threshold, radii)
                if iteration%opt.opacity_reset_interval==0 or (dataset.white_background and iteration==opt.densify_from_iter):
                    gaussians.reset_opacity()

            # --- optimizer step ---
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    vis_mask = radii>0
                    gaussians.optimizer.step(vis_mask, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))

    print("\nTraining complete.")


def training_report(tb_writer, iteration, Ll1, loss, l1_loss_func, elapsed,
                    testing_iterations, scene, render_fn, render_args):
    # identical to your existing function; logs L1 / PSNR / SSIM on test & train views
    if tb_writer:
        tb_writer.add_scalar('train/L1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train/iter_time_ms', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        val_configs = [
            {'name':'test',  'cameras': scene.getTestCameras()},
            {'name':'train', 'cameras': scene.getTrainCameras()[5:30:5]}
        ]
        for cfg in val_configs:
            if not cfg['cameras']: continue
            l1_sum = psnr_sum = 0.0
            for idx, vp in enumerate(cfg['cameras']):
                out  = torch.clamp(render_fn(vp, *render_args)['render'],0,1)
                gt   = torch.clamp(vp.original_image.to("cuda"),0,1)
                if tb_writer and idx<5:
                    tb_writer.add_image(f"{cfg['name']}/{vp.image_name}/render",
                                        out[None], iteration)
                    if iteration==testing_iterations[0]:
                        tb_writer.add_image(f"{cfg['name']}/{vp.image_name}/gt",
                                            gt[None], iteration)
                l1_sum   += l1_loss_func(out, gt).mean().item()
                psnr_sum += psnr(out, gt).mean().item()
            n = len(cfg['cameras'])
            print(f"[ITER {iteration}] {cfg['name']} — L1 {l1_sum/n:.4f}, PSNR {psnr_sum/n:.2f}")
            if tb_writer:
                tb_writer.add_scalar(f"{cfg['name']}/L1_loss", l1_sum/n, iteration)
                tb_writer.add_scalar(f"{cfg['name']}/PSNR", psnr_sum/n, iteration)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000,30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args()
    args.save_iterations.append(args.iterations)

    print("Optimizing", args.model_path)
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args.test_iterations,
             args.save_iterations,
             args.checkpoint_iterations,
             args.start_checkpoint,
             args.debug_from)
