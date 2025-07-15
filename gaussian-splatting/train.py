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
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state, get_expon_lr_func
from utils.benchmarks import BenchmarkLogger             # --- BENCHMARKING ---
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Sparse Adam requested but 3dgs_accel not installed.")

    # 1) prepare logger
    tb_writer = prepare_output_and_logger(dataset)

    # --- BENCHMARKING: initialize BenchmarkLogger and timer ---
    bench = BenchmarkLogger(
        os.path.join(dataset.model_path, "benchmarks"),
        tb_writer
    )
    start_time = time.time()
    # --------------------------------------------

    # 2) build scene
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    first_iter = 0
    if checkpoint:
        (model_args, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_args, opt)

    # 3) background color
    bg_color = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 4) timer events
    use_cuda_timer = torch.cuda.is_available()
    if use_cuda_timer:
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end   = torch.cuda.Event(enable_timing=True)
    else:
        iter_start = iter_end = None

    # 5) optim & schedule
    use_sparse_adam = (opt.optimizer_type=="sparse_adam") and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 6) sampling order
    viewpoint_stack  = scene.getTrainCameras().copy()
    viewpoint_indices= list(range(len(viewpoint_stack)))
    ema_loss = ema_depth = 0.0

    # 7) training loop
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    for iteration in range(first_iter+1, opt.iterations+1):
        # GUI handling omitted for brevity...
        if use_cuda_timer: iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # pick & render a random train view
        if not viewpoint_stack:
            viewpoint_stack   = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        ri = randint(0, len(viewpoint_indices)-1)
        cam = viewpoint_stack.pop(ri)
        _  = viewpoint_indices.pop(ri)
        bg = torch.rand((3,), device="cuda") if opt.random_background else background

        pkg = render(cam, gaussians, pipe, bg,
                     use_trained_exp=dataset.train_test_exp,
                     separate_sh=SPARSE_ADAM_AVAILABLE)
        image, view_pts, vis_filter, radii = pkg["render"], pkg["viewspace_points"], pkg["visibility_filter"], pkg["radii"]

        # loss
        gt = cam.original_image.cuda()
        L1 = l1_loss(image, gt)
        s  = fused_ssim(image.unsqueeze(0), gt.unsqueeze(0)) if FUSED_SSIM_AVAILABLE else ssim(image, gt)
        loss = (1-opt.lambda_dssim)*L1 + opt.lambda_dssim*(1-s)

        # optional depth‐L1
        if depth_l1_weight(iteration)>0 and cam.depth_reliable:
            pred = pkg["depth"]
            mono = cam.invdepthmap.cuda()
            mask = cam.depth_mask.cuda()
            dL1 = torch.abs((pred-mono)*mask).mean()
            loss += depth_l1_weight(iteration)*dL1

        loss.backward()
        if use_cuda_timer: iter_end.record()

        # logging, save, densify, optimizer step (omitted here for brevity;
        # keep exactly your existing logic up through `training_report(...)`.)

        # --- Insert here after training_report(...) ----
        if iteration in testing_iterations or iteration==opt.iterations:
            # run default evaluation
            training_report(tb_writer, iteration, L1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end) if use_cuda_timer else 0,
                            testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp)

            # BENCHMARK: measure + log
            metrics = bench.measure_training(
                start_time,
                scene,
                lambda c: render(c, gaussians, pipe, background,
                                 use_trained_exp=dataset.train_test_exp,
                                 separate_sh=SPARSE_ADAM_AVAILABLE),
                scene.getTestCameras()
            )
            bench.log("reconstruction_quality", iteration, metrics)
        # -----------------------------------------------

        # ... rest of your loop (saving, densification, optimizer step, checkpointing)

    print("Training complete.")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique = os.getenv('OAR_JOB_ID', str(uuid.uuid4())[:10])
        args.model_path = os.path.join("./output", unique)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path,"cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    print("Output folder:", args.model_path)

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available")
        return None


def training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                    elapsed, testing_iterations, scene: Scene,
                    renderFunc, renderArgs, train_test_exp):
    # your existing test/train logging (unchanged)...

    pass  # keep your full existing implementation here


if __name__=="__main__":
    parser = ArgumentParser("Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--test_iterations', nargs="+", type=int, default=[7000,30000])
    parser.add_argument('--save_iterations', nargs="+", type=int, default=[7000,30000])
    parser.add_argument('--checkpoint_iterations', nargs="+", type=int, default=[])
    parser.add_argument('--start_checkpoint')
    args = parser.parse_args()
    args.save_iterations.append(args.iterations)

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
