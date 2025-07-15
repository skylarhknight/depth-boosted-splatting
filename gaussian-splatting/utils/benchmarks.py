import os, time, shutil, psutil, torch, lpips, pandas as pd
from utils.image_utils import psnr, ssim

# one global LPIPS model
_LPIPS = lpips.LPIPS(net='vgg').to('cuda' if torch.cuda.is_available() else 'cpu')

class BenchmarkLogger:
    def __init__(self, out_dir, tb_writer=None):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.tb = tb_writer

    def _disk_usage(self, path):
        du = shutil.disk_usage(path)
        return du.used

    def log(self, scene_name, iteration, metrics: dict):
        """
        metrics: a flat dict of all numbers you care about.
        """
        # 1) TensorBoard
        if self.tb:
            for k,v in metrics.items():
                self.tb.add_scalar(f"{scene_name}/{k}", v, iteration)

        # 2) CSV
        csv_path = os.path.join(self.out_dir, f"{scene_name}.csv")
        df = pd.DataFrame([{**{"iter": iteration}, **metrics}])
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', header=header, index=False)

    def measure_training(self, start_time, scene, render_func, test_cams):
        """
        Perform rendering+quality metrics, measure perf counters,
        and return a metrics dict.
        """
        # wall-clock
        total_time = time.time() - start_time

        # PSNR / SSIM / LPIPS over all test views
        Q = {"psnr": [], "ssim": [], "lpips": []}
        for cam in test_cams:
            out = render_func(cam)["render"].clamp(0,1)
            gt  = cam.original_image.to(out.device).clamp(0,1)
            Q["psnr"].append( psnr(out, gt).item() )
            Q["ssim"].append( ssim(out, gt).item() )
            # LPIPS expects batch & range [-1,+1]
            Q["lpips"].append( _LPIPS(2*out[None]-1, 2*gt[None]-1).item() )

        # peak GPU memory
        gpu_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats()

        # CPU percent
        cpu_pct = psutil.cpu_percent(interval=None)

        # disk usage of your model folder
        disk_used = self._disk_usage(scene.model_path)

        # #points & radii stats
        xyz = scene.gaussians.get_xyz
        radii = scene.gaussians.max_radii2D
        num_pts = xyz.shape[0]
        rad_mean = float(radii.mean().item())
        rad_var  = float(radii.var().item())

        # depth‐map stats if provided
        depth = None
        try:
            depth = render_func(test_cams[0])["depth"]
        except:
            depth = None

        dmin = float(depth.min().item()) if depth is not None else None
        dmax = float(depth.max().item()) if depth is not None else None
        dvar = float(depth.var().item()) if depth is not None else None

        # assemble a flat dict
        metrics = {
            "time_s": total_time,
            "gpu_peak_bytes": gpu_peak,
            "cpu_pct": cpu_pct,
            "disk_bytes": disk_used,
            "num_images": len(scene.getTrainCameras()),
            "image_res_h": test_cams[0].image_height,
            "image_res_w": test_cams[0].image_width,
            "num_gaussians": num_pts,
            "radius_mean": rad_mean,
            "radius_var":  rad_var,
            "depth_min":   dmin,
            "depth_max":   dmax,
            "depth_var":   dvar,
            # quality averages:
            "psnr": sum(Q["psnr"])/len(Q["psnr"]),
            "ssim": sum(Q["ssim"])/len(Q["ssim"]),
            "lpips": sum(Q["lpips"])/len(Q["lpips"]),
        }
        return metrics
