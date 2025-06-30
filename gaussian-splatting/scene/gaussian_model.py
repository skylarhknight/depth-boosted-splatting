import os
import json
import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement
from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    strip_symmetric,
    build_scaling_rotation,
    build_rotation,
)
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud

# fall back to pure-Python knn if no extension
try:
    from simple_knn._C import distCUDA2
except ImportError:
    def distCUDA2(X, Y):
        return ((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2).sum(-1)

# Attempt SparseGaussianAdam if available
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except ImportError:
    SparseGaussianAdam = None

# Pick default device
if torch.cuda.is_available():
    _device = torch.device("cuda")
elif torch.backends.mps.is_available():
    _device = torch.device("mps")
else:
    _device = torch.device("cpu")

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0, device=_device)
        self._features_dc = torch.empty(0, device=_device)
        self._features_rest = torch.empty(0, device=_device)
        self._scaling = torch.empty(0, device=_device)
        self._rotation = torch.empty(0, device=_device)
        self._opacity = torch.empty(0, device=_device)
        self.max_radii2D = torch.empty(0, device=_device)
        self.xyz_gradient_accum = torch.empty(0, device=_device)
        self.denom = torch.empty(0, device=_device)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer else None,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
         self._xyz, 
         self._features_dc, 
         self._features_rest,
         self._scaling, 
         self._rotation, 
         self._opacity,
         self.max_radii2D, 
         self.xyz_gradient_accum, 
         self.denom,
         opt_dict, 
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        # move raw points & colors onto _device
        # fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(_device)
        # fused_color       = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(_device))
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(_device)
        fused_color       = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(_device))


        # initialize features on _device
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
                               dtype=torch.float, device=_device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation:", fused_point_cloud.shape[0])

        # distance matrix
        # pts = torch.from_numpy(np.asarray(pcd.points)).float().to(_device)
        # dist2 = torch.clamp_min(distCUDA2(pts, pts), 1e-7)
        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3).to(_device)
        pts = torch.from_numpy(np.asarray(pcd.points)).float().to(_device)
        dist2 = torch.clamp_min(distCUDA2(pts, pts), 1e-7)
        # now repeat across the third dimension
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 1, 3).to(_device)

        # zero rotations & opacities on _device
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=_device)
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=_device)
        )

        # register as parameters
        self._xyz           = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc   = nn.Parameter(features[:, :, :1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling       = nn.Parameter(scales.requires_grad_(True))
        self._rotation      = nn.Parameter(rots.requires_grad_(True))
        self._opacity       = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D    = torch.zeros((self.get_xyz.shape[0],), device=_device)
        self.exposure_mapping = {cam.image_name: i for i, cam in enumerate(cam_infos)}
        self.pretrained_exposures = None

        exposure = torch.eye(3, 4, device=_device)[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        # Initialize accumulators on the configured device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=_device)
        self.denom              = torch.zeros((self.get_xyz.shape[0], 1), device=_device)

        # Build optimizer parameter groups
        params = [
            {'params': [self._xyz],       'lr': training_args.position_lr_init * self.spatial_lr_scale, 'name': 'xyz'},
            {'params': [self._features_dc],    'lr': training_args.feature_lr,                   'name': 'f_dc'},
            {'params': [self._features_rest],  'lr': training_args.feature_lr / 20.0,            'name': 'f_rest'},
            {'params': [self._opacity],   'lr': training_args.opacity_lr, 'name': 'opacity'},
            {'params': [self._scaling],   'lr': training_args.scaling_lr, 'name': 'scaling'},
            {'params': [self._rotation],  'lr': training_args.rotation_lr,'name': 'rotation'},
        ]

        # Choose optimizer
        if self.optimizer_type == "sparse_adam" and SparseGaussianAdam:
            try:
                self.optimizer = SparseGaussianAdam(params, lr=0.0, eps=1e-15)
            except:
                self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        # Exposure optimizer
        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # --- Patched: safely read delay_steps & delay_mult, with defaults ---
        pos_delay_steps = getattr(training_args, 'position_lr_delay_steps', 0)
        pos_delay_mult  = getattr(training_args, 'position_lr_delay_mult', 1.0)
        pos_max_steps   = getattr(training_args, 'position_lr_max_steps', training_args.iterations)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_steps=pos_delay_steps,
            lr_delay_mult=pos_delay_mult,
            max_steps=pos_max_steps
        )

        # Also patch exposure scheduler similarly
        exp_delay_steps = getattr(training_args, 'exposure_lr_delay_steps', 0)
        exp_delay_mult  = getattr(training_args, 'exposure_lr_delay_mult', 1.0)
        exp_max_steps   = getattr(training_args, 'iterations', pos_max_steps)

        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=exp_delay_steps,
            lr_delay_mult=exp_delay_mult,
            max_steps=exp_max_steps
        )

    def update_learning_rate(self, iteration):
        if self.pretrained_exposures is None:
            for g in self.exposure_optimizer.param_groups:
                g['lr'] = self.exposure_scheduler_args(iteration)
        for g in self.optimizer.param_groups:
            if g['name'] == 'xyz':
                lr = self.xyz_scheduler_args(iteration)
                g['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x','y','z','nx','ny','nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz    = self._xyz.detach().cpu().numpy()
        normals= np.zeros_like(xyz)
        f_dc   = self._features_dc.detach().transpose(1,2).flatten(1).cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1,2).flatten(1).cpu().numpy()
        opac   = self._opacity.detach().cpu().numpy()
        scale  = self._scaling.detach().cpu().numpy()
        rot    = self._rotation.detach().cpu().numpy()

        dtype_full = [(a,'f4') for a in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attrs = np.concatenate((xyz, normals, f_dc, f_rest, opac, scale, rot), axis=1)
        elements[:] = list(map(tuple, attrs))
        PlyElement.describe(elements, 'vertex')
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    def reset_opacity(self):
        new_op = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)
        )
        self._opacity = self.replace_tensor_to_optimizer(new_op, "opacity")["opacity"]

    def load_ply(self, path, use_train_test_exp=False):
        plydata = PlyData.read(path)
        # optionally load pretrained exposures
        expo_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
        if use_train_test_exp and os.path.exists(expo_file):
            with open(expo_file) as f:
                expos = json.load(f)
            self.pretrained_exposures = {
                name: torch.FloatTensor(expos[name]).to(_device).requires_grad_(False)
                for name in expos
            }
        else:
            self.pretrained_exposures = None

        # read xyz / features
        xyz = np.stack([
            np.asarray(plydata.elements[0]['x']),
            np.asarray(plydata.elements[0]['y']),
            np.asarray(plydata.elements[0]['z'])
        ], axis=1)
        opacities = np.asarray(plydata.elements[0]['opacity'])[...,None]

        # DC vs rest
        f_dc = np.zeros((xyz.shape[0], 3, 1))
        for i in range(3):
            f_dc[i, i, 0] = np.asarray(plydata.elements[0][f'f_dc_{i}'])
        rest_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')],
                             key=lambda x: int(x.split('_')[-1]))
        feats_rest = np.stack([np.asarray(plydata.elements[0][n]) for n in rest_names], axis=1)
        feats_rest = feats_rest.reshape((xyz.shape[0], 3, -1))

        scales = [np.asarray(plydata.elements[0][n]) for n in sorted([p.name for p in plydata.elements[0].properties if n.startswith('scale_')],
                                                                       key=lambda x: int(x.split('_')[-1]))]
        rots   = [np.asarray(plydata.elements[0][n]) for n in sorted([p.name for p in plydata.elements[0].properties if n.startswith('rot')],
                                                                      key=lambda x: int(x.split('_')[-1]))]

        # register parameters on _device
        self._xyz           = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=_device).requires_grad_(True))
        self._features_dc   = nn.Parameter(torch.tensor(f_dc, dtype=torch.float, device=_device).transpose(1,2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(feats_rest, dtype=torch.float, device=_device).transpose(1,2).contiguous().requires_grad_(True))
        self._opacity       = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=_device).requires_grad_(True))
        self._scaling       = nn.Parameter(torch.tensor(np.stack(scales,1), dtype=torch.float, device=_device).requires_grad_(True))
        self._rotation      = nn.Parameter(torch.tensor(np.stack(rots,1), dtype=torch.float, device=_device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
