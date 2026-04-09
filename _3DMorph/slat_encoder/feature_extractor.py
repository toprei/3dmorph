import os
import sys
import shutil
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open3d as o3d
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils3d

from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir))

from _3DMorph.utils.voxel_utils import VoxelUtils
from _3DMorph.renderer.render_simple import BlenderRenderer

class FeatureExtractor:
    def __init__(self, model_name='dinov2_vitl14_reg', batch_size=16, n_views=10, resolution=518):
        self.model_name = model_name
        self.batch_size = batch_size
        self.n_views = n_views
        self.resolution = resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).eval().to(self.device)
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.n_patch = 518 // 14
        self.renderer = BlenderRenderer(resolution)

    def compute_visible_mask(self, uv01, depth, bin_size=1, eps=1e-6):
        device = uv01.device
        if uv01.dim() == 2:
            uv01 = uv01.unsqueeze(0)
            depth = depth.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        B, N, _ = uv01.shape
        res = self.resolution

        in_front = depth > 0
        u_pix = uv01[..., 0] * (res - 1)
        v_pix = uv01[..., 1] * (res - 1)

        Wb = (res + bin_size - 1) // bin_size
        Hb = (res + bin_size - 1) // bin_size

        ui_b = torch.floor(u_pix / bin_size).long()
        vi_b = torch.floor(v_pix / bin_size).long()

        in_wb = (ui_b >= 0) & (ui_b < Wb)
        in_hb = (vi_b >= 0) & (vi_b < Hb)
        valid = in_front & in_wb & in_hb

        visible = torch.zeros((B, N), dtype=torch.bool, device=device)
        if valid.any():
            flat_b = vi_b * Wb + ui_b
            NB = Hb * Wb
            depth_min = torch.full((B, NB), float('inf'), device=device)
            z_src = depth.clone()
            z_src[~valid] = float('inf')
            idx = flat_b.clamp_min(0).clamp_max(NB - 1)
            depth_min.scatter_reduce_(1, idx, z_src, reduce="amin", include_self=True)
            z_ref = depth_min.gather(1, idx)
            is_best = (depth <= (z_ref + max(eps, 1e-6))) & valid
            visible = is_best

        print(f"[mask] valid={int(valid.sum() / uv01.shape[0])} visible={int(visible.sum() / uv01.shape[0])} bins={Wb}x{Hb}")
        return visible[0] if squeeze_back else visible


    
    def extract_features(self, images, extrinsics, intrinsics, positions):
        N = positions.shape[0]
        positions = positions.float().to(self.device)
        indices = ((positions + 0.5) * 64).long()
        assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
        sum_patchtokens = None
        cnt_per_point = None
        count = 0
        tensor_images = [self.preprocess_image(image) for image in images]

        torch.cuda.empty_cache()

        pbar = tqdm(range(0, self.n_views, self.batch_size), desc="Processing batches")
        for i in pbar:
            batch_images = torch.stack(tensor_images[i:i+self.batch_size]).to(self.device)
            batch_extrinsics = torch.stack(extrinsics[i:i+self.batch_size]).to(self.device)
            batch_intrinsics = torch.stack(intrinsics[i:i+self.batch_size]).to(self.device)

            batch_images = self.transform(batch_images)
            bs = batch_images.size(0)

            with torch.no_grad():
                features = self.model(batch_images, is_training=True)

            uv, depth = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)
            uv01 = uv.clone().detach()

            visible_mask = self.compute_visible_mask(uv01, depth)

            uv = uv01 * 2 - 1
            print(f"uv01 min: {uv01.min().item():.4f}, max: {uv01.max().item():.4f}")
            grid = uv.view(bs, -1, 1, 2)

            patchtokens = features['x_prenorm'][:, self.model.num_register_tokens + 1:] \
                .permute(0, 2, 1).reshape(bs, 1024, self.n_patch, self.n_patch)

            sampled_patchtokens = F.grid_sample(
                patchtokens, grid, mode='bilinear', align_corners=False
            ).squeeze(3).permute(0, 2, 1)

            maskf = visible_mask.to(self.device).float().unsqueeze(2)   # [B,N,1]
            sampled_vis = sampled_patchtokens * maskf                   # [B,N,1024]

            batch_sum = sampled_vis.sum(dim=0)                          # [N,1024]
            batch_cnt = visible_mask.sum(dim=0)                         # [N]

            if sum_patchtokens is None:
                sum_patchtokens = batch_sum
                cnt_per_point = batch_cnt
            else:
                sum_patchtokens += batch_sum
                cnt_per_point += batch_cnt
            
            count += bs

            del batch_images, batch_extrinsics, batch_intrinsics, features, uv, patchtokens, sampled_patchtokens
            torch.cuda.empty_cache()

        # self.model.to('cpu')
        avg_patchtokens = (sum_patchtokens / cnt_per_point.clamp_min(1).unsqueeze(1)).cpu().numpy().astype(np.float16)
        print(f"Avg valid views per point: {(cnt_per_point.float().mean()).item():.2f} / {self.n_views}")

        return {
            'indices': indices.cpu().numpy().astype(np.uint8),
            'patchtokens': avg_patchtokens,
        }
    
    def save_features(self, output_dir, obj_name, features):
        os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
        save_path = os.path.join(output_dir, 'features', f'{obj_name}.npz')
        np.savez_compressed(save_path, **features)
        print(f"Features saved to {save_path}")
        return save_path

    def rotate_and_normalize_mesh(self, mesh, mesh_ref):
        bounding_box = mesh_ref.get_axis_aligned_bounding_box()
        center = bounding_box.get_center()

        def normalize_mesh(mesh, bounding_box, center):
            mesh.translate(-center, relative=True)
            min_bound, max_bound = bounding_box.min_bound, bounding_box.max_bound
            scale = max(max_bound - min_bound)
            mesh.scale(1.0 / scale, center=(0, 0, 0))
            vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
            mesh.vertices = o3d.utility.Vector3dVector(vertices) 
            return mesh
        
        new_mesh = normalize_mesh(mesh, bounding_box, center)
        new_mesh_ref = normalize_mesh(mesh_ref, bounding_box, center)
        new_mesh_unrot = o3d.geometry.TriangleMesh(new_mesh)
        rotation_matrix = new_mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        new_mesh.rotate(rotation_matrix, center=(0, 0, 0))
        return new_mesh_unrot, new_mesh, new_mesh_ref
    
    def blender_c2w_to_w2c(self, c2w):
        c2w = torch.tensor(c2w)
        c2w[:3, 1:3] *= -1
        return torch.inverse(c2w)
    
    def preprocess_image(self, image):
        image = image.resize((518, 518), Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image = image[:, :, :3] * image[:, :, 3:]
        return torch.from_numpy(image).permute(2, 0, 1).float()

    def run_extractor(self, input_mesh_path, size_ref_mesh_path, force_render=False):
        '''
        Runs the feature extraction pipeline on a given mesh and its size reference mesh.
        Args:
            - input_mesh_path (str): Path to the input mesh file.
            - size_ref_mesh_path (Optional, str): Path to the size reference mesh file. If given, the input mesh will be normalized and shifted to match the normalization settings of this reference mesh. (In this way we can ensure the two meshes are perfectly aligned.)
            - force_render (Optional, bool): If True, forces re-rendering of images even if they already exist.
        Returns:
            - str: Path to the saved features .npz file.
        '''
        mesh = o3d.io.read_triangle_mesh(input_mesh_path)

        if size_ref_mesh_path is not None:
            mesh_ref = o3d.io.read_triangle_mesh(size_ref_mesh_path)
            mesh_org, mesh_rot, mesh_ref = self.rotate_and_normalize_mesh(mesh, mesh_ref)
        else:
            mesh_org, mesh_rot, mesh_ref = mesh, mesh, None

        slat_preparation_dir = os.path.join(os.path.dirname(input_mesh_path), 'slat_preparation')
        image_dir = os.path.join(slat_preparation_dir, "blender_render")
        normalized_mesh_path = os.path.join(slat_preparation_dir, os.path.basename(input_mesh_path).replace('.obj', '_normalized_for_slat.obj'))

        if mesh_ref is not None:
            normalized_mesh_ref_path = os.path.join(slat_preparation_dir, os.path.basename(size_ref_mesh_path).replace('.obj', '_normalized_for_slat.obj'))
        else:
            normalized_mesh_ref_path = None
        
        if not os.path.exists(slat_preparation_dir) or force_render:
            if force_render and os.path.exists(slat_preparation_dir):
                shutil.rmtree(slat_preparation_dir)

            print(f"Rendering object {input_mesh_path} for SLAT preparation...")
            os.makedirs(slat_preparation_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(normalized_mesh_path, mesh_org)

            if mesh_ref is not None:
                o3d.io.write_triangle_mesh(normalized_mesh_ref_path, mesh_ref)
            self.renderer.render(normalized_mesh_path, normalized_mesh_ref_path, save_dir=image_dir,n_views=self.n_views, mode='slat', save_transforms=True)

        else:
            o3d.io.write_triangle_mesh(normalized_mesh_path, mesh_org)

        images = [Image.open(os.path.join(image_dir, f'render_{i:03d}.png')) for i in range(self.n_views)]

        with open(os.path.join(image_dir, 'transforms.json'), 'r') as f:
            transforms = json.load(f)

        fov = torch.tensor([transform['camera_angle_x'] for transform in transforms[:self.n_views]])
        intrinsics = [utils3d.torch.intrinsics_from_fov_xy(fov_value, fov_value).clone().detach().float() for fov_value in fov]
        extrinsics = [self.blender_c2w_to_w2c(transform['transform_matrix']).clone().detach().float() for transform in transforms[:self.n_views]]

        positions = torch.tensor(VoxelUtils.get_voxelized_vertices(mesh_rot)).float()
        utils3d.io.write_ply(normalized_mesh_path.replace('_normalized_for_slat.obj','_voxels_for_slat.ply'), np.array(positions))

        features = self.extract_features(images, extrinsics, intrinsics, positions)
        obj_name = os.path.splitext(os.path.basename(input_mesh_path))[0]
        output_dir = os.path.dirname(input_mesh_path)
        torch.cuda.empty_cache()

        return self.save_features(output_dir, obj_name, features)


if __name__ == "__main__":
    from pathlib import Path

    example_name = "car-luggage-rack"  # TODO: Pick an example from assets/example_objects
    root_dir = Path(__file__).parents[2]

    input_mesh_path = root_dir / "assets" / "example_objects" / example_name / "unmodified.obj"
    ref_mesh_path = None
    extractor = FeatureExtractor(batch_size=50, n_views=10)
    extractor.run_extractor(str(input_mesh_path), ref_mesh_path, force_render=False)
