import os
import sys
import shutil
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import *
import logging
import csv
import imageio
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import trimesh

os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from pathlib import Path
root_dir = Path(__file__).parents[2]
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))

from trellis.pipelines import _3DMorphInpaint
from trellis.utils import render_utils, postprocessing_utils
from trellis.modules import sparse as sp
from _3DMorph.renderer.render_simple import BlenderRenderer
from _3DMorph.utils.metrics_utils import MeshComparison
from _3DMorph.utils.voxel_utils import VoxelUtils


class TrellisInpaintPipeline:
    def __init__(self, output_path, work_dir, render_params={"offset": (-45, 30), "set_fov": 30}, steps=12, resolution=1024, mode="hybrid", seed=1):
        self.output_path = output_path
        self.work_dir = work_dir
        self.render_params = render_params
        self.resolution = resolution
        self.steps = steps
        self.mode = mode
        self.seed = seed
        self.folder_name = self._setup_environment()
        self.parent_folder_name = self.folder_name
        self._setup_logging()

    def _setup_environment(self):
        local_tz = ZoneInfo('Europe/Berlin')
        local_time = datetime.now().astimezone(local_tz)
        time = local_time.strftime("%b%d_%H:%M").upper()
        mesh_id = os.path.basename(self.work_dir).split('/')[0]
        folder_name = os.path.join(
            self.output_path,
            f"{time}_bsl5_{mesh_id}")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name

    def _setup_logging(self):
        log_file = os.path.join(self.folder_name, "log.txt")
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def set_work_dir(self, work_dir):
        mesh_id = self.make_save_name(os.path.basename(work_dir))
        folder_name = os.path.join(
            self.parent_folder_name,
            f"{mesh_id}_cam_{self.render_params['offset'][0]}_{self.render_params['offset'][1]}")
        os.makedirs(folder_name, exist_ok=True)
        self.folder_name = folder_name
        self.make_output_folders()

    def make_output_folders(self):
        self.gt_dir = os.path.join(self.folder_name, "gt")
        os.makedirs(self.gt_dir, exist_ok=True)
        self.inputs_dir = os.path.join(self.folder_name, "inputs")
        os.makedirs(self.inputs_dir, exist_ok=True)
        self.voxel_dir = os.path.join(self.folder_name, "voxels")
        os.makedirs(self.voxel_dir, exist_ok=True)
        self.outputs_dir = os.path.join(self.folder_name, "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.metrics_dir = os.path.join(self.folder_name, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.scripts_dir = os.path.join(self.folder_name, "scripts")
        os.makedirs(self.scripts_dir, exist_ok=True)
        self.graphs_dir = os.path.join(self.folder_name, "graphs")
        os.makedirs(self.graphs_dir, exist_ok=True)

    def load_pipeline(self):
        logging.info("Loading pretrained pipeline...")
        self.inpaint_pipeline = _3DMorphInpaint.from_pretrained(str(root_dir / "pretrained_weights" / "TRELLIS-image-large"))
        self.inpaint_pipeline.cuda()
        logging.info(f"Loaded models: {list(self.inpaint_pipeline.models.keys())}")

    def make_save_name(self, mesh_id: str) -> str:
        parts = mesh_id.split("_")
        obj_id = parts[0]
        if len(parts) == 3:
            return f"{obj_id}_{parts[-1]}"
        return obj_id

    def set_cam_offset(self, offset=(-45, 20)):
        self.render_params['offset'] = offset

    def render_mesh(self, input_mesh_path, n_views=1):

        renderer = BlenderRenderer(self.resolution)
        view_args = self.render_params if self.render_params else {
            'offset': (-45, 30),
            'step': 3,
            'direction': 'left',
            'set_fov': 30
        }
        images = renderer.render(input_mesh_path, save_dir=self.inputs_dir, n_views=n_views, mode='linear', lin_view_args=view_args)
        img = images[0]
        logging.info(f"Rendered mesh using blender.")
        del images
        return img

    def load_gt_slat(self, gt_slat_path):
        feats = np.load(gt_slat_path)
        slat = sp.SparseTensor(
            feats=torch.from_numpy(feats['feats']).float(),
            coords = torch.cat([
                        torch.zeros(feats['feats'].shape[0], 1).int(),
                        torch.from_numpy(feats['coords']).int(),
                    ], dim=1),
        ).cuda()
        
        return slat
    
    def extract_boundary(self, mask):
        assert mask.ndim == 5, "Input mask must be 5D tensor"

        kernel = torch.ones((1, 1, 3, 3, 3), device=mask.device)
        kernel[0, 0, 1, 1, 1] = 0
        mask = mask[:, :1, :, :, :]
        conv = F.conv3d(mask.float(), kernel, padding=1)

        boundary = (mask == 1) & (conv < 17)

        return boundary.float()
    
    def densify_voxel(self, input_voxel_path):
        gt_voxel_grid = VoxelUtils.mesh_voxel_to_dense_voxel_grid(input_voxel_path)
        
        return gt_voxel_grid.cuda()
    
    def calcualate_mask(self, voxel1, voxel2):
        '''
        Calculate the mask from two voxel grids
        voxel1 & voxel2: [1, 1, D, H, W]
        Divide the space into 8 regions (2x2x2), and activate the mask for any region with changes.
        The final output is [1, 8, 16, 16, 16].
        '''
        change_mask = (voxel2 - voxel1).abs().float()
        print(f"active voxels in voxel1: {voxel1.sum().int()}")
        mask_downsampled = F.avg_pool3d(change_mask, kernel_size=8, stride=8)
        mask_downsampled = (mask_downsampled > 1 / 256).float()
        mask_upsampled = F.interpolate(mask_downsampled, size=(16, 16, 16), mode='nearest')
        mask_upsampled = mask_upsampled.repeat(1, 8, 1, 1, 1)

        return mask_upsampled
    

    def calcualate_mask_bb_box(self, voxel1, voxel2):
        '''
        Calculate a bounding box mask covering all changed voxels.
        voxel1 & voxel2: [1, 1, D, H, W]
        Return: [1, 8, 16, 16, 16] with a bounding box region as 1, others as 0
        '''
        change_mask = (voxel2 - voxel1).abs().float()
        mask_downsampled = F.avg_pool3d(change_mask, kernel_size=8, stride=8)
        mask_downsampled = (mask_downsampled > 1 / 256).float()
        mask_upsampled = F.interpolate(mask_downsampled, size=(16, 16, 16), mode='nearest')
        mask_upsampled = mask_upsampled.repeat(1, 8, 1, 1, 1)
        nonzero_coords = (mask_upsampled > 0).nonzero(as_tuple=False)[:, -3:]

        zmin, ymin, xmin = nonzero_coords.min(dim=0).values.tolist()
        zmax, ymax, xmax = nonzero_coords.max(dim=0).values.tolist()

        bbox_mask = torch.zeros_like(mask_upsampled)
        bbox_mask[..., zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = 1.0
        return bbox_mask

    def compare_meshes(self, gt_mesh_path, pred_mesh_path, input_mesh_name, rotate_axis):
        logging.info(f"Starting mesh comparison for {input_mesh_name}...")
        print(f"Comparing {gt_mesh_path} and {pred_mesh_path}")
        comparator = MeshComparison()
        results, rotaion = comparator.compare(gt_mesh_path, pred_mesh_path, mode='normalize&icp', mute=True, deformation=False, rotate_axis=rotate_axis)
        logging.info(f"Comparison Results for {input_mesh_name}: {results} with rotation {rotaion} degrees")
        results = {key: round(value, 6) for key, value in results.items()}

        results_file = os.path.join(self.metrics_dir, f"{input_mesh_name}_results.txt")
        with open(results_file, "w") as f:
            f.write("Comparison Results:\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
                f.write(f"Rotation: {rotaion} degrees\n")
        return results, rotaion
    
    def compare_voxels_in_bb_box(self, gt_coords, pred_coords, mask, input_mesh_name, rotate_axis='none'):
        logging.info(f"Starting voxel comparison in BB box for {input_mesh_name}...")
        bb_box = F.interpolate(mask, size=(64, 64, 64), mode='nearest')
        gt_coords = gt_coords * bb_box
        pred_coords = VoxelUtils.sparse_to_dense_grid(pred_coords) * bb_box
        gt_mesh = VoxelUtils.visualize_dense_voxels(save_dir=None, input_mesh_name=None, dense_voxel_grid=gt_coords, ret_trimesh=True)
        pred_mesh = VoxelUtils.visualize_dense_voxels(save_dir=None, input_mesh_name=None, dense_voxel_grid=pred_coords, ret_trimesh=True)

        cropped_dir = os.path.join(self.voxel_dir, "cropped_voxel")
        os.makedirs(cropped_dir, exist_ok=True)
        VoxelUtils.visualize_dense_voxels(cropped_dir, input_mesh_name + '_gt_cropped_voxel', gt_coords, ret_mode=['ply'], normalize=False)

        VoxelUtils.visualize_dense_voxels(cropped_dir, input_mesh_name + '_pred_cropped_voxel', pred_coords, ret_mode=['ply'], normalize=False)
        comparator = MeshComparison()
        results, _ = comparator.compare(gt_mesh, pred_mesh, mode='normalize&icp', mute=True, deformation=False, rotate_axis=rotate_axis)

        results = {key: round(value, 6) for key, value in results.items()}

        results_file = os.path.join(self.metrics_dir, f"{input_mesh_name}_voxel_bb_box_results.txt")
        with open(results_file, "w") as f:
            f.write("Comparison Results:\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        return results
    
    def compare_meshes_in_bb_box(self, gt_mesh_path, pred_mesh_path, mask, input_mesh_name, rotate_axis='none'):
        logging.info(f"Starting mesh comparison in BB box for {input_mesh_name}...")
        bb_box = F.interpolate(mask, size=(64, 64, 64), mode='nearest')
        bb_box_mesh = VoxelUtils.visualize_dense_voxels(
            save_dir=None,
            input_mesh_name=None,
            dense_voxel_grid=bb_box,
            ret_trimesh=True
        )
        norm_bb_box_mesh = VoxelUtils.normalize_dense_grid_mesh_to_one(bb_box_mesh)
        norm_bb_box_mesh.export(os.path.join(self.voxel_dir, input_mesh_name + '_bb_box_mesh.ply'))

        bb_min, bb_max = norm_bb_box_mesh.bounds
        eps = 1e-9

        gt_mesh = trimesh.load(gt_mesh_path, process=False)
        pred_mesh = trimesh.load(pred_mesh_path, process=False)

        def _clip_triangle_to_aabb(tri, bb_min, bb_max, eps=1e-9):
            poly = tri.tolist()
            for axis in range(3):
                for bound, keep_ge in [(bb_min[axis], True), (bb_max[axis], False)]:
                    if not poly: break
                    new = []
                    for i in range(len(poly)):
                        A, B = np.array(poly[i]), np.array(poly[(i+1)%len(poly)])
                        Ain = (A[axis] >= bound-eps) if keep_ge else (A[axis] <= bound+eps)
                        Bin = (B[axis] >= bound-eps) if keep_ge else (B[axis] <= bound+eps)
                        if Ain: new.append(A)
                        if Ain ^ Bin:
                            t = (bound-A[axis])/(B[axis]-A[axis]+1e-12)
                            new.append(A + t*(B-A))
                    poly = new
            return np.array(poly) if len(poly) >= 3 else np.zeros((0,3))

        def crop_mesh_to_aabb(mesh, bb_min, bb_max, eps=1e-9):
            if mesh.is_empty: 
                return trimesh.Trimesh(vertices=[], faces=[], process=False)

            V, F = mesh.vertices, mesh.faces
            new_V, new_F = [], []

            inside = ((V[:,0]>=bb_min[0]-eps)&(V[:,0]<=bb_max[0]+eps)&
                    (V[:,1]>=bb_min[1]-eps)&(V[:,1]<=bb_max[1]+eps)&
                    (V[:,2]>=bb_min[2]-eps)&(V[:,2]<=bb_max[2]+eps))
            fully_in = np.all(inside[F], axis=1)

            for fi in np.nonzero(fully_in)[0]:
                tri = V[F[fi]]
                base = len(new_V)
                new_V.extend(tri)
                new_F.append([base, base+1, base+2])

            for fi in np.nonzero(~fully_in)[0]:
                poly = _clip_triangle_to_aabb(V[F[fi]], bb_min, bb_max, eps)
                if len(poly) >= 3:
                    base = len(new_V)
                    new_V.extend(poly)
                    for k in range(1, len(poly)-1):
                        new_F.append([base, base+k, base+k+1])

            return trimesh.Trimesh(vertices=np.array(new_V), faces=np.array(new_F), process=False)
        
        gt_cropped = crop_mesh_to_aabb(gt_mesh, bb_min, bb_max, eps)
        pred_cropped = crop_mesh_to_aabb(pred_mesh, bb_min, bb_max, eps)
        
        cropped_dir = os.path.join(self.voxel_dir, "cropped_mesh")
        os.makedirs(cropped_dir, exist_ok=True)
        gt_cropped_path = os.path.join(cropped_dir, f"{input_mesh_name}_gt_cropped_mesh.ply")
        pred_cropped_path = os.path.join(cropped_dir, f"{input_mesh_name}_pred_cropped_mesh.ply")
        gt_cropped.export(gt_cropped_path)
        if pred_cropped.vertices.shape[0] == 0:
            trimesh.Trimesh(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int64),
                process=False
            ).export(pred_cropped_path)
        else:
            pred_cropped.export(pred_cropped_path)

        comparator = MeshComparison()
        results, rotation = comparator.compare(
            gt_cropped_path,
            pred_cropped_path,
            mode='normalize&icp',
            mute=True,
            deformation=False,
            rotate_axis=rotate_axis
        )

        results = {k: round(v, 6) for k, v in results.items()}
        results_file = os.path.join(self.metrics_dir, f"{input_mesh_name}_mesh_bb_box_results.txt")
        with open(results_file, "w") as f:
            f.write("Comparison Results (inside AABB of bb_box):\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Rotation: {rotation} degrees\n")
        return results, rotation

    def save_results_to_csv(self, results, rotation, input_mesh_name):
        csv_file = os.path.join(self.metrics_dir, "results.csv")
        file_exists = os.path.isfile(csv_file)

        fieldnames = ['Mesh Name', 'Renderer Type', 'Rotation', 'Denoise_steps'] + list(results.keys())

        with open(csv_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row = {'Mesh Name': input_mesh_name, 'Renderer Type': 'blender', 'Rotation': rotation, 'Denoise_steps': self.steps}
            row.update(results)

            writer.writerow(row)
        
    def export_mesh(self, outputs, mesh_name):
        vertices = outputs['mesh'][0].vertices.cpu().numpy()
        faces = outputs['mesh'][0].faces.cpu().numpy()
        
        vertices, faces = postprocessing_utils.postprocess_mesh(vertices, faces, simplify_ratio=0.8)

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces
            )
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.radians(-90),
            direction=[1, 0, 0],
            point=[0, 0, 0]
        )

        mesh.apply_transform(rotation_matrix)
        mesh.export(self.folder_name + "/outputs/mesh_output_" + mesh_name + ".ply")
        return self.folder_name + "/outputs/mesh_output_" + mesh_name + ".ply"
        
    def run_pipeline(self, image1, input_mesh_name, gt_slat, mask):
        logging.info(f"Running the pipeline for {input_mesh_name}...")   
        interpolated_mask = F.interpolate(mask, size=(64, 64, 64), mode='nearest')
        original_binary_ss_map = VoxelUtils.sparse_to_dense_grid(gt_slat.coords)
        VoxelUtils.visualize_dense_voxels(self.voxel_dir, input_mesh_name + '_slat_coords', original_binary_ss_map, ret_mode=['ply'], normalize=False)
        cond = self.inpaint_pipeline.get_cond([image1])
        coords = self.inpaint_pipeline.sparse_structure_inpaint(
            cond,
            original_binary_ss_map=original_binary_ss_map,
            mask=mask,
            num_samples=1,
            seed=self.seed,
            mode=self.mode,
            sparse_structure_sampler_params={
                "steps": self.steps,
                "cfg_strength": 7.5,
            }
        )

        sparse_grid_mask = VoxelUtils.dense_to_sparse_grid(interpolated_mask)
        boundary_mask = self.extract_boundary(interpolated_mask)
        pred_voxel_path = VoxelUtils.visualize_sparse_voxels(self.voxel_dir, input_mesh_name, coords, normalize=False)
        mask_voxel = VoxelUtils.visualize_dense_voxels(self.voxel_dir, "mask", boundary_mask, ret_mode=['ply'], normalize=False)
        VoxelUtils.merge_meshes([mask_voxel, pred_voxel_path], os.path.join(self.voxel_dir, input_mesh_name + "_masked_output.ply"))

        new_slat = self.inpaint_pipeline.second_stage_slat_inpaint(
            cond,
            coords, 
            slat=gt_slat,
            sparse_grid_mask=sparse_grid_mask,
            seed=self.seed,
            mode=self.mode,
            slat_sampler_params={
                "steps": self.steps,
                "cfg_strength": 3.0,
            },
        )

        outputs = self.inpaint_pipeline.decode_slat(new_slat, formats=["mesh"])
        pred_slat = {'feats': new_slat.feats.cpu().numpy().astype(np.float32),
                'coords': new_slat.coords[:, 1:].cpu().numpy().astype(np.uint8)}
        np.savez_compressed(os.path.join(self.outputs_dir, "pred_slat"), **pred_slat)
        return pred_voxel_path, outputs, coords
    
    def run_for_meshes(self, modified_mesh_path, unmodified_mesh_path):
        unmodified_mesh_name = os.path.splitext(os.path.basename(unmodified_mesh_path))[0]
        modified_mesh_name = os.path.splitext(os.path.basename(modified_mesh_path))[0]

        unmodified_voxel_path = unmodified_mesh_path.replace('.obj', '_voxels.ply')
        modified_voxel_path = modified_mesh_path.replace('.obj', '_voxels.ply')
        normalized_modified_mesh_path = os.path.join(os.path.dirname(modified_mesh_path), 'slat_preparation', modified_mesh_name + '_normalized_for_slat.obj')
        unmodified_mesh_slat_path = os.path.join(os.path.dirname(unmodified_mesh_path), 'features', unmodified_mesh_name + '_slat.npz')

        mesh_id = os.path.dirname(unmodified_mesh_path).split('/')[-1]
        save_name = self.make_save_name(mesh_id)

        shutil.copy(unmodified_voxel_path, os.path.join(self.gt_dir, unmodified_mesh_name + '_gt_voxels.ply'))
        shutil.copy(modified_voxel_path, os.path.join(self.gt_dir, modified_mesh_name + '_gt_voxels.ply'))
        shutil.copy(unmodified_mesh_path, self.gt_dir)
        shutil.copy(modified_mesh_path, self.gt_dir)
        image1 = self.render_mesh(modified_mesh_path)
        unmodified_mesh_slat = self.load_gt_slat(unmodified_mesh_slat_path)

        unmodified_dense_voxel = self.densify_voxel(unmodified_voxel_path)
        modified_dense_voxel = self.densify_voxel(modified_voxel_path)
        mask = self.calcualate_mask_bb_box(unmodified_dense_voxel, modified_dense_voxel)
        pred_voxel_path, outputs, coords = self.run_pipeline(image1, save_name, unmodified_mesh_slat, mask)
        results=self.compare_voxels_in_bb_box(modified_dense_voxel, coords, mask, save_name, rotate_axis='none')
        self.save_results_to_csv(results, "none", save_name + '_bb_box_voxel')
        results, rotation = self.compare_meshes(modified_voxel_path, pred_voxel_path, save_name + '_voxel', rotate_axis='none')
        self.save_results_to_csv(results, rotation, save_name + '_voxel')
        pred_mesh_path = self.export_mesh(outputs, save_name)
        results, rotation = self.compare_meshes_in_bb_box(normalized_modified_mesh_path, pred_mesh_path, mask, save_name, rotate_axis='none')
        self.save_results_to_csv(results, rotation, save_name + '_bb_box_mesh')
        results, rotation = self.compare_meshes(normalized_modified_mesh_path, pred_mesh_path, save_name + '_mesh', rotate_axis='none')
        self.save_results_to_csv(results, rotation, save_name + '_mesh')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Trellis Inpainting Pipeline (benchmark / test)")
    parser.add_argument("--mode", choices=["benchmark", "test"], default="benchmark",
                        help="benchmark = run the whole dataset, test = run a single folder")
    parser.add_argument("--dataset_dir", type=str, default="assets/Assembly_Pairs",
                        help="Dataset root path (only required for benchmark mode)")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Single work directory (only required for test mode)")
    parser.add_argument("--output_path", type=str, default="results/ours",
                        help="Where to save results")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--sampler_mode", type=str, default="hybrid", choices=["hybrid", "hard", "repaint"])

    args = parser.parse_args()

    render_params = {"offset": (-45, 30), "set_fov": 30}
    if args.mode == "benchmark":
        if args.dataset_dir is None:
            raise ValueError("You must provide --dataset_dir in benchmark mode")
        with torch.no_grad():
            dataset_range = range(0, 11)
            total_dirs = 0
            for i in dataset_range:
                parent_dir = os.path.join(args.dataset_dir, f"Assembly_Dataset_{i}")
                sub_dirs = [
                    sub_dir for sub_dir in os.listdir(parent_dir)
                    if os.path.isdir(os.path.join(parent_dir, sub_dir))
                ]
                total_dirs += len(sub_dirs)
            print(f"Found {total_dirs} sub-directories in {args.dataset_dir}")

            with tqdm(total=total_dirs * 12, desc="Processing folders") as pbar:
                for i in dataset_range:
                    parent_dir = os.path.join(args.dataset_dir, f"Assembly_Dataset_{i}")
                    pipeline = TrellisInpaintPipeline(
                        args.output_path, 
                        parent_dir,
                        render_params=render_params, 
                        steps=12, 
                        seed=args.seed, 
                        resolution=args.resolution, 
                        mode=args.sampler_mode
                    )
                    pipeline.load_pipeline()
                    logging.info("Pipeline loaded successfully.")

                    offsets = [(-45 + 30 * j, 30) for j in range(12)]
                    for offset in offsets:
                        pipeline.set_cam_offset(offset)
                        sub_dirs = [
                            sub_dir for sub_dir in os.listdir(parent_dir)
                            if os.path.isdir(os.path.join(parent_dir, sub_dir))
                        ]

                        for sub_dir in sub_dirs:
                            work_dir = os.path.join(parent_dir, sub_dir)
                            if work_dir == parent_dir or not os.path.isdir(work_dir):
                                continue

                            input_mesh_path_list = [
                                os.path.join(work_dir, file)
                                for file in os.listdir(work_dir)
                                if file.endswith(".obj")
                            ]
                            input_mesh_path_list.sort(
                                key=lambda path: int(os.path.basename(path).split("_")[-1].split(".")[0])
                                if os.path.basename(path).split("_")[-1].split(".")[0].isdigit()
                                else float("inf")
                            )
                            pairs = [(input_mesh_path_list[j], input_mesh_path_list[j - 1])
                                    for j in range(1, len(input_mesh_path_list))]

                            pipeline.set_work_dir(work_dir)
                            for img_cond_mesh_path, noise_embed_mesh_path in pairs:
                                pipeline.run_for_meshes(img_cond_mesh_path, noise_embed_mesh_path)
                                torch.cuda.empty_cache()

                            pbar.update(1)

    elif args.mode == "test":
        if args.work_dir is None:
            raise ValueError("You must provide --work_dir in test mode")
        with torch.no_grad():
            input_mesh_path_list = [
                os.path.join(args.work_dir, file)
                for file in os.listdir(args.work_dir)
                if file.endswith(".obj")
            ]
            input_mesh_path_list.sort(
                key=lambda path: int(os.path.basename(path).split("_")[-1].split(".")[0])
                if os.path.basename(path).split("_")[-1].split(".")[0].isdigit()
                else float("inf")
            )
            pairs = [(input_mesh_path_list[i], input_mesh_path_list[i - 1])
                    for i in range(1, len(input_mesh_path_list))]


            pipeline = TrellisInpaintPipeline(
                        args.output_path, 
                        args.work_dir,
                        render_params=render_params, 
                        steps=12, 
                        seed=args.seed, 
                        resolution=args.resolution, 
                        mode=args.sampler_mode
                    )
            pipeline.load_pipeline()

            offsets = [(-45 + 30 * j, 30) for j in range(12)]
            for offset in tqdm(offsets, desc="Processing camera angles"):
                pipeline.set_cam_offset(offset)
                pipeline.set_work_dir(args.work_dir)
                for modified_mesh_path, unmodified_mesh_path in pairs:
                    pipeline.run_for_meshes(modified_mesh_path, unmodified_mesh_path)
                    torch.cuda.empty_cache()