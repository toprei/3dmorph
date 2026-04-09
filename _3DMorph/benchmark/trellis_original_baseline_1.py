import os
import sys
import shutil
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

os.system("Xvfb :99 -screen 0 1024x768x16 &")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from pathlib import Path
root_dir = Path(__file__).parents[2]
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.modules import sparse as sp
from _3DMorph.renderer.render_simple import BlenderRenderer
from _3DMorph.utils.metrics_utils import MeshComparison
from _3DMorph.utils.voxel_utils import VoxelUtils

class TrellisFirstStagePipeline:
    def __init__(self, output_path, work_dir, render_params,  steps=12, cfg=7.5):
        self.output_path = output_path
        self.work_dir = work_dir
        self.render_params = render_params
        self.steps = steps
        self.cfg = cfg
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
            f"{time}_bsl1_{mesh_id}")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name

    def _setup_logging(self):
        log_file = os.path.join(self.folder_name, "log.txt")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w"),
            ]
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
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
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(str(root_dir / "pretrained_weights" / "TRELLIS-image-large"))
        self.pipeline.cuda()
        logging.info(f"Loaded models: {list(self.pipeline.models.keys())}")
        
    def make_save_name(self, mesh_id: str) -> str:
        parts = mesh_id.split("_")
        obj_id = parts[0]
        if len(parts) == 3:
            return f"{obj_id}_{parts[-1]}"
        return obj_id

    def set_cam_offset(self, offset=(-45, 20)):
        self.render_params['offset'] = offset

    def render_mesh(self, input_mesh_path, n_views=1):

        renderer = BlenderRenderer(resolution=1024)
        view_args = self.render_params if self.render_params else {
            'offset': (-45, 20),
            'step': 3,
            'direction': 'left',
            'set_fov': 30
        }
        images = renderer.render(input_mesh_path, save_dir=self.inputs_dir, n_views=n_views, mode='linear', lin_view_args=view_args)
        
        logging.info(f"Rendered mesh using blender.")

        return images[0]
    
    def voxelize_mesh(self, input_voxel_path):
        gt_voxel_grid = VoxelUtils.mesh_to_dense_voxel_grid(input_voxel_path)
        return gt_voxel_grid.cuda()


    def run_pipeline(self, image, input_mesh_name):
        logging.info(f"Running the pipeline for {input_mesh_name}...")   

        coords = self.pipeline.run_first_stage(
            image,
            seed=1,
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
        )
        torch.cuda.empty_cache()

        outputs = self.pipeline.run_sencond_stage(
            image, 
            coords, 
            slat_sampler_params={
                "steps": self.steps,
                "cfg_strength": self.cfg,
            },
            formats=["mesh", "gaussian"],
        )
        
        pred_voxel_path = VoxelUtils.visualize_sparse_voxels(self.voxel_dir, input_mesh_name, coords, normalize=False)
        return pred_voxel_path, outputs

    def compare_meshes(self, gt_mesh_path, pred_mesh_path, input_mesh_name, rotate_axis):
        logging.info(f"Starting mesh comparison for {input_mesh_name}...")
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
        logging.info(f"Saved comparison results to {results_file}")
        return results, rotaion

    def save_results_to_csv(self, results, rotation, input_mesh_name):
        csv_file = os.path.join(self.metrics_dir, "results.csv")
        file_exists = os.path.isfile(csv_file)

        fieldnames = ['Mesh Name', 'Renderer Type', 'Rotation', 'Denoise_steps', 'CFG_strength'] + list(results.keys())

        with open(csv_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row = {'Mesh Name': input_mesh_name, 'Renderer Type': 'blender', 'Rotation': rotation, 'Denoise_steps': self.steps, 'CFG_strength': self.cfg}
            row.update(results)

            writer.writerow(row)
        
        logging.info(f"Saved comparison results for {input_mesh_name} to {csv_file}")

    def export_mesh(self, outputs, mesh_name):
        glb = False
        if glb:
            with torch.no_grad():
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,          # Ratio of triangles to remove in the simplification process
                    texture_size=512,      # Size of the texture used for the GLB
                )

                # glb.export(self.folder_name + "/outputs/mesh_output_" + mesh_name + ".glb")
                glb.export(self.folder_name + "/outputs/mesh_output_" + mesh_name + ".ply")
                
                if self.render_video:
                    video = render_utils.render_video(outputs['gaussian'][0])['color']
                    imageio.mimsave(f"{self.folder_name}/outputs/mesh_output_{mesh_name}.mp4", video, fps=30)
                    logging.info(f"Saved video output")
        else:
            vertices = outputs['mesh'][0].vertices.cpu().numpy()
            faces = outputs['mesh'][0].faces.cpu().numpy()
            
            vertices, faces = postprocessing_utils.postprocess_mesh(vertices, faces, simplify_ratio=0.9)

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
            

    def run_for_mesh(self, input_mesh_path):
        mesh_id = os.path.dirname(input_mesh_path).split('/')[-1]
        save_name = self.make_save_name(mesh_id)
        logging.info(f"Processing: {save_name}")
        shutil.copy(input_mesh_path, self.gt_dir)
        image = self.render_mesh(input_mesh_path)
        gt_voxel_path = input_mesh_path.replace('.obj', '_voxels.ply')
        pred_voxel_path, outputs = self.run_pipeline(image, save_name)
        results, rotation = self.compare_meshes(gt_voxel_path, pred_voxel_path, save_name + '_voxel', rotate_axis='y')
        pred_mesh_path = self.export_mesh(outputs, save_name)
        self.save_results_to_csv(results, rotation, save_name + '_voxel')
        results, rotation = self.compare_meshes(input_mesh_path, pred_mesh_path, save_name + '_mesh', rotate_axis='y')
        self.save_results_to_csv(results, rotation, save_name + '_mesh')

if __name__ == "__main__":
    output_path = str(root_dir / "results" / "ours")
    # work_dir = str(root_dir / "assets" / "Assembly_Pairs" / "Assembly_Dataset_0" / "20500")
    dataset_dir = str(root_dir / "assets" / "Assembly_Pairs")

    if 'work_dir' not in locals():
        pbar = tqdm(range(0, 10), desc=f"Processing {dataset_dir}", unit="subdir")
        for i in pbar:
            parent_dir = os.path.join(dataset_dir, f"Assembly_Dataset_{i}")
            render_params = {
                "offset": (-45, 30),
                "step": 3, # Viewing angle step for multi-views
                "direction": 'left',
                "set_fov": 30
            }
            pipeline = TrellisFirstStagePipeline(output_path, parent_dir, render_params, steps=12, cfg=7.5) 
            pipeline.load_pipeline()
            logging.info(f"Pipeline loaded successfully.")
            offsets = [(-45 + 30*i, 30) for i in range(12)]
            for offset in offsets:
                pipeline.set_cam_offset(offset)
                sub_dirs = [sub_dir for sub_dir in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, sub_dir))]

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
                        key=lambda path: int(os.path.basename(path).split('_')[-1].split('.')[0]) 
                        if os.path.basename(path).split('_')[-1].split('.')[0].isdigit() 
                        else float('inf')
                    )
                    pairs = [(input_mesh_path_list[i], input_mesh_path_list[i - 1]) for i in range(1, len(input_mesh_path_list))]

                    pipeline.set_work_dir(work_dir)

                    for img_cond_mesh_path, _ in pairs:
                        torch.cuda.empty_cache()
                        try:
                            pipeline.run_for_mesh(img_cond_mesh_path)
                        except RuntimeError as e:
                            if "CUDA" in str(e):
                                print(f"CUDA RuntimeError: {e}. Skipping this pair.")



    else:
        input_mesh_path_list = [
            os.path.join(work_dir, file)
            for file in os.listdir(work_dir)
            if file.endswith(".obj")
        ]


        input_mesh_path_list.sort(
            key=lambda path: int(os.path.basename(path).split('_')[-1].split('.')[0]) 
            if os.path.basename(path).split('_')[-1].split('.')[0].isdigit() 
            else float('inf')
        )
        pairs = [(input_mesh_path_list[i], input_mesh_path_list[i - 1]) for i in range(1, len(input_mesh_path_list))]

        render_params = {
            "offset": (-30, 30),
            "step": 3, # Viewing angle step for multi-views
            "direction": 'left',
            "set_fov": 30
        }
        pipeline = TrellisFirstStagePipeline(output_path, work_dir, render_params, steps=12, cfg=7.5)
        pipeline.load_pipeline()
        pipeline.set_work_dir(work_dir)
        logging.info(f"Pipeline loaded successfully.")
        pbar = tqdm(pairs, desc=f"Processing meshes in {work_dir}", unit="mesh")
        for modified_mesh_path, unmodified_mesh_path in pbar:
            torch.cuda.empty_cache()
            pipeline.run_for_mesh(modified_mesh_path)