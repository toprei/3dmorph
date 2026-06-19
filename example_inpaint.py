import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

os.system("pgrep -f 'Xvfb :99' >/dev/null || Xvfb :99 -screen 0 1024x768x16 &")
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

from trellis.pipelines import _3DMorphInpaint
from trellis.utils import postprocessing_utils
from trellis.modules import sparse as sp
from _3DMorph.utils.mask_predictor import MaskPredictor
from _3DMorph.utils.work_assets import WorkAssets

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # enable if needed for debugging


def _load_slat(npz_path: str) -> sp.SparseTensor:
    '''Load slat from .npz and move them to CUDA. '''
    feats = np.load(npz_path)
    coords = torch.from_numpy(feats["coords"]).int()
    feats_t = torch.from_numpy(feats["feats"]).float()
    batch_col = torch.zeros(feats_t.shape[0], 1).int()
    coords_batched = torch.cat([batch_col, coords], dim=1)
    return sp.SparseTensor(feats=feats_t, coords=coords_batched).cuda()

def prepare_assets(work_dir: str) -> WorkAssets:
    '''Parse work directory, collect all paths and load SLAT. '''
    work = Path(work_dir)

    mesh_path = str(work / "unmodified.obj")
    unmod_img_path = str(work / "explore_inpaint" / "unmodified.png")
    mod_img_path = str(work / "explore_inpaint" / "modified.png")
    transforms_json = str(work / "explore_inpaint" / "transforms.json")
    slat_npz_path = str(work / "features" / "unmodified_slat.npz")

    slat = _load_slat(slat_npz_path)
    mesh_id = Path(mesh_path).parent.name

    return WorkAssets(
        mesh_path=mesh_path,
        unmod_img_path=unmod_img_path,
        mod_img_path=mod_img_path,
        transforms_json=transforms_json,
        slat=slat,
        save_name=mesh_id,
    )


def rotate_mesh_x_neg90(m: trimesh.Trimesh) -> None:
    '''Rotate mesh -90 degrees around X-axis. '''
    rot = trimesh.transformations.rotation_matrix(
        angle=np.radians(-90), direction=[1, 0, 0], point=[0, 0, 0]
    )
    m.apply_transform(rot)


def main():
    # Initialize pretrained pipeline
    pipeline = _3DMorphInpaint.from_pretrained("pretrained_weights/TRELLIS-image-large")
    pipeline.cuda()

    # Load assets from work directory
    example_name = "car-luggage-rack"  # TODO: Pick an example from assets/example_objects
    work_dir = f"assets/example_objects/{example_name}"
    assets = prepare_assets(work_dir)

    # Run inpainting pipeline
    outputs, new_slat = pipeline.run_inpaint(
        assets,
        seed=42,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 3},
    )
    
    # Post-process mesh and save result
    mesh_out = outputs["mesh"][0]
    vertices = mesh_out.vertices.cpu().numpy()
    faces = mesh_out.faces.cpu().numpy()
    vertices, faces = postprocessing_utils.postprocess_mesh(
        vertices, faces, simplify_ratio=0.9
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    rotate_mesh_x_neg90(mesh)
    mesh.export(f"{work_dir}/results/{assets.save_name}.ply")
    pred_slat = {'feats': new_slat.feats.cpu().numpy(). astype(np.float32),
                'coords': new_slat.coords[:, 1:].cpu().numpy().astype(np.uint8)}
    np.savez_compressed(os.path.join(f"{work_dir}/results", "pred_slat"), **pred_slat)

if __name__ == "__main__":
    main()
