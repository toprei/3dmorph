import os
import sys
import trimesh
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import torch

os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir))

from trellis.modules import sparse as sp
from trellis.models import from_pretrained
from trellis.utils.postprocessing_utils import postprocess_mesh

class SLATDecoder:
    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f'PATH: {str(root_dir / "pretrained_weights" / "TRELLIS-image-large" / "ckpts" / "slat_dec_mesh_swin8_B_64l8m256c_fp16")}')
        self.mesh_decoder = from_pretrained(
            str(root_dir / "pretrained_weights" / "TRELLIS-image-large" / "ckpts" / "slat_dec_mesh_swin8_B_64l8m256c_fp16")
        ).eval().cuda()
        self.save_dir = None


    def create_output_folder(self, save_dir):
        '''
        Creates a unique timestamped output folder for saving results as usual in this work.
        Example: If save_dir is "/path/to/output", the created folder might be "/path/to/output/25-Jun_14:30".
        '''
        local_tz = ZoneInfo('Europe/Berlin')
        local_time = datetime.now().astimezone(local_tz)
        time = local_time.strftime("%d-%b_%H:%M")
        save_dir = os.path.join(save_dir, time)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        return save_dir
    
    def decode(self, slat):
        """
        Decodes features from a given feature file using the SLAT decoder.

        Args:
            - slat (str or sp.SparseTensor): Path to the input .npz file containing feature data or sp.SparseTensor.
        Returns:
            - mesh (trimesh.Trimesh): The decoded 3D mesh.
        """
        with torch.no_grad():
            if isinstance(slat, str):
                mesh_name = os.path.dirname(slat).split('/')[-2]
                slat_feats = np.load(slat)
                feats = torch.from_numpy(slat_feats['feats']).float()
                coords = torch.cat([
                    torch.zeros(slat_feats['feats'].shape[0], 1, dtype=torch.int32),
                    torch.from_numpy(slat_feats['coords']).int(),
                ], dim=1)
                slat = sp.SparseTensor(
                    feats=feats,
                    coords=coords,
                ).cuda()
            elif isinstance(slat, sp.SparseTensor):
                mesh_name = "decoded_mesh"
            else:
                raise ValueError("Input must be a path to a .npz file or a SparseTensor.")
            
            outputs = self.mesh_decoder(slat)

        vertices = outputs[0].vertices.cpu().numpy()
        faces = outputs[0].faces.cpu().numpy()
        
        vertices, faces = postprocess_mesh(vertices, faces, simplify_ratio=0.8)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.radians(-90),
            direction=[1, 0, 0],
            point=[0, 0, 0]
        )
        mesh.apply_transform(rotation_matrix)
        torch.cuda.empty_cache()

        if self.save_dir is not None:
            mesh.export(self.save_dir + "/" + mesh_name + ".ply")
            print(f"Mesh saved to {self.save_dir}/{mesh_name}.ply")
            return mesh
        else:
            return mesh
    
if __name__ == "__main__":
    slat_decoder = SLATDecoder()
    example_name = "car-heck-spoiler"  # TODO: Pick an example from assets/example_objects
    root_dir = Path(__file__).parents[2]  # TODO: Remove

    output_dir = root_dir / "assets" / "example_objects" / example_name
    slat_path = output_dir / "features" / "unmodified_slat.npz"
    slat_decoder.create_output_folder(str(output_dir))
    print("Starting to decode SLAT...")
    slat_decoder.decode(str(slat_path))
    print("Decoding completed.")
