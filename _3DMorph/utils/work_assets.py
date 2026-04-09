from dataclasses import dataclass
from trellis.modules import sparse as sp

@dataclass
class WorkAssets:
    mesh_path: str
    unmod_img_path: str
    mod_img_path: str
    transforms_json: str
    slat: sp.SparseTensor
    save_name: str
