import os
import torch
import numpy as np
import sys
from pathlib import Path

os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir))

from trellis.modules import sparse as sp
from trellis.models import from_pretrained
from _3DMorph.slat_encoder.feature_extractor import FeatureExtractor

class LatentEncoder:
    def __init__(self):
        self.encoder = from_pretrained(
            str(root_dir / "pretrained_weights" / "TRELLIS-image-large" / "ckpts" / "slat_enc_swin8_B_64l8_fp16")
        ).eval().cuda()

    def encode_features(self, feat_path):
        """
        Extracts features from a given feature file using the SLAT encoder.

        Args:
            feat_path (str): Path to the input .npz file containing feature data.

        Returns:
            dict: A dictionary containing:
                - 'feats' (np.ndarray): Extracted feature tensor as a NumPy array.
                - 'coords' (np.ndarray): Corresponding coordinates as a NumPy array.
        """
        with torch.no_grad():
            feats = np.load(feat_path)
            
            sparse_tensor = sp.SparseTensor(
                feats=torch.from_numpy(feats['patchtokens']).float(),
                coords=torch.cat([
                    torch.zeros(feats['patchtokens'].shape[0], 1).int(),
                    torch.from_numpy(feats['indices']).int(),
                ], dim=1),
            ).cuda()
            
            latent = self.encoder(sparse_tensor, sample_posterior=False)
            
            assert torch.isfinite(latent.feats).all(), "Non-finite latent"
            
            return {
                'feats': latent.feats.cpu().numpy().astype(np.float32),
                'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8)
            }

    def run_slat_encoder(self, feat_path):
        """
        Runs the SLAT encoder on a given feature file and saves the output.

        Args:
            - feat_path (str): Path to the input .npz file containing feature data.
        Returns:
            - str: Path to the saved .npz file containing encoded features.
        """
        features = self.encode_features(feat_path)

        obj_name = os.path.splitext(os.path.basename(feat_path))[0]
        save_path = os.path.join(os.path.dirname(feat_path), f'{obj_name}_slat.npz')
        np.savez_compressed(save_path, **features)
        
        print(f"Features saved to {save_path}")
        print("Features shape:", features['feats'].shape)
        print("Coordinates shape:", features['coords'].shape)

        torch.cuda.empty_cache()
        return save_path

if __name__ == '__main__':
    # TODO: Decrease batch_size to 50 if you have less than 20GB VRAM
    extractor = FeatureExtractor(batch_size=150, n_views=150)
    slat_encoder = LatentEncoder()

    example_name = "car-luggage-rack"  # TODO: Pick an example from assets/example_objects
    ref_mesh_path = None
    root_dir = Path(__file__).parents[2]  # TODO: Remove
    input_mesh_path = root_dir / "assets" / "example_objects" / example_name / "unmodified.obj"

    feat_path = extractor.run_extractor(str(input_mesh_path), ref_mesh_path, force_render=True)
    slat_path = slat_encoder.run_slat_encoder(feat_path)
