from typing import *
import torch
import torch.nn.functional as F
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from _3DMorph.utils.voxel_utils import VoxelUtils
from _3DMorph.utils.work_assets import WorkAssets
from _3DMorph.utils.mask_predictor import MaskPredictor
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline



class _3DMorphInpaint(TrellisImageTo3DPipeline):

    @staticmethod
    def from_pretrained(path: str) -> "_3DMorphInpaint":
        """
        Load a pretrained model.

        Args:
            - path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(_3DMorphInpaint, _3DMorphInpaint).from_pretrained(path)
        new_pipeline = _3DMorphInpaint()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    @torch.no_grad()
    def sparse_structure_inpaint(
        self,
        img_cond: dict,
        original_binary_ss_map: torch.Tensor,
        mask: torch.Tensor = None,
        num_samples: int = 1,
        seed: int = None,
        mode: str = 'hybrid',
        sparse_structure_sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            - img_cond (dict): The conditioned image latent.
            - original_binary_ss_map (torch.Tensor): The original binary sparse structure map.
            - mask (torch.Tensor): The mask for inpainting in sparse structure space, expected to be in 16^3 space.
            - num_samples (int): The number of samples to generate.
            - seed (int): The random seed for sampling.
            - sparse_structure_sampler_params (dict): Additional parameters for the sampler.
            - mode (str): The inpainting mode, can be 'hard' or 'repaint' or 'hybrid'. If 'hybrid', use 'hard' for sparse structure and 'repaint' for slat.
        """
        if seed is not None:
            torch.manual_seed(seed)
        flow_model = self.models['sparse_structure_flow_model']
        sparse_structure_sampler_params = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}
        
        encoder = self.models['sparse_structure_encoder']
        original_binary_ss_map = (original_binary_ss_map > 0).float()
        encoded_latent_ss = encoder(original_binary_ss_map)

        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        z_s = self.sparse_structure_sampler.inpaint(
            flow_model,
            encoded_latent_ss,
            noise,
            mask,
            **img_cond,
            blend_type = 'hard' if mode!='repaint' else 'repaint',
            **sparse_structure_sampler_params,
            verbose=True
        ).samples
        
        decoder = self.models['sparse_structure_decoder']
        binary_sparse_structure_map = decoder(z_s)
        coords = torch.argwhere(binary_sparse_structure_map>0)[:, [0, 2, 3, 4]].int()

        return coords

    @torch.no_grad()
    def second_stage_slat_inpaint(
        self,
        img_cond: dict,
        new_coords: torch.Tensor,
        slat: sp.SparseTensor,
        sparse_grid_mask: torch.Tensor, # shape [, 4]
        seed: int = None,
        mode: str = 'hybrid',
        slat_sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            - img_cond (dict): The conditioned image latent.
            - new_coords (torch.Tensor): The coordinates of the sparse structure, expected to be sparse coords.
            - slat (sp.SparseTensor): The original latent to be inpainted.
            - sparse_grid_mask (torch.Tensor): The mask for the sparse grid.
            - seed (int): The random seed for sampling.
            - slat_sampler_params (dict): Additional parameters for the sampler.
            - mode (str): The inpainting mode, can be 'hard' or 'repaint' or 'hybrid'. If 'hybrid', use 'repaint' for slat and 'hard' for sparse structure.

        Returns:
            - sp.SparseTensor: The inpainted structured latent.
        """
        if seed is not None:
            torch.manual_seed(seed)
        flow_model = self.models['slat_flow_model']

        original_coords = slat.coords
        original_feats = slat.feats
        
        def get_intersection(a: torch.Tensor, b: torch.Tensor):
            def _norm(x: torch.Tensor):
                if x.ndim != 2:
                    x = x.view(-1, x.size(-1))
                if x.size(1) == 4:
                    x = x[:, 1:]
                elif x.size(1) != 3:
                    raise AssertionError("coords must be [N,3] or [N,4]")
                if x.dtype not in (torch.int32, torch.int64):
                    x = x.round().long()
                else:
                    x = x.long()
                return x.contiguous()

            a = _norm(a)
            b = _norm(b)

            if a.numel() == 0 or b.numel() == 0:
                dev = a.device if a.is_cuda else b.device
                return (torch.empty(0, dtype=torch.long, device=dev),
                        torch.empty(0, dtype=torch.long, device=dev))

            if a.device != b.device:
                b = b.to(a.device)

            mins = torch.min(torch.stack([a.min(0).values, b.min(0).values]), dim=0).values
            a -= mins
            b -= mins
            maxs = torch.max(torch.stack([a.max(0).values, b.max(0).values]), dim=0).values
            dims = maxs + 1

            if torch.any(dims > (1 << 21)):
                shift1, shift2 = 21, 42
                a_key = (a[:, 0] << shift2) ^ (a[:, 1] << shift1) ^ a[:, 2]
                b_key = (b[:, 0] << shift2) ^ (b[:, 1] << shift1) ^ b[:, 2]
            else:
                Dy = int(dims[1].item())
                Dz = int(dims[2].item())
                a_key = (a[:, 0] * Dy + a[:, 1]) * Dz + a[:, 2]
                b_key = (b[:, 0] * Dy + b[:, 1]) * Dz + b[:, 2]

            a_key_sorted, a_ord = torch.sort(a_key)
            b_key_sorted, b_ord = torch.sort(b_key)

            if a_key_sorted.numel() == 0 or b_key_sorted.numel() == 0:
                dev = a_key_sorted.device
                return (torch.empty(0, dtype=torch.long, device=dev),
                        torch.empty(0, dtype=torch.long, device=dev))

            pos = torch.searchsorted(a_key_sorted, b_key_sorted)

            inb = pos < a_key_sorted.numel()
            if a_key_sorted.numel() == 0:
                valid = torch.zeros_like(inb, dtype=torch.bool)
            else:
                pos_safe = pos.clamp_max(max(a_key_sorted.numel() - 1, 0))
                eq = (a_key_sorted[pos_safe] == b_key_sorted)
                valid = inb & eq

            if not valid.any():
                dev = a_key_sorted.device
                return (torch.empty(0, dtype=torch.long, device=dev),
                        torch.empty(0, dtype=torch.long, device=dev))

            b_idx_in_b = b_ord[valid]
            a_idx_in_a = a_ord[pos[valid]]
            return a_idx_in_a, b_idx_in_b

        new_feats = torch.zeros(new_coords.shape[0], original_feats.shape[1], dtype=original_feats.dtype, device=self.device)
        new_org_coords_matched_indices_in_new, new_org_coords_matched_indices_in_org = get_intersection(new_coords, original_coords)

        new_feats[new_org_coords_matched_indices_in_new] = original_feats[new_org_coords_matched_indices_in_org]

        mask_coords = sparse_grid_mask[:, 1:]

        new_mask_coords_matched_indices_in_new, _ = get_intersection(new_coords[:, 1:], mask_coords)
        slat_feat_binary_mask = torch.zeros(new_coords.shape[0], dtype=torch.int, device=self.device).unsqueeze(1)
        slat_feat_binary_mask[new_mask_coords_matched_indices_in_new] = 1

        noise = sp.SparseTensor(
            feats=torch.randn_like(new_feats).to(self.device),
            coords=new_coords,
        )

        updated_slat = sp.SparseTensor(
            feats=new_feats,
            coords=new_coords,
        )

        slat_sampler_params = {**self.slat_sampler_params, **slat_sampler_params}

        new_slat = self.slat_sampler.inpaint(
            flow_model,
            updated_slat,
            noise,
            slat_feat_binary_mask,
            **img_cond,
            blend_type = 'repaint' if mode!='hard' else 'hard',
            **slat_sampler_params,
            verbose=True
        ).samples


        std = torch.tensor(self.slat_normalization['std'])[None].to(new_slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(new_slat.device)

        new_feats = new_slat.feats.clone()
        new_feats[new_mask_coords_matched_indices_in_new] = new_feats[new_mask_coords_matched_indices_in_new] * std + mean

        new_slat = sp.SparseTensor(
            feats=new_feats,
            coords=new_slat.coords
        )   

        return new_slat
    
    @torch.no_grad()
    def run_inpaint(
        self,
        assets: WorkAssets,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            - assets (WorkAssets): The work assets containing the input data.
            - num_samples (int): The number of samples to generate.
            - seed (int): The random seed for sampling.
            - sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            - slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            - formats (List[str]): The output formats. Can be any of 'mesh', 'gaussian', 'radiance_field'.
        Returns:
            - dict: A dictionary containing the decoded outputs in the specified formats.
        """
        modified_image = Image.open(assets.mod_img_path)
        cond = self.get_cond([modified_image])
        mask_pred = MaskPredictor()
        ss_mask, _ = mask_pred.predict_mask(
        assets.mesh_path,
        assets.unmod_img_path,
        assets.mod_img_path,
        assets.transforms_json,
        resolution=modified_image.size[0],
    )
        original_binary_ss_map = VoxelUtils.sparse_to_dense_grid(assets.slat.coords)
        torch.manual_seed(seed)
        coords = self.sparse_structure_inpaint(
            cond,
            original_binary_ss_map,
            ss_mask,
            num_samples,
            mode='hybrid',
            sparse_structure_sampler_params=sparse_structure_sampler_params,
        )
        
        interpolated_mask = F.interpolate(ss_mask, size=(64, 64, 64), mode='nearest')
        sparse_grid_mask = VoxelUtils.dense_to_sparse_grid(interpolated_mask)

        new_slat = self.second_stage_slat_inpaint(
            cond, 
            coords, 
            assets.slat,
            sparse_grid_mask,
            mode='hybrid',
            slat_sampler_params=slat_sampler_params,
        )
        torch.cuda.empty_cache()
        return self.decode_slat(new_slat, formats=formats), new_slat