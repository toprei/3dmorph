import os
import sys
import numpy as np
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
from tqdm import tqdm

os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir))

from trellis.modules import sparse as sp
from _3DMorph.slat_encoder.slat_decoder import SLATDecoder

class SLATInterpolator:
    @staticmethod
    def map_pos(sub_idx, all_idx):
        s, ord = torch.sort(all_idx)
        return ord[torch.searchsorted(s, sub_idx)]

    @staticmethod
    def cumcat(chunks, k, device):
        return torch.cat(chunks[:k]) if (len(chunks) and k > 0) else torch.empty(0, dtype=torch.long, device=device)

    @staticmethod
    def build_activation_order(nn_dists, diff_idx):
        return diff_idx[torch.argsort(nn_dists)]

    @staticmethod
    def make_step_slices(ordered_idx, num_steps):
        n = ordered_idx.numel()
        if n == 0:
            return [ordered_idx] * num_steps
        num_steps = min(max(1, num_steps), n)
        return list(torch.chunk(ordered_idx, num_steps))

    @staticmethod
    def create_output_folder(save_dir):
        local_tz = ZoneInfo('Europe/Berlin')
        time = datetime.now().astimezone(local_tz).strftime("%d-%b_%H:%M")
        out = os.path.join(save_dir, time)
        os.makedirs(out, exist_ok=True)
        return out

    @staticmethod
    def get_intersection(a, b):
        mask = (a[:, None, :] == b[None, :, :]).all(dim=2)
        return *torch.nonzero(mask, as_tuple=True),

    @staticmethod
    def get_difference(a, b):
        mask = (a[:, None, :] == b[None, :, :]).all(dim=2)
        in_a, in_b = mask.any(dim=1), mask.any(dim=0)
        return (~in_a).nonzero(as_tuple=True)[0], (~in_b).nonzero(as_tuple=True)[0]

    @staticmethod
    def nearest_neighbors(query, ref):
        if query.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=ref.device), torch.empty(0, device=ref.device)
        d = torch.cdist(query.float(), ref.float(), p=2)
        return d.min(dim=1)[1], d.min(dim=1)[0]

    @staticmethod
    def load_slat(p):
        x = np.load(p, allow_pickle=True)
        return torch.from_numpy(x['feats']).float(), torch.from_numpy(x['coords']).int()

    @staticmethod
    def run(
        start_slat_path: str,
        end_slat_path: str,
        total_steps: int = 20,
        save_root: str | None = None
    ):
        '''
        Interpolates between two SLAT representations and decodes the intermediate steps into meshes.
        
        Args:
            - end_slat_path (str): Path to the starting SLAT .npz file.
            - start_slat_path (str): Path to the ending SLAT .npz file.
            - total_steps (int): Number of interpolation steps.
            - save_root (str | None): Directory to save the output meshes. If None, a timestamped folder will be created.
            
        Returns:
            - str: Path to the directory containing the saved meshes.
            '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pred_feats, pred_coords = SLATInterpolator.load_slat(end_slat_path)
        gt_feats, gt_coords = SLATInterpolator.load_slat(start_slat_path)

        pred_feats, pred_coords = pred_feats.to(device), pred_coords.to(device)
        gt_feats,   gt_coords   = gt_feats.to(device),   gt_coords.to(device)

        inter_p, inter_g = SLATInterpolator.get_intersection(pred_coords, gt_coords)
        diff_p, diff_g   = SLATInterpolator.get_difference(pred_coords, gt_coords)

        nn_idx_p2g, nn_d_p2g = SLATInterpolator.nearest_neighbors(pred_coords[diff_p], gt_coords)
        nn_idx_g2p, nn_d_g2p = SLATInterpolator.nearest_neighbors(gt_coords[diff_g], pred_coords)

        act_slices = SLATInterpolator.make_step_slices(SLATInterpolator.build_activation_order(nn_d_p2g, diff_p), total_steps)
        rm_slices  = SLATInterpolator.make_step_slices(SLATInterpolator.build_activation_order(nn_d_g2p, diff_g), total_steps)

        if save_root is None:
            save_root = os.path.join(os.path.dirname(end_slat_path), "interpolations")
        save_dir = SLATInterpolator.create_output_folder(save_root)

        torch.cuda.empty_cache()

        decoder = SLATDecoder()

        pbar = tqdm(range(0, total_steps + 1), desc="Interpolating steps", unit="step")
        for step_id in pbar:
            a = step_id / total_steps

            act = SLATInterpolator.cumcat(act_slices, step_id, device=diff_p.device)
            rm  = SLATInterpolator.cumcat(rm_slices,  step_id, device=diff_g.device)

            keep_gt_mask = torch.ones(diff_g.size(0), dtype=torch.bool, device=diff_g.device)
            if rm.numel() > 0:
                keep_gt_mask.index_fill_(0, SLATInterpolator.map_pos(rm, diff_g), False)
            keep_g = diff_g[keep_gt_mask]

            nn_for_act  = nn_idx_p2g[SLATInterpolator.map_pos(act,    diff_p)] if act.numel()    > 0 else act
            nn_for_keep = nn_idx_g2p[SLATInterpolator.map_pos(keep_g, diff_g)] if keep_g.numel() > 0 else keep_g

            coords_step = torch.cat([gt_coords[inter_g], pred_coords[act], gt_coords[keep_g]], dim=0)

            feats_inter = (1 - a) * gt_feats[inter_g]    + a * pred_feats[inter_p]
            feats_act   = (1 - a) * gt_feats[nn_for_act] + a * pred_feats[act]
            feats_keep  = (1 - a) * gt_feats[keep_g]     + a * pred_feats[nn_for_keep]
            feats_step  = torch.cat([feats_inter, feats_act, feats_keep], dim=0)

            slat = sp.SparseTensor(
                feats=feats_step,
                coords=torch.cat([torch.zeros(feats_step.shape[0], 1, device=device, dtype=torch.int32),
                                  coords_step.int()], dim=1),
            ).to(device)

            m = decoder.decode(slat)
            out_path = os.path.join(save_dir, f"mesh_step_{step_id:02d}.obj")
            m.export(out_path)
            torch.cuda.empty_cache()
            print(f"{step_id}-> {out_path}")

        return save_dir


if __name__ == "__main__":
    example_name = "car-luggage-rack"  # TODO: Pick an example from assets/example_objects
    root_dir = Path(__file__).parents[2]
    start_slat_path = root_dir / "assets" / "example_objects" / example_name / "features" / "unmodified_slat.npz"
    end_slat_path = ""  # TODO: Enter a path to a generated SLAT
    save_root = ""  # TODO: Enter a path where the intermediate SLATs are going to be saved

    SLATInterpolator.run(
        end_slat_path=end_slat_path,
        start_slat_path=start_slat_path,
        total_steps=20,
        save_root=save_root
    )
