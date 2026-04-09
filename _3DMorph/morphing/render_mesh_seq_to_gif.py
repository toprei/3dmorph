import os
import re
import sys
from tqdm import tqdm
from PIL import Image
from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir))
from _3DMorph.renderer.render_simple import BlenderRenderer


class MeshSeq2GIF():
    @staticmethod
    def render(input_mesh_path, ref_obj_path):
        renderer = BlenderRenderer(resolution=512)
        view_args = {   
                    'offset': (-45, 20),
                    'step': 3,
                    'direction': 'left',
                    'set_fov': 35
                    }
        save_dir=os.path.dirname(input_mesh_path)
        images = renderer.render(input_mesh_path, ref_obj_path=ref_obj_path, n_views=1, mode='linear', lin_view_args=view_args, geo_mode=True)

        return images[0]
    
    @staticmethod
    def run(input_dir, output_gif_path):
        '''
        Renders a sequence of 3D mesh files in a directory and compiles them into a GIF.
        Args:
            - input_dir (str): Directory containing the 3D mesh files (.obj).
            - output_gif_path (str): Path to save the output GIF file.
        '''
        ply_files = [file for file in os.listdir(input_dir) if file.endswith(".obj")]
        max_file = max(ply_files, key=lambda f: int(re.search(r'\d+', f).group()))
        ref_obj_path = os.path.join(input_dir, max_file)

        image_list = []
        pbar = tqdm(sorted(ply_files), desc="Rendering frames")
        for file in pbar:
            input_mesh_path = os.path.join(input_dir, file)
            image = MeshSeq2GIF.render(input_mesh_path, ref_obj_path)
            image_list.append(image)

        image_list = image_list + image_list[-2:0:-1]
        durations = [100] * len(image_list)
        durations[0] = 500
        durations[len(image_list)//2] = 500
        image_list[0].save(
            output_gif_path,
            save_all=True,
            append_images=image_list[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=True
        )


class MeshSeq2GIF360():
    @staticmethod
    def render(input_mesh_path, ref_obj_path, azimuth_deg):
        renderer = BlenderRenderer(resolution=512)
        view_args = {
            'offset': (azimuth_deg, 20),
            'step': 3,
            'direction': 'left',
            'set_fov': 35
        }
        images = renderer.render(
            input_mesh_path,
            ref_obj_path=ref_obj_path,
            n_views=1,
            mode='linear',
            lin_view_args=view_args
        )
        return images[0]
    
    @staticmethod
    def run(input_dir, output_gif_path, start_az=60, total_rotation=30, direction='left'):
        '''
        Renders a sequence of 3D mesh files in a directory with a 360-degree rotation and compiles them into a GIF.
        Args:
            - input_dir (str): Directory containing the 3D mesh files (.obj).
            - output_gif_path (str): Path to save the output GIF file.
            - start_az (float): Starting azimuth angle in degrees.
            - total_rotation (float): Total rotation angle in degrees.
            - direction (str): Direction of rotation ('left' or 'right').
        '''
        ply_files = [f for f in os.listdir(input_dir) if f.endswith(".obj")]
        def num_key(f):
            m = re.search(r'\d+', f)
            return int(m.group()) if m else -1

        ply_files_sorted = sorted(ply_files, key=num_key)

        max_file = max(ply_files_sorted, key=num_key)
        ref_obj_path = os.path.join(input_dir, max_file)

        pal_files = ply_files_sorted + ply_files_sorted[-2:0:-1]

        total_frames = len(pal_files)

        image_list = []
        pbar = tqdm(enumerate(pal_files), desc="Rendering frames")
        for i, fname in pbar:
            if direction == 'left':
                az = start_az - (total_rotation * i) / max(1, total_frames - 1)
            elif direction == 'right':
                az = start_az + (total_rotation * i) / max(1, total_frames - 1)
            input_mesh_path = os.path.join(input_dir, fname)
            image = MeshSeq2GIF360.render(input_mesh_path, ref_obj_path, az)
            image_list.append(image)

        image_list = image_list + image_list[-2:0:-1]
        durations = [100] * len(image_list)
        durations[0] = 500
        for k in [1, 3]:
            durations[k * len(image_list) // 4] = 500
        image_list[0].save(
            output_gif_path,
            save_all=True,
            append_images=image_list[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False
        )


if __name__ == "__main__":
    input_dir = ""  # TODO: Replace with save_root (+ timestamp) from slat_interpolate_and_decoder.py
    output_gif_path = os.path.join(input_dir, "output.gif")
    MeshSeq2GIF.run(input_dir, output_gif_path)
    print(f"GIF saved to {output_gif_path}")
