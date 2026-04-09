import os
import sys
import json
import shutil
import hashlib
import subprocess
import numpy as np
from PIL import Image

from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir))

from _3DMorph.utils.dataset_utils import sphere_hammersley_sequence

class BlenderRenderer:
    def __init__(self, resolution=518):
        self.blender_path = '/home/ubuntu/blender/blender-3.0.1-linux-x64/blender'  # TODO: Replace with your blender path
        self.blender_script = os.path.join(os.path.dirname(__file__), 'blender_script', 'render_simple.py')
        self.resolution = resolution

    def stable_hash(self, s: str, mod: int = 10**8):
        return int(hashlib.sha1(s.encode()).hexdigest(), 16) % mod

    def safe_remove(self, path):
        if os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)

    def interpolate_views_with_params(self, n_views, radius, fov, **view_args):
        offset = view_args.get('offset', (0, 0))
        step = view_args.get('step', 30) / 180 * np.pi
        direction = view_args.get('direction', 'clockwise')
        yaw_values = view_args.get('yaw_values', None)
        fov = [view_args.get('set_fov', fov) / 180 * np.pi] * n_views

        if yaw_values is not None:
            assert len(yaw_values) == n_views, "Number of yaw_values must match n_views"

        else:  
            start_yaw = offset[0] / 180 * np.pi
            start_pitch = offset[1] / 180 * np.pi
            yaw_values = [start_yaw + i * step for i in range(n_views)] if direction == 'right' else \
                         [start_yaw - i * step for i in range(n_views)]

        pitch_values = [start_pitch] * n_views

        views = [
            {'yaw': y, 'pitch': p, 'radius': r, 'fov': f}
            for y, p, r, f in zip(yaw_values, pitch_values, radius, fov)
        ]
        return views


    def generate_views(self, n_views, mode, view_args):
        yaws = []
        pitchs = []
        radius = [2] * n_views
        fov = 35
        if mode == 'slat':
            fov = [fov] * n_views
            fov = [f / 180 * np.pi for f in fov]  # Convert degrees to radians
            offset = (0.1, 0.1)
            for i in range(n_views):
                y, p = sphere_hammersley_sequence(i, n_views, offset)
                yaws.append(y)
                pitchs.append(p)
            views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
        else:
            views = self.interpolate_views_with_params(n_views, radius, fov, **view_args)  

        return views

    def render(self, obj_path, ref_obj_path=None, save_dir=None, n_views=100, mode='slat', lin_view_args=None, save_transforms=False, geo_mode=False):
        '''
        Renders an object using Blender with specified views.

        Args:
            - obj_path (str): Path to the object file to be rendered.
            - ref_obj_path (str or None): Path to the reference object file. If None, no size reference is used.(size reference also see in feature extractor, voxel_utils...)
            - save_dir (str or None): Directory to save rendered images. If None, images are not saved.
            - n_views (int): Number of views to render.
            - mode (str): Mode of rendering, either 'slat' or 'lin'. This decides the view generation method, for normal use choose 'lin'.
            - lin_view_args (dict): Additional arguments for linear view generation. Only used if mode is 'lin'. Example: {'offset': (45, 30), 'step': 30, 'direction': 'right', 'set_fov': 35}, see interpolate_views_with_params for details.
            - save_transforms (bool): Whether to save camera transformation matrices.
            - geo_mode (bool): Whether to use geometric mode for rendering. (Faster but lower quality)

        Returns:
            - images (list of PIL.Image): List of rendered images.

        Caution:
            - tmp_output_folder can be different on different systems, here we use /dev/shm for faster I/O. Make sure the path exists and has enough space.
        '''
        views = self.generate_views(n_views, mode, lin_view_args)
        args = [
            self.blender_path,
            '-b',
            '-P', self.blender_script,
            '--',
            '--views', json.dumps(views),
            '--object', os.path.expanduser(obj_path),
            '--ref_object', os.path.expanduser(ref_obj_path) if ref_obj_path else 'none',
            '--resolution', str(self.resolution),
            '--mode', str(mode),
            '--geo_mode', str(geo_mode).lower(),    
            ]
        print_args = args.copy()
        print_args[print_args.index('--views') + 1] = f"'{json.dumps(views)}'"

        try:
            e = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! Failed to render: {obj_path}")
            print(f"Return code: {e.returncode}")
            print("---- STDOUT ----")
            print(e.stdout)
            print("---- STDERR ----")
            print(e.stderr)

        tmp_output_folder = f"/dev/shm/blender_render_{self.stable_hash(str(obj_path))}"
        image_paths = [os.path.join(tmp_output_folder, f'render_{i:03d}.png') for i in range(n_views)]
        images = [Image.open(img_path) for img_path in image_paths]
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for file_name in os.listdir(tmp_output_folder):
                if file_name.endswith('.png'):
                    shutil.copy(os.path.join(tmp_output_folder, file_name), os.path.join(save_dir, file_name))
                if save_transforms:
                    shutil.copy(os.path.join(tmp_output_folder, 'transforms.json'), os.path.join(save_dir, 'transforms.json'))
        
        self.safe_remove(tmp_output_folder)
        return images



if __name__ == "__main__":
    import time
    from pathlib import Path

    example_name = "car-luggage-rack"  # TODO: Pick an example from assets/example_objects
    root_dir = Path(__file__).parents[2]

    obj_path = root_dir / "assets" / "example_objects" / example_name / "unmodified.obj"
    ref_obj_path = None

    save_dir = obj_path.parent / 'explore_inpaint'
    renderer = BlenderRenderer(resolution=1024)
    view_args = {'offset': (-75, 10), 'step': 30, 'direction': 'right', 'set_fov': 25}

    start_time = time.time()
    images = renderer.render(
        obj_path, 
        ref_obj_path, 
        save_dir=save_dir, 
        n_views=1, 
        mode='linear', 
        lin_view_args=view_args, 
        save_transforms=True, 
        geo_mode=False
    )
    end_time = time.time()
    print(f"Rendering completed in {end_time - start_time:.2f} seconds.")