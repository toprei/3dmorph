import os
import sys
from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir))

from _3DMorph.renderer.render_simple import BlenderRenderer

class Mesh2GIF():
    @staticmethod
    def render(input_mesh_path, n_views, start_az):
        renderer = BlenderRenderer(resolution=768)
        view_args = {
            'offset': (start_az, 30),
            'step': 360 / n_views,
            'direction': 'right',
            'set_fov': 35
        }
        return renderer.render(input_mesh_path, n_views=n_views, mode='linear', lin_view_args=view_args, geo_mode=False)
    
    @staticmethod
    def run(input_mesh_path, output_gif_path, total_frames=20):
        '''
        Renders a 3D mesh with a 360-degree rotation and compiles them into a GIF.
        Args:
            - input_mesh_path (str): Path to the 3D mesh file.
            - output_gif_path (str): Path to save the output GIF file.
        '''
        start_az = 60
        frames = Mesh2GIF.render(input_mesh_path, total_frames, start_az)
        os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
        image_list = frames[:-1]
        durations = [100] * len(image_list)
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
    from pathlib import Path
    example_name = "car-luggage-rack"  # TODO: Pick an example from assets/example_objects
    root_dir = Path(__file__).parents[2]
    input_mesh_path = root_dir / "assets" / "example_objects" / example_name / "unmodified.obj"
    output_gif_path = input_mesh_path.parent / 'colored_diff.gif'
    total_frames = 60
    Mesh2GIF.run(input_mesh_path, output_gif_path, total_frames)
    print(f"Saved GIF to {output_gif_path}")