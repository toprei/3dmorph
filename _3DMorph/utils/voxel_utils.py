import numpy as np
import torch
import trimesh
import open3d as o3d
from joblib import Parallel, delayed
import os
import utils3d
import torch.nn.functional as F

class VoxelUtils:
    def __init__(self):
        pass

    @staticmethod
    def visualize_sparse_voxels(save_dir, input_mesh_name, coords, ret_mode=['obj', 'ply'], voxel_size=1, normalize=True) -> str:
        '''
        Visualize sparse voxels from coordinates.

        Args:
            - save_dir (str): Directory to save the output mesh.
            - input_mesh_name (str): Base name for the output mesh file.
            - coords (torch.Tensor): Sparse voxel coordinates of shape (N, 3) or (N, 4).
            - ret_mode (list): List of formats to export the mesh. Supports ['obj', 'ply', 'stl'].
            - voxel_size (float): Size of each voxel.
            - normalize (bool): Whether to normalize the mesh vertices.

        Returns:
            - str: Path to the saved .ply file.
        '''
        coords_gpu = coords.cuda()
        if coords_gpu.shape[1] == 4:
            voxels = coords_gpu[:, 1:]
        else:
            voxels = coords_gpu

        cube_offsets = torch.tensor([
            [0, 0, 0],
            [voxel_size, 0, 0],
            [0, voxel_size, 0],
            [voxel_size, voxel_size, 0],
            [0, 0, voxel_size],
            [voxel_size, 0, voxel_size],
            [0, voxel_size, voxel_size],
            [voxel_size, voxel_size, voxel_size]
        ], device='cuda')

        voxel_vertices = voxels[:, None, :] + cube_offsets[None, :, :]  # Shape: [N, 8, 3]
        voxel_vertices = voxel_vertices.reshape(-1, 3)  # Flatten to [N*8, 3]

        unique_vertices, inverse_indices = torch.unique(voxel_vertices, dim=0, return_inverse=True)

        cube_faces = torch.tensor([
            [0, 1, 2], [1, 2, 3],  # Front face
            [4, 5, 6], [5, 6, 7],  # Back face
            [0, 1, 4], [1, 4, 5],  # Bottom face
            [2, 3, 6], [3, 6, 7],  # Top face
            [0, 2, 4], [2, 4, 6],  # Left face
            [1, 3, 5], [3, 5, 7]   # Right face
        ], device='cuda')

        voxel_faces = inverse_indices.reshape(-1, 8)[:, cube_faces].reshape(-1, 3)  # Shape: [N*12, 3]

        vertices = unique_vertices.cpu().numpy()
        faces = voxel_faces.cpu().numpy()

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        if normalize:
            mesh.vertices -= mesh.centroid
            max_extent = mesh.bounds[1] - mesh.bounds[0]
            max_extent_length = max(max_extent)
            mesh.vertices /= max_extent_length

        for mode in ret_mode:
            if mode == 'ply':
                mesh.export(save_dir + '/' + input_mesh_name + '_output_voxels.ply')
            elif mode == 'obj':
                mesh.export(save_dir + '/' + input_mesh_name + '_output_voxels.obj')
            elif mode == 'stl':
                mesh.export(save_dir + '/' + input_mesh_name + '_output_voxels.stl')

        return save_dir + '/' + input_mesh_name + '_output_voxels' + f".{ret_mode[0]}"

    @staticmethod
    def visualize_dense_voxels(save_dir, input_mesh_name, dense_voxel_grid, ret_mode=['obj', 'ply'], voxel_size=1, normalize=True, ret_trimesh=False, up=None) -> str:
        '''
        Visualize dense voxels from a dense voxel grid.

        Args:
            - save_dir (str or None): Directory to save the output mesh. If ret_trimesh is True, this can be None.
            - input_mesh_name (str or None): Base name for the output mesh file. If ret_trimesh is True, this can be None.
            - dense_voxel_grid (torch.Tensor): Dense voxel grid of shape (1, ..., D, H, W).
            - ret_mode (list): List of formats to export the mesh. Supports ['obj', 'ply', 'stl'].
            - voxel_size (float): Size of each voxel.
            - normalize (bool): Whether to normalize the mesh vertices.
            - ret_trimesh (bool): Whether to return the trimesh object instead of saving.
            - up (np.ndarray or None): Optional up vector for orientation, ONLY used in process_mesh().

        Returns:
            - str: Path to the saved mesh file if ret_trimesh is False.
            - trimesh.Trimesh: The generated mesh if ret_trimesh is True.
        '''
        assert len(dense_voxel_grid.shape) == 5

        dense_voxel_grid = dense_voxel_grid[0, 0].cuda()
        D, H, W = dense_voxel_grid.shape

        active_voxels = (dense_voxel_grid > 0).nonzero(as_tuple=False)
        active_voxels = active_voxels.float()

        cube_offsets = torch.tensor([
            [0, 0, 0],
            [voxel_size, 0, 0],
            [0, voxel_size, 0],
            [voxel_size, voxel_size, 0],
            [0, 0, voxel_size],
            [voxel_size, 0, voxel_size],
            [0, voxel_size, voxel_size],
            [voxel_size, voxel_size, voxel_size]
        ], device='cuda')

        voxel_vertices = active_voxels[:, None, :] + cube_offsets[None, :, :]  # Shape: [N, 8, 3]
        voxel_vertices = voxel_vertices.reshape(-1, 3)  # Flatten to [N*8, 3]

        # Remove duplicate vertices by using a hash map
        unique_vertices, inverse_indices = torch.unique(voxel_vertices, dim=0, return_inverse=True)

        cube_faces = torch.tensor([
            [0, 1, 2], [1, 2, 3],  # Front face
            [4, 5, 6], [5, 6, 7],  # Back face
            [0, 1, 4], [1, 4, 5],  # Bottom face
            [2, 3, 6], [3, 6, 7],  # Top face
            [0, 2, 4], [2, 4, 6],  # Left face
            [1, 3, 5], [3, 5, 7]   # Right face
        ], device='cuda')

        voxel_faces = inverse_indices.reshape(-1, 8)[:, cube_faces].reshape(-1, 3)  # Shape: [N*12, 3]

        vertices = unique_vertices.cpu().numpy()
        faces = voxel_faces.cpu().numpy()
        mesh = trimesh.Trimesh(vertices, faces=faces)
        
        if ret_trimesh:
            return mesh

        if normalize:
            mesh.vertices -= mesh.centroid
            max_extent = mesh.bounds[1] - mesh.bounds[0]
            max_extent_length = max(max_extent)
            mesh.vertices /= max_extent_length
        
        if up is not None:
            center = np.array([32, 32, 32])

            translation_to_origin = trimesh.transformations.translation_matrix(-center)
            mesh.apply_transform(translation_to_origin)

            y_axis = np.array([0, 1, 0])
            rotation_axis = np.cross(y_axis, up)
            rotation_angle = np.arccos(np.dot(y_axis, up))
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
            mesh.apply_transform(rotation_matrix)

            translation_back = trimesh.transformations.translation_matrix(center)
            mesh.apply_transform(translation_back)

        for mode in ret_mode:
            if mode == 'obj':
                mesh.export(save_dir + '/' + input_mesh_name + '_voxels.obj')
            elif mode == 'ply':
                mesh.export(save_dir + '/' + input_mesh_name + '_voxels.ply')
            elif mode == 'stl':
                mesh.export(save_dir + '/' + input_mesh_name + '_voxels.stl')

        return save_dir + '/' + input_mesh_name + '_voxels' + f".{ret_mode[0]}"


    @staticmethod
    def mesh_voxel_to_dense_voxel_grid(input_mesh_path, resolution=64) -> torch.Tensor:
        '''
        Convert a NORMALIZED PLY voxel in mesh representation (0 to 64*64*64) back to dense_voxel_grid.

        Args:
            - input_mesh_path (str): Path to the input PLY mesh file.
            - resolution (int): Resolution of the voxel grid (default: 64).

        Returns:
            - torch.Tensor: Dense voxel grid of shape (1, 1, resolution, resolution, resolution).
        '''
        mesh = o3d.io.read_triangle_mesh(input_mesh_path)
        vertices = np.asarray(mesh.vertices)

        vertices = np.clip(vertices, 0, resolution - 1e-6)

        voxel_indices = np.floor(vertices).astype(np.int32)

        dense_voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

        for voxel in voxel_indices:
            dense_voxel_grid[voxel[0], voxel[1], voxel[2]] = 1.0

        dense_voxel_grid = torch.tensor(dense_voxel_grid, dtype=torch.float32, device='cuda')
        dense_voxel_grid = dense_voxel_grid.unsqueeze(0).unsqueeze(0)

        return dense_voxel_grid

    
    @staticmethod
    def sparse_to_dense_grid(sparse_grid, resolution=64) -> torch.Tensor:
        '''
        Convert a sparse voxel grid to a dense voxel grid.
        Args:
            - sparse_grid (torch.Tensor): Sparse voxel grid of shape (N, 4) where N is the number of occupied voxels.
            - resolution (int): Resolution of the dense voxel grid (default: 64).
        Returns:
            - torch.Tensor: Dense voxel grid of shape (1, 1, resolution, resolution, resolution).
        '''
        dense_voxel_grid = torch.zeros((resolution, resolution, resolution), dtype=torch.float32, device=sparse_grid.device)
        if sparse_grid.shape[1] == 4:
            indices = sparse_grid[:, 1:].long()
        else:
            indices = sparse_grid.long()
            
        dense_voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
        dense_voxel_grid = dense_voxel_grid.unsqueeze(0).unsqueeze(0)

        return dense_voxel_grid
    
    @staticmethod
    def dense_to_sparse_grid(dense_voxel_grid: torch.Tensor) -> torch.Tensor:
        '''
        Convert a dense voxel grid to a sparse voxel grid.
        
        Args:
            - dense_voxel_grid (torch.Tensor): Dense voxel grid of shape (1, 1, resolution, resolution, resolution).
        
        Returns:
            - torch.Tensor: Sparse voxel grid of shape (N, 4) where N is the number of occupied voxels.   
        '''
        assert len(dense_voxel_grid.shape) == 5
        dense_voxel_grid = dense_voxel_grid[0, 0]
        indices = torch.nonzero(dense_voxel_grid >= 0.5)
        sparse_voxel_grid = torch.cat([torch.zeros(indices.shape[0], 1, device=dense_voxel_grid.device), indices.float()], dim=1)

        return sparse_voxel_grid

    @staticmethod
    def voxelize_mesh(mesh_path=None, o3d_mesh=None, resolution=64, max_extent=None, box_center=None) -> torch.Tensor:
        '''
        Voxelize a mesh into a dense voxel grid.

        Args:
            - mesh_path (str or None): Path to the input mesh file. If None, o3d_mesh must be provided.
            - o3d_mesh (o3d.geometry.TriangleMesh or None): An Open3D TriangleMesh object. If None, mesh_path must be provided.
            - resolution (int): Resolution of the voxel grid (default: 64).
            - max_extent (float or None): Maximum extent for normalization. If None, it will be computed from the mesh.
            - box_center (np.ndarray or None): Center of the bounding box for normalization. If None, it will be computed from the mesh.

        Returns:
            - torch.Tensor: Dense voxel grid of shape (1, 1, resolution, resolution, resolution).

        Caution:
            - Either mesh_path or o3d_mesh must be provided, but not both.
            - In preprocessing data, max_extent and box_center should be computed from the entire dataset to ensure consistency. (see also size_ref_mesh_path in feature_extractor.py and process_directory() which computes max_extent and box_center from the "biggest" mesh in the dataset)
        '''
        if (mesh_path is None and o3d_mesh is None) or (mesh_path is not None and o3d_mesh is not None):
            raise ValueError("Either mesh_path or o3d_mesh must be provided, but not both.")

        if mesh_path is not None:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
        else:
            mesh = o3d_mesh

        if max_extent is None:
            bounding_box = mesh.get_axis_aligned_bounding_box()
            min_bound = bounding_box.min_bound
            max_bound = bounding_box.max_bound
            max_extent = max(max_bound - min_bound)
            box_center = bounding_box.get_center()

        mesh.translate(-box_center, relative=True)
        mesh.scale(1.0 / max_extent, center=(0, 0, 0))

        vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)  

        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))


        dense_voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        for voxel in voxel_grid.get_voxels():
            grid_index = voxel.grid_index
            dense_voxel_grid[grid_index[0], grid_index[1], grid_index[2]] = 1.0
        
        dense_voxel_grid = torch.tensor(dense_voxel_grid, dtype=torch.float32, device='cuda')
        dense_voxel_grid = dense_voxel_grid.unsqueeze(0).unsqueeze(0)
        
        return dense_voxel_grid

    
    @staticmethod
    def get_max_extent(input_mesh) -> list:
        '''
        Get the maximum extent, bounding box center, and bounding box volume of a mesh.
        Args:
            - input_mesh (str or o3d.geometry.TriangleMesh): Path to the input mesh file or an Open3D TriangleMesh object.

        Returns:
            - list: [max_extent (float), box_center (np.ndarray), bb_box_volume (float)]'''
        if isinstance(input_mesh, str):
            mesh = o3d.io.read_triangle_mesh(input_mesh)
        else:
            mesh = input_mesh   
        bounding_box = mesh.get_axis_aligned_bounding_box()
        min_bound = bounding_box.min_bound
        max_bound = bounding_box.max_bound
        max_extent = max(max_bound - min_bound)
        box_center = bounding_box.get_center()
        bb_box_volume = np.prod(max_bound - min_bound)
        return max_extent, box_center, bb_box_volume
    
    @staticmethod
    def fill_voxel(voxel_grid: torch.Tensor) -> torch.Tensor:
        '''
        Fill the inside of a voxel grid.

        Args:
            - voxel_grid (torch.Tensor): Dense voxel grid of shape (1, 1, D, H, W).
            
        Returns:
            - torch.Tensor: Filled dense voxel grid of shape (1, 1, D, H, W).

        Caution:
            - This function assumes that the input voxel grid is binary (0s and 1s).
            - Watertightness: This method assumes that the input voxel grid represents a watertight shape. If the shape has holes or is not fully enclosed, the filling may not work as intended.
            '''
        filled = voxel_grid.clone()
        device = filled.device

        D, H, W = filled.shape[-3:]

        outside = torch.zeros_like(filled, dtype=torch.bool)

        outside[..., 0, :, :] = filled[..., 0, :, :] == 0
        outside[..., -1, :, :] = filled[..., -1, :, :] == 0
        outside[..., :, 0, :] = filled[..., :, 0, :] == 0
        outside[..., :, -1, :] = filled[..., :, -1, :] == 0
        outside[..., :, :, 0] = filled[..., :, :, 0] == 0
        outside[..., :, :, -1] = filled[..., :, :, -1] == 0

        kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=device)
        kernel[:, :, 1, 1, 1] = 0

        while True:
            expanded = F.conv3d(outside.float(), kernel, padding=1) > 0
            new_outside = (expanded & (filled == 0)) | outside

            if torch.equal(new_outside, outside):
                break
            outside = new_outside

        inside = (filled == 0) & (~outside)
        filled[inside] = 1

        return filled
    
    @staticmethod
    def merge_meshes(mesh_paths, output_path):
        '''
        Merge multiple meshes into a single mesh and save it.

        Args:
            - mesh_paths (list): List of paths to the input mesh files.
            - output_path (str): Path to save the merged mesh file.

        Returns:
            - str: Path to the saved merged mesh file.
        '''
        merged_mesh = o3d.geometry.TriangleMesh()

        for mesh_path in mesh_paths:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            merged_mesh += mesh

        o3d.io.write_triangle_mesh(output_path, merged_mesh)
        return output_path
    
    @staticmethod
    def rotate_and_normalize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        '''
        Rotate the mesh around the x-axis by -90 degrees and normalize it to fit within a unit cube centered at the origin.
        
        Args:
            - mesh (o3d.geometry.TriangleMesh): Input mesh to be rotated and normalized.

        Returns:
            - o3d.geometry.TriangleMesh: Rotated and normalized mesh.
        '''
        bounding_box = mesh.get_axis_aligned_bounding_box()
        center = bounding_box.get_center()
        mesh.translate(-center, relative=True)

        min_bound = bounding_box.min_bound
        max_bound = bounding_box.max_bound
        scale = np.max(max_bound - min_bound)

        vertices = np.asarray(mesh.vertices)
        vertices = vertices / scale
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        bounding_box = mesh.get_axis_aligned_bounding_box()
        return mesh
    
    @staticmethod
    def normalize_dense_grid_mesh_to_one(dense_grid_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        '''
        Normalize a dense grid mesh from coordinates [0,64]^3 (centered at (32,32,32))
        to coordinates [-0.5,0.5]^3 (centered at (0,0,0)),
        and finally rotate around the x-axis by -90 degrees.

        Args:
            - dense_grid_mesh (trimesh.Trimesh): Input dense grid mesh to be normalized and rotated.
        
        Returns:
            - trimesh.Trimesh: Normalized and rotated mesh.
        '''
        mesh = dense_grid_mesh.copy()
        mesh.apply_translation(np.array([-32.0, -32.0, -32.0], dtype=np.float64))
        mesh.apply_scale(1.0 / 64.0)
        rot_matrix = trimesh.transformations.rotation_matrix(
            angle=np.deg2rad(-90),
            direction=[1, 0, 0],
            point=[0, 0, 0]
        )
        mesh.apply_transform(rot_matrix)
        return mesh

    @staticmethod
    def get_voxelized_vertices(mesh) -> np.ndarray:
        '''
        Get voxelized vertices from a mesh for feature mapping. See also voxelize_mesh().

        Args:
            - mesh (o3d.geometry.TriangleMesh): Input mesh to be voxelized.

        Returns:
            - np.ndarray: Voxelized vertices of shape (N, 3) in the range [-0.5, 0.5].
        '''

        vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
        vertices = (vertices + 0.5) / 64 - 0.5 
        return vertices


def process_mesh(input_mesh_path, save_dir, max_extent, box_center) -> None:
    '''
    Process a single mesh: voxelize and save it's rotated voxel. Specialised for preprocessing data in parallel.
    Args:
        - input_mesh_path (str): Path to the input mesh file.
        - save_dir (str): Directory to save the output mesh.
        - max_extent (float): Maximum extent for normalization.
        - box_center (np.ndarray): Center of the bounding box for normalization.
    '''
    dense_voxel_grid = VoxelUtils.voxelize_mesh(input_mesh_path, max_extent=max_extent, box_center=box_center)
    VoxelUtils.visualize_dense_voxels(
        save_dir,
        os.path.splitext(os.path.basename(input_mesh_path))[0],
        dense_voxel_grid,
        ret_mode=['ply'],
        normalize=False,
        up=[0, 0, 1]
    )

def process_directory(work_dir):

    '''
    Process a directory containing multiple .obj mesh files to find the one with the largest bounding box volume to ensure alignment.

    Args:
        - work_dir (str): Path to the directory containing .obj mesh files.
        
    Returns:
        - list: List of tuples containing (input_mesh_path, work_dir, max_extent, box_center) for each .obj file in the directory.
    '''
    input_mesh_path_list = [
        os.path.join(work_dir, file)
        for file in os.listdir(work_dir)
        if file.endswith(".obj")
    ]

    if not input_mesh_path_list:
        print(f"No .obj files in {work_dir}, skipping...")
        return []

    input_mesh_path_list.sort(
        key=lambda path: int(os.path.basename(path).split('_')[-1].split('.')[0]) 
        if os.path.basename(path).split('_')[-1].split('.')[0].isdigit() 
        else float('inf')
    )

    max_extent_list = []
    box_center_list = []
    box_volome_list = []

    for input_mesh_path in input_mesh_path_list:
        max_extent, box_center, box_volome = VoxelUtils.get_max_extent(input_mesh_path)
        max_extent_list.append(max_extent)
        box_center_list.append(box_center)
        box_volome_list.append(box_volome)


    max_volume = max(box_volome_list)
    max_index = box_volome_list.index(max_volume)
    max_extent = max_extent_list[max_index]
    box_center = box_center_list[max_index]

    return [(input_mesh_path, work_dir, max_extent, box_center) for input_mesh_path in input_mesh_path_list]

if __name__ == "__main__":
    from pathlib import Path

    example_name = "car-luggage-rack"  # TODO: Pick an example from assets/example_objects
    root_dir = Path(__file__).parents[2]

    output_dir = root_dir / "assets" / "example_objects" / example_name

    tasks = process_directory(str(output_dir))
    Parallel(n_jobs=-1, verbose=10)(
        delayed(process_mesh)(*task) for task in tasks
    )

