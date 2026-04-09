import os
import trimesh
import numpy as np
import torch
from pytorch3d.ops import iterative_closest_point, knn_points
from pytorch3d.structures import Pointclouds
from scipy.ndimage import binary_fill_holes
import plotly.graph_objects as go


class MeshComparison:
    def __init__(self, voxel_size=0.075, number_of_points=20000, threshold=7e-7, max_iteration=20000):
        self.voxel_size = voxel_size
        self.number_of_points = number_of_points
        self.threshold = threshold
        self.max_iteration = max_iteration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    def load_mesh(self, file_path):
        '''Load a mesh using Trimesh.'''
        return trimesh.load(file_path, force='mesh')

    def sample_points(self, mesh):
        '''Sample points uniformly from a mesh.'''
        return trimesh.sample.sample_surface(mesh, self.number_of_points)[0]
    
    def rotate_mesh(self, mesh, rotation_matrix=None, euler_angles=None, degrees=True):
        '''Rotate a mesh using either a rotation matrix or Euler angles.
        
        Args:
            - mesh (trimesh.Trimesh): The mesh to be rotated.
            - rotation_matrix (np.ndarray or None): A 3x3 rotation matrix. If provided, this will be used for rotation.
            - euler_angles (tuple or None): A tuple of three angles (in degrees or radians) representing rotation around the x, y, and z axes. Used if rotation_matrix is None.
            - degrees (bool): If True, euler_angles are in degrees; if False, they are in radians.

        Returns:
            - trimesh.Trimesh: The rotated mesh.

        Examples:
        --------
        >>> rotated_mesh = rotate_mesh(mesh, euler_angles=(30, 0, 0), degrees=True)
        '''
        if rotation_matrix is not None:
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            mesh.apply_transform(transform)

        elif euler_angles is not None:
            if degrees:
                euler_angles = np.radians(euler_angles)
            rotation_matrix = trimesh.transformations.euler_matrix(*euler_angles)[:3, :3]
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            mesh.apply_transform(transform)
        else:
            raise ValueError("Either rotation_matrix or euler_angles must be provided.")

        return mesh

    def plot_mesh(self, mesh1, mesh2):
        '''
        Visualize two meshes using Plotly for notebook usage.
        '''
        vertices1 = mesh1.vertices
        faces1 = mesh1.faces
        x1, y1, z1 = vertices1[:, 0], vertices1[:, 1], vertices1[:, 2]
        i1, j1, k1 = faces1[:, 0], faces1[:, 1], faces1[:, 2]

        vertices2 = mesh2.vertices
        faces2 = mesh2.faces
        x2, y2, z2 = vertices2[:, 0], vertices2[:, 1], vertices2[:, 2]
        i2, j2, k2 = faces2[:, 0], faces2[:, 1], faces2[:, 2]

        mesh1_trace = go.Mesh3d(x=x1, y=y1, z=z1, i=i1, j=j1, k=k1, opacity=0.3, color='blue')
        mesh2_trace = go.Mesh3d(x=x2, y=y2, z=z2, i=i2, j=j2, k=k2, opacity=0.3, color='red')

        fig = go.Figure(data=[mesh1_trace,mesh2_trace])
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()

    def plot_voxel(self, voxel_grid1, voxel_grid2):
        '''
        Visualize two voxel grids using Plotly.
        '''
        min_bound = self.min_bound
        voxel_size = self.voxel_size

        def create_voxel_mesh(coords, color):
        
            meshes = []
            for coord in coords:
                x, y, z = coord * voxel_size + min_bound
                corners = np.array([
                    [x, y, z],
                    [x + voxel_size, y, z],
                    [x, y + voxel_size, z],
                    [x, y, z + voxel_size],
                    [x + voxel_size, y + voxel_size, z],
                    [x + voxel_size, y, z + voxel_size],
                    [x, y + voxel_size, z + voxel_size],
                    [x + voxel_size, y + voxel_size, z + voxel_size]
                ])

                i, j, k = np.array([
                    [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6],
                    [1, 2, 3, 4, 5, 4, 6, 7, 5, 6, 6, 7],
                    [3, 3, 1, 5, 3, 6, 7, 6, 7, 7, 4, 4]
                ])

                mesh = go.Mesh3d(
                    x=corners[:, 0],
                    y=corners[:, 1],
                    z=corners[:, 2],
                    i=i, j=j, k=k,
                    opacity=0.3,
                    color=color
                )
                meshes.append(mesh)
            return meshes
        coords1 = np.argwhere(voxel_grid1)
        coords2 = np.argwhere(voxel_grid2)

        meshes1 = create_voxel_mesh(coords1, color='blue')
        meshes2 = create_voxel_mesh(coords2, color='red')

        fig = go.Figure(data=meshes1+meshes2)
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()

    def normalize_mesh(self, mesh_gt, mesh_pre, deformation):
        '''
        Normalize two point clouds.
        
        Args:
            - mesh_gt (trimesh.Trimesh): Ground truth mesh.
            - mesh_pre (trimesh.Trimesh): Predicted mesh.
            - deformation (bool): If True, apply non-uniform scaling; otherwise, uniform scaling.

        Returns:
            - (trimesh.Trimesh, trimesh.Trimesh): Normalized ground truth and predicted meshes.
        '''
        points_gt = torch.tensor(self.sample_points(mesh_gt), dtype=torch.float32, device=self.device)
        points_pre = torch.tensor(self.sample_points(mesh_pre), dtype=torch.float32, device=self.device)

        mesh_gt.apply_translation(-mesh_gt.centroid)
        mesh_pre.apply_translation(-mesh_pre.centroid)

        bbox_gt = points_gt.max(dim=0).values - points_gt.min(dim=0).values
        bbox_pre = points_pre.max(dim=0).values - points_pre.min(dim=0).values
        if deformation:
            scale_gt = 1.0 / bbox_gt
            scale_pre = 1.0 / bbox_pre

        else:
            scale_gt = 1.0 / bbox_gt.max()
            scale_pre = 1.0 / bbox_pre.max()

        mesh_gt.apply_scale(scale_gt.cpu().numpy())
        mesh_pre.apply_scale(scale_pre.cpu().numpy())

        return mesh_gt, mesh_pre

    def icp(self, mesh_gt, mesh_pre):
        '''
        Perform Iterative Closest Point (ICP) alignment using PyTorch3D.

        Args:
            - mesh_gt (trimesh.Trimesh): Ground truth mesh.
            - mesh_pre (trimesh.Trimesh): Predicted mesh to be aligned.

        Returns:
            - trimesh.Trimesh: Aligned predicted mesh.
        '''

        points_pre = torch.tensor(self.sample_points(mesh_pre), dtype=torch.float32, device=self.device)
        points_gt = torch.tensor(self.sample_points(mesh_gt), dtype=torch.float32, device=self.device)

        pointcloud1 = Pointclouds(points=[points_pre])
        pointcloud2 = Pointclouds(points=[points_gt])


        # Perform ICP
        icp_result = iterative_closest_point(
            pointcloud1, pointcloud2, max_iterations=self.max_iteration, relative_rmse_thr=self.threshold
        )

        # Extract transformation components
        R = icp_result.RTs.R[0].cpu().numpy()  # Rotation matrix (3x3)
        T = icp_result.RTs.T[0].cpu().numpy()  # Translation vector (3,)
        s = icp_result.RTs.s[0].cpu().numpy()  # Scaling factor (scalar)

        R = torch.tensor(R, device=self.device)
        T = torch.tensor(T, device=self.device)
        s = torch.tensor(s, device=self.device)

        # Apply transformation to points1
        transformation_matrix = torch.eye(4, device=self.device)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = T
        mesh_pre.apply_transform(transformation_matrix.cpu())

        return mesh_pre

    def mesh_to_voxel(self, mesh1, mesh2):
        '''
        Convert meshes to voxel grids using trimesh and pytorch3d.
        
        Args:
            - mesh1 (trimesh.Trimesh): First mesh.
            - mesh2 (trimesh.Trimesh): Second mesh.

        Returns:
            - (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Voxel grids and sampled points for both meshes.
        '''
        bbox1_min, bbox1_max = mesh1.bounds
        bbox2_min, bbox2_max = mesh2.bounds
        min_bound = np.minimum(bbox1_min, bbox2_min)
        max_bound = np.maximum(bbox1_max, bbox2_max)
        self.min_bound = min_bound

        # Define voxel grid shape
        grid_shape = ((max_bound - min_bound) / self.voxel_size).astype(int) + 1
        voxel_grid1 = np.zeros(grid_shape, dtype=bool)
        voxel_grid2 = np.zeros(grid_shape, dtype=bool)

        points1 = self.sample_points(mesh1)
        points2 = self.sample_points(mesh2)

        for points, voxel_grid in [(points1, voxel_grid1), (points2, voxel_grid2)]:
            indices = ((points - min_bound) / self.voxel_size).astype(int)

            valid = (indices >= 0).all(axis=1) & (indices < grid_shape).all(axis=1)
            indices = indices[valid]
            voxel_grid[tuple(indices.T)] = True

        voxel_grid1 = binary_fill_holes(voxel_grid1)
        voxel_grid2 = binary_fill_holes(voxel_grid2)

        return voxel_grid1, voxel_grid2, points1, points2
    
    def compute_point_clouds_distance(self, points1: np.ndarray, points2: np.ndarray):
        '''
        Compute the distance between two normalized point clouds.
        
        Args:
            - points1 (np.ndarray): First point cloud.
            - points2 (np.ndarray): Second point cloud.

        Returns:
            - dict: Dictionary containing mean, max, and 95th percentile Hausdorff distances.
            '''
        points1 = torch.tensor(points1, dtype=torch.float32).unsqueeze(0)
        points2 = torch.tensor(points2, dtype=torch.float32).unsqueeze(0)

        knn_1_to_2 = knn_points(points1, points2, K=1)
        knn_2_to_1 = knn_points(points2, points1, K=1)

        all_distances = torch.cat([knn_1_to_2.dists.sqrt(), knn_2_to_1.dists.sqrt()], dim=1)

        average_distance = all_distances.mean().item()
        max_distance = all_distances.max().item()
        hausdorff_distance = torch.quantile(all_distances, 0.95).item()

        return {
            'mean pcd': average_distance,
            'max pcd': max_distance,
            '95th HD': hausdorff_distance
        }
    
    def dice_coefficient(self, voxel_gt, voxel_pre):
        '''
        Calculate Dice coefficient and related metrics.
        
        Args: 
            - voxel_gt (np.ndarray): Ground truth voxel grid (binary).
            - voxel_pre (np.ndarray): Predicted voxel grid (binary).
        
        Returns:
            - dict: Dictionary containing Dice, logt, lop, lou, and f1_score.
        '''
        intersection = np.sum(voxel_gt & voxel_pre)
        volume_gt = np.sum(voxel_gt)
        volume_pre = np.sum(voxel_pre)

        dice = 2 * intersection / (volume_gt + volume_pre)
        union = volume_gt + volume_pre - intersection
        lou = intersection / union if union > 0 else 0
        precision = intersection / volume_pre if volume_pre > 0 else 0
        recall = intersection / volume_gt if volume_gt > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'dice': dice,
            'logt': intersection / volume_gt if volume_gt > 0 else 0,
            'lop': intersection / volume_pre if volume_pre > 0 else 0,
            'lou': lou,
            'f1_score': f1_score,
        }

    def compare(self, path_gt, path_pre, mode='normalize&icp', mute=True, deformation=False, rotate_axis='z'):
        '''
        Main function to compare two meshes.

        Args:
            - path_gt (str): Path to the ground truth mesh file or Trimesh.mesh.
            - path_pre (str): Path to the predicted mesh file or Trimesh.mesh.
            - mode (str): Preprocessing mode ('normalize&icp' or other).
            - mute (bool): If True, suppress visualization.
            - deformation (bool): If True, apply non-uniform scaling during normalization.
            - rotation (dict): Optional rotation parameters for the predicted mesh.
                             Example: {'euler_angles': (30, 0, 0), 'degrees': True}

        Returns:
            dict: Comparison metrics including Dice coefficient and point cloud distances.
        '''
        if not (isinstance(path_gt, trimesh.Trimesh) and isinstance(path_pre, trimesh.Trimesh)):
            mesh_gt = self.load_mesh(path_gt)
            mesh_pre = self.load_mesh(path_pre)
        else:
            mesh_gt = path_gt
            mesh_pre = path_pre

        if mesh_gt is None or mesh_pre is None or mesh_gt.is_empty or mesh_pre.is_empty:
            zero_metrics = {
                'mean pcd': 0.0,
                'max pcd': 0.0,
                '95th HD': 0.0,
                'dice': 0.0,
                'logt': 0.0,
                'lop': 0.0,
                'lou': 0.0,
                'f1_score': 0.0,
            }
            return zero_metrics, 0

        best_metrics = None
        best_rotation = None
        angles = [0, 90, 180, 270]
        if rotate_axis == 'none':
            angles = [0]
        for angle in angles:
            if rotate_axis == 'x':
                current_rotation = {
                    'euler_angles': (angle, 0, 0),
                    'degrees': True
                }
            elif rotate_axis == 'y':
                current_rotation = {
                    'euler_angles': (0, angle, 0),
                    'degrees': True
                }
            elif rotate_axis == 'z':
                current_rotation = {
                    'euler_angles': (0, 0, angle),
                    'degrees': True
                }
            else:
                current_rotation = {
                    'euler_angles': (0, 0, 0),
                    'degrees': True
                }

            rotated_mesh_pre = self.rotate_mesh(mesh_pre.copy(), **current_rotation)

            if mode == 'normalize&icp':
                mesh_gt, rotated_mesh_pre = self.normalize_mesh(mesh_gt, rotated_mesh_pre, deformation)
                rotated_mesh_pre = self.icp(mesh_gt, rotated_mesh_pre)
                # print(f"Applied ICP with rotation: {angle} degrees")

            voxel_gt, voxel_pre, points_gt, points_pre = self.mesh_to_voxel(mesh_gt, rotated_mesh_pre)

            dice = self.dice_coefficient(voxel_gt, voxel_pre)  
            if dice['dice'] >= 0.9:
                best_metrics = dice
                best_rotation = angle
                break  
            if best_metrics is None or dice['dice'] > best_metrics['dice']:

                best_metrics = dice
                best_rotation = angle

        if not mute:
            self.plot_mesh(mesh_gt, rotated_mesh_pre)
            self.plot_voxel(voxel_gt, voxel_pre)

        pcd_distance = self.compute_point_clouds_distance(points_gt, points_pre)
        best_metrics = pcd_distance | best_metrics
        print(f"\n Best f1_score: {best_metrics['f1_score']:.4f} at rotation {best_rotation} degrees")
        del mesh_gt, mesh_pre, rotated_mesh_pre, voxel_gt, voxel_pre, points_gt, points_pre
        torch.cuda.empty_cache()
        return best_metrics, best_rotation




if __name__ == "__main__":
    from pathlib import Path

    example_name = "car-luggage-rack"  # TODO: Pick an example from assets/example_objects
    root_dir = Path(__file__).parents[2]

    output_dir = root_dir / "assets" / "example_objects" / example_name
    torch.cuda.empty_cache()
    comparator = MeshComparison()

    name = output_dir / "unmodified.obj"  # Path to ground-truth mesh
    name2 = output_dir / "unmodified_voxels.ply"  # TODO: Path to generated mesh
    results = comparator.compare(str(name), str(name2) , mode='normalize&icp', mute=True, deformation=False)
    print("Comparison Results:", results)
