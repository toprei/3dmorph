import os
import sys
import cv2
import trimesh
from PIL import Image
import math
import json
import torch
import utils3d
import numpy as np
import math
import open3d as o3d
import torch.nn.functional as F
from copy import deepcopy
from skimage.metrics import structural_similarity as ssim

from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir))

from _3DMorph.renderer.render_simple import BlenderRenderer
from _3DMorph.utils.voxel_utils import VoxelUtils


class MaskPredictor:
    def blender_c2w_to_w2c(self, c2w):
        '''
        Convert Blender camera-to-world matrix to world-to-camera matrix.
        '''
        c2w = torch.tensor(c2w)
        c2w[:3, 1:3] *= -1
        return torch.inverse(c2w)
    
    def get_bounding_box(self, mesh_path):
        '''
        Get the bounding box size of a 3D mesh.

        Returns:
            size (list): List of three floats representing the size of the bounding box along each axis.
        '''
        mesh = trimesh.load(mesh_path)
        size = mesh.extents
        return size

    def get_pixel_width(self, pil_image: Image.Image):
        '''
        Get the pixel width and height of the non-transparent area in a PIL image.

        Returns:
            - size (list): List of two integers representing the pixel width and height.
        '''
        if pil_image.mode == 'RGBA':
            alpha = np.array(pil_image)[:, :, 3]
            pil_image = pil_image.convert('RGB')
        else:
            raise ValueError("Image must have an alpha channel (RGBA)")

        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


        ys, xs = np.where(alpha > 0)

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        pixel_width = x_max - x_min + 1
        pixel_height = y_max - y_min + 1

        return [pixel_width, pixel_height]

    def pixel_length_of_object(self, mesh_path, top_view_image: Image.Image):
        '''
        Calculate the real-world length represented by each pixel in the top-down view of a 3D mesh.

        Returns:
            - width_2d (float): Real-world length per pixel along the width.
            - length_2d (float): Real-world length per pixel along the length.
    
        Caution:
            - This function assumes that the mesh is aligned with the axes and that the top-down view is taken from directly above (y-axis).
        '''
        image_size = self.get_pixel_width(top_view_image)
        mesh_size = self.get_bounding_box(mesh_path)
        width_2d = mesh_size[2] / image_size[0]
        length_2d = mesh_size[0] / image_size[1]

        return width_2d, length_2d
    
    def rotate_and_normalize_mesh(self, mesh, mesh_ref=None):
        '''
        Rotate and normalize a 3D mesh to fit within a unit cube centered at the origin.

        Args:
            - mesh (o3d.geometry.TriangleMesh): The input 3D mesh to be normalized.
            - mesh_ref (o3d.geometry.TriangleMesh or None): Reference mesh for size normalization. If None, the input mesh is used.

        Returns:
            - new_mesh (o3d.geometry.TriangleMesh): The rotated and normalized mesh.
        '''
        if mesh_ref is None:
            bounding_box = mesh.get_axis_aligned_bounding_box()
            center = bounding_box.get_center()
        else:
            bounding_box = mesh_ref.get_axis_aligned_bounding_box()
            center = bounding_box.get_center()

        def normalize_mesh(mesh, bounding_box, center):
            mesh.translate(-center, relative=True)
            translate=center
            min_bound, max_bound = bounding_box.min_bound, bounding_box.max_bound
            scale = max(max_bound - min_bound)
            mesh.scale(1.0 / scale, center=(0, 0, 0))
            vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
            mesh.vertices = o3d.utility.Vector3dVector(vertices) 
            return mesh, scale, translate
        
        new_mesh, scale, translate = normalize_mesh(mesh, bounding_box, center)
        rotation_matrix = new_mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        new_mesh.rotate(rotation_matrix, center=(0, 0, 0))
        return new_mesh, scale, translate

    def extract_uv(self, extrinsics, intrinsics, positions, device='cuda'):
        '''
        Extract UV coordinates for given 3D positions using camera extrinsics and intrinsics.

        Args:
            - extrinsics (list of torch.Tensor): List of camera extrinsic matrices (4x4).
            - intrinsics (list of torch.Tensor): List of camera intrinsic matrices (3x3).
            - positions (torch.Tensor): Tensor of 3D positions (N x 3) from the voxelized mesh.
        Returns:
            - uv (torch.Tensor): Tensor of UV coordinates (N x 2) in the range [-1, 1].
        '''
        N = positions.shape[0]
        positions = positions.float().to(device)
        indices = ((positions + 0.5) * 64).long()
        assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"

        batch_extrinsics = extrinsics[0].to(device)
        batch_intrinsics = intrinsics[0].to(device)

        uv = utils3d.torch.project_cv(positions, extrinsics=batch_extrinsics, intrinsics=batch_intrinsics)[0]
        uv = uv * 2 - 1
        return uv
        
    def extract_contour_bboxes(self, diff_image, min_area=50):
        '''
        Extract bounding boxes from the difference image using contours.

        Args:
            - diff_image (PIL.Image or np.ndarray): The difference image (grayscale).
            - min_area (int): Minimum area threshold to filter out small contours.

        Returns:
            - vis_image (PIL.Image): Image with bounding boxes drawn.
            - mask_image (PIL.Image): Binary mask image with bounding boxes filled.
            - bboxes (list of tuples): List of bounding boxes in the format (center_x, center_y, width, height).
        '''

        if isinstance(diff_image, Image.Image):
            diff_np = np.array(diff_image.convert("L"))
        else:
            diff_np = diff_image.copy()
            if diff_np.ndim == 3:
                diff_np = cv2.cvtColor(diff_np, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(diff_np, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        img_color = cv2.cvtColor(diff_np, cv2.COLOR_GRAY2BGR)
        mask = np.zeros_like(diff_np, dtype=np.uint8)

        raw_bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            raw_bboxes.append((x, y, w, h))

        filtered_bboxes = []
        for i, (x1, y1, w1, h1) in enumerate(raw_bboxes):
            keep = True
            for j, (x2, y2, w2, h2) in enumerate(raw_bboxes):
                if i != j:
                    if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                        if w2 * h2 > w1 * h1:
                            keep = False
                            break
            if keep:
                filtered_bboxes.append((x1, y1, w1, h1))

        bboxes = []
        for (x, y, w, h) in filtered_bboxes:
            cx, cy = x + w // 2, y + h // 2
            bboxes.append((cx, cy, w, h))

            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img_color, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(img_color, f"C:({cx},{cy})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(img_color, f"W:{w}, H:{h}", (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, cv2.FILLED)

        return Image.fromarray(img_color), Image.fromarray(mask), bboxes


    def get_ssim_diff(self, img1_path, img2_path, min_area=50):
        '''
        Compute SSIM(Structural Similarity Index) between two images and extract bounding boxes of differing regions.

        Args:
            - img1_path (str): Path to the first image.
            - img2_path (str): Path to the second image.
            - min_area (int): Minimum area threshold to filter out small contours.

        Returns:
            - vis_img (PIL.Image): Image with bounding boxes drawn.
            - bboxes (list of tuples): List of bounding boxes in the format (center_x, center_y, width, height).
        '''
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if img1.shape != img2.shape:
            H, W = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (W, H))
            img2 = cv2.resize(img2, (W, H))

        score, diff = ssim(img1, img2, full=True)
        diff = (1 - diff) * 255
        diff = diff.astype(np.uint8)

        vis_img, mask_img, bbox = self.extract_contour_bboxes(diff, min_area=min_area)
        return vis_img, bbox

    def white_background(self, image_path):
        '''
        Convert an image with alpha channel to one with a white background.

        Args:
            - image_path (str): Path to the input image.

        Returns:
            - image (PIL.Image): Image with a white background.
        '''
        image = Image.open(image_path)
        if image.mode != 'RGBA':
            return image.convert('RGB')
        white_image = Image.new("RGB", image.size, (255, 255, 255))
        white_image.paste(image, mask=image.split()[3])
        return white_image

    def render_top_view(self, input_mesh_path, res):
        '''
        Render a top-down view of a 3D mesh using Blender.
        '''
        renderer = BlenderRenderer(resolution=res)
        view_args = {'offset': (0, 90)}
        images = renderer.render(input_mesh_path, n_views=1, mode='linear', lin_view_args=view_args)
        top_view_image = images[0]
        return top_view_image

    def get_3d_dimensions(self, mesh_path, top_view_image, pitch, yaw, _2d_box_width, _2d_box_height):
        '''
        Calculate the 3D dimensions of an object based on its 2D bounding box.

        Algorithm:
        1. Convert pitch and yaw from degrees to radians.
        2. Calculate the length and height in 3D space using the formulas:
                - length_3d = |_2d_box_width / sin(yaw)|
                - height_3d = |_2d_box_height / cos(pitch)|
        3. Determine the real-world length per pixel using the pixel_length_of_object function.
        4. Calculate the final 3D dimensions:
                - mask_length_in_object = length_3d * per_pixel_length
                - mask_width_in_object = height_3d * per_pixel_width

        Args:
            - mesh_path (str): Path to the 3D mesh file.
            - top_view_image (PIL.Image): Top-down view image of the object. 
            - pitch (float): Pitch angle in degrees.
            - yaw (float): Yaw angle in degrees.
            - _2d_box_width (float): Width of the 2D bounding box in pixels.
            - _2d_box_height (float): Height of the 2D bounding box in pixels.

        Returns:
            - mask_length_in_object (float): Length of the object in 3D space.
            - mask_width_in_object (float): Width of the object in 3D space.
        '''
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        length_3d = np.abs(_2d_box_width / math.sin(yaw))
        height_3d = np.abs(_2d_box_height / math.cos(pitch))

        per_pixel_width, per_pixel_length = self.pixel_length_of_object(mesh_path, top_view_image)
        assert abs(per_pixel_width - per_pixel_length) < 1e-1, "Pixel width and length should be approximately equal"

        mask_length_in_object = length_3d * per_pixel_length
        mask_width_in_object = height_3d * per_pixel_width
        return mask_length_in_object, mask_width_in_object

    def get_uv_and_translate(self, transforms_json_path, input_mesh_path, image_width=1024, image_height=1024, device='cuda'):
        '''
        Get UV coordinates and translation for a 3D mesh. 
        ### This is an improvement over the previous version by ensuring mapping only visible vertices by computing a visibility mask.

        Args:
            - transforms_json_path (str): Path to the JSON file containing camera transforms.
            - input_mesh_path (str): Path to the input 3D mesh file.
            - image_width (int): Width of the rendered images.
            - image_height (int): Height of the rendered images.
            - device (str): Device to use for computations ('cuda' or 'cpu').

        Returns:
            - mesh_rot (o3d.geometry.TriangleMesh): The rotated and normalized mesh.
            - uv (torch.Tensor): UV coordinates for the mesh vertices.
            - scale (float): Scale factor used for normalization.
            - translate (np.ndarray): Translation vector used for normalization.
            - positions (torch.Tensor): Voxelized vertex positions.
            - visible_mask (torch.Tensor): Mask indicating which vertices are visible in the first view.
        '''
        with open(transforms_json_path, 'r') as f:
            transforms = json.load(f)

        fov = torch.tensor([transform['camera_angle_x'] for transform in transforms])
        intrinsics = [utils3d.torch.intrinsics_from_fov_xy(fv, fv).clone().detach().float() for fv in fov]
        extrinsics = [self.blender_c2w_to_w2c(transform['transform_matrix']).clone().detach().float() for transform in transforms]
        mesh = o3d.io.read_triangle_mesh(input_mesh_path)
        mesh_rot, scale, translate = self.rotate_and_normalize_mesh(mesh)
        positions = torch.tensor(VoxelUtils.get_voxelized_vertices(mesh_rot)).float()
        uv = self.extract_uv(extrinsics, intrinsics, positions, device=device)
        visible_mask = self.compute_visible_mask(
            positions=positions.to(device),
            extrinsic=extrinsics[0].to(device),
            intrinsic=intrinsics[0].to(device),
            image_width=image_width,
            image_height=image_height,
            device=device
        )

        return mesh_rot, uv, scale, translate, positions, visible_mask


    def get_uv_and_translate_using_gt(self, transforms_json_path, input_mesh_path, size_ref_mesh_path, image_width=1024, image_height=1024, device='cuda'):
        '''
        Get UV coordinates and translation for a 3D mesh using size_ref_mesh_path, this depends on how the image is rendered.
        ### This is an improvement over the previous version by ensuring mapping only visible vertices by computing a visibility mask.

        Args:
            - transforms_json_path (str): Path to the JSON file containing camera transforms.
            - input_mesh_path (str): Path to the input 3D mesh file.
            - image_width (int): Width of the rendered images.
            - image_height (int): Height of the rendered images.
            - device (str): Device to use for computations ('cuda' or 'cpu').

        Returns:
            - mesh_rot (o3d.geometry.TriangleMesh): The rotated and normalized mesh.
            - uv (torch.Tensor): UV coordinates for the mesh vertices.
            - scale (float): Scale factor used for normalization.
            - translate (np.ndarray): Translation vector used for normalization.
            - positions (torch.Tensor): Voxelized vertex positions.
            - visible_mask (torch.Tensor): Mask indicating which vertices are visible in the first view.
        '''
        with open(transforms_json_path, 'r') as f:
            print(f"Using path:{transforms_json_path}")
            transforms = json.load(f)
            
        fov = torch.tensor([transform['camera_angle_x'] for transform in transforms])
        intrinsics = [utils3d.torch.intrinsics_from_fov_xy(fov_value, fov_value).clone().detach().float() for fov_value in fov]
        extrinsics = [self.blender_c2w_to_w2c(transform['transform_matrix']).clone().detach().float() for transform in transforms]
        mesh = o3d.io.read_triangle_mesh(input_mesh_path)
        mesh_ref = o3d.io.read_triangle_mesh(size_ref_mesh_path)
        mesh_rot, scale, translate = self.rotate_and_normalize_mesh(mesh, mesh_ref)
        positions = torch.tensor(VoxelUtils.get_voxelized_vertices(mesh_rot)).float()
        uv = self.extract_uv(extrinsics, intrinsics, positions)
        visible_mask = self.compute_visible_mask(
            positions=positions.to(device),
            extrinsic=extrinsics[0].to(device),
            intrinsic=intrinsics[0].to(device),
            image_width=image_width,
            image_height=image_height,
            device=device
        )
        return mesh_rot, uv, scale, translate, positions, visible_mask

    def query_uv(self, uv, positions, center_x, center_y, image_width=1024, image_height=1024, device='cuda', visible_mask=None):
        '''
        Query the 3D position corresponding to a given UV coordinate.

        Args:
            - uv (torch.Tensor): UV coordinates for the voxel centers (N x 2).
            - positions (torch.Tensor): Voxelized vertex positions (N x 3).
            - center_x (float): X coordinate of the query point in pixel space.
            - center_y (float): Y coordinate of the query point in pixel space.
            - image_width (int): Width of the rendered images.
            - image_height (int): Height of the rendered images.
            - device (str): Device to use for computations ('cuda' or 'cpu').
            - visible_mask (torch.Tensor or None): Optional mask indicating which vertices are visible.

        Debug: Generates a debug image showing UV points and the query point.

        Returns:    
            - matching_position (torch.Tensor): The 3D position corresponding to the nearest UV coordinate.
            - min_dist (float): The minimum distance in UV space to the query point.
        '''

        uv_query_x = (center_x / image_width) * 2 - 1
        uv_query_y = (center_y / image_height) * 2 - 1
        uv_query = torch.tensor([uv_query_x, uv_query_y], device=device)

        if visible_mask is not None:
            mask = visible_mask.to(device)
            positions = positions.to(device)
            uv_used = uv[mask]
            positions_used = positions[mask]
        else:
            uv_used = uv
            positions_used = positions

        dists = torch.norm(uv_used - uv_query[None, :], dim=1)
        min_rel = torch.argmin(dists)
        min_dist = dists[min_rel]
        matching_position = positions_used[min_rel].cpu()
        return matching_position, min_dist.item()

    def uv_debug_colored(
        self,
        uv: torch.Tensor,
        query_uv: torch.Tensor,
        nearest_uv: torch.Tensor,
        image_width=1024,
        image_height=1024,
        device="cuda",
        flip_y=False,
        point_size=1,
        visible_mask=None
    ):
        uv = uv.to(device)
        img = torch.zeros((image_height, image_width, 3), dtype=torch.uint8, device=device)

        def ndc_to_pix(u, v):
            x = (u + 1) * 0.5 * (image_width - 1)
            if flip_y:
                y = (1 - (v + 1) * 0.5) * (image_height - 1)
            else:
                y = (v + 1) * 0.5 * (image_height - 1)
            return int(round(x.item())), int(round(y.item()))

        def draw_point(x, y, color):
            for dx in range(-point_size, point_size + 1):
                for dy in range(-point_size, point_size + 1):
                    xi, yi = x + dx, y + dy
                    if 0 <= xi < image_width and 0 <= yi < image_height:
                        img[yi, xi] = torch.tensor(color, dtype=torch.uint8, device=device)

        if visible_mask is None:
            visible_mask = torch.ones(uv.shape[0], dtype=torch.bool, device=device)

        for (u, v), vis in zip(uv, visible_mask):
            x, y = ndc_to_pix(u, v)
            if vis:


                draw_point(x, y, (100, 100, 100))

        qx, qy = ndc_to_pix(query_uv[0], query_uv[1])
        draw_point(qx, qy, (255, 0, 0))

        nx, ny = ndc_to_pix(nearest_uv[0], nearest_uv[1])
        draw_point(nx, ny, (0, 0, 255))

        return img

    def compute_visible_mask(self, positions, extrinsic, intrinsic,
                            image_width=1024, image_height=1024, device='cuda',
                            bin_size=5, eps=1e-2):
        '''
        Core algorithm to compute visibility mask for 3D points.

        Args:
            - positions (torch.Tensor): Tensor of 3D positions (N x 3).
            - extrinsic (torch.Tensor): Camera extrinsic matrix (4x4).
            - intrinsic (torch.Tensor): Camera intrinsic matrix (3x3).
            - image_width (int): Width of the rendered image.
            - image_height (int): Height of the rendered image.
            - device (str): Device to use for computations ('cuda' or 'cpu').
            - bin_size (int): Size of the bins to group pixels.
            - eps (float): Small epsilon value to handle numerical precision.
        '''
        positions = positions.to(device).float()
        extrinsic = extrinsic.to(device).float()
        intrinsic = intrinsic.to(device).float()

        uv01, depth = utils3d.torch.project_cv(positions, extrinsic, intrinsic)
        in_front = depth > 0

        u_pix = uv01[:, 0] * (image_width  - 1)
        v_pix = uv01[:, 1] * (image_height - 1)

        Wb = (image_width  + bin_size - 1) // bin_size
        Hb = (image_height + bin_size - 1) // bin_size

        ui_b = torch.floor(u_pix / bin_size).long()
        vi_b = torch.floor(v_pix / bin_size).long()

        in_wb = (ui_b >= 0) & (ui_b < Wb)
        in_hb = (vi_b >= 0) & (vi_b < Hb)
        valid = in_front & in_wb & in_hb

        N = positions.shape[0]
        if not torch.any(valid):
            return torch.zeros(N, dtype=torch.bool, device=device)

        flat_b = (vi_b[valid] * Wb + ui_b[valid]).view(-1)
        z_v    = depth[valid].view(-1)

        depth_min = torch.full((Hb * Wb,), float('inf'), device=device)
        depth_min.scatter_reduce_(0, flat_b, z_v, reduce="amin", include_self=True)

        z_ref = depth_min.index_select(0, flat_b)
        is_best = z_v <= (z_ref + eps)

        visible_mask = torch.zeros(N, dtype=torch.bool, device=device)
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        visible_mask[valid_idx[is_best]] = True

        print(f"[mask] valid={int(valid.sum())} visible={int(visible_mask.sum())} bins={Wb}x{Hb}")
        return visible_mask

    def transform_center(self, center_raw, scale, translate): 
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        center_rot = center_raw @ rotation_matrix
        center_rot_scale = center_rot * scale
        center = center_rot_scale + translate
        return center

    def make_bb_box(self, bb_box_dim, center):
        dx = bb_box_dim[0] / 2
        dy = bb_box_dim[1] / 2
        dz = bb_box_dim[2] / 2

        vertices = np.array([
            [center[0] - dx, center[1] - dy, center[2] - dz],
            [center[0] + dx, center[1] - dy, center[2] - dz],
            [center[0] + dx, center[1] + dy, center[2] - dz],
            [center[0] - dx, center[1] + dy, center[2] - dz],
            [center[0] - dx, center[1] - dy, center[2] + dz],
            [center[0] + dx, center[1] - dy, center[2] + dz],
            [center[0] + dx, center[1] + dy, center[2] + dz],
            [center[0] - dx, center[1] + dy, center[2] + dz],
        ])

        faces = [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
            [0, 3, 7], [0, 7, 4],
        ]

        bounding_box_mesh = o3d.geometry.TriangleMesh()
        bounding_box_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        bounding_box_mesh.triangles = o3d.utility.Vector3iVector(faces)
        return bounding_box_mesh

    def run(self, input_mesh_path, unmod_image_path, mod_image_path, transforms_json_path, ref_mesh_path=None, device='cuda', res=1024):
        '''
        Main function to run the mask prediction pipeline.
        
        Args:
            - input_mesh_path (str): Path to the input 3D mesh file.
            - unmod_image_path (str): Path to the unmodified image file.
            - mod_image_path (str): Path to the modified image file.
            - transforms_json_path (str): Path to the JSON file containing camera transforms.
            - ref_mesh_path (str or None): Path to the reference mesh file for size normalization.
            - device (str): Device to use for computations ('cuda' or 'cpu').
            - res (int): Resolution for rendering and image processing.

        Returns:
            - combined_mesh (o3d.geometry.TriangleMesh): The combined mesh with bounding boxes.
            - mesh_rot (o3d.geometry.TriangleMesh): The rotated and normalized input mesh.
            - bounding_box_mesh_list (list of o3d.geometry.TriangleMesh): List of bounding box meshes.
            - diff_img (PIL.Image): Image showing the differences between unmodified and modified images.

        '''
        with open(transforms_json_path, 'r') as f:
            transforms = json.load(f)

        yaw, pitch = math.degrees(transforms[0]['yaw']), math.degrees(transforms[0]['pitch'])
        diff_img, bb_boxes_params = self.get_ssim_diff(unmod_image_path, mod_image_path)
        top_view_image = self.render_top_view(input_mesh_path, res)

        if ref_mesh_path is None:
            mesh_rot, uv, scale, translate, positions, visible_mask = self.get_uv_and_translate(transforms_json_path, input_mesh_path, image_height=res, image_width=res, device=device)
        else:
            mesh_rot, uv, scale, translate, positions, visible_mask = self.get_uv_and_translate_using_gt(transforms_json_path, input_mesh_path, ref_mesh_path, image_height=res, image_width=res, device=device)


        combined_mesh = deepcopy(mesh_rot)
        bounding_box_mesh_list = []
        for bb_box_params in bb_boxes_params:
            cx, cy, w, h = bb_box_params
            mask_length, mask_width = self.get_3d_dimensions(
                input_mesh_path,
                top_view_image,
                pitch,
                yaw,
                w,
                h
            )
            matching_position, min_dist = self.query_uv(uv, positions, cx, cy, res, res,device=device, visible_mask=visible_mask)
            print(f"Min Dist: {min_dist}")

            # Assume mask_width as mask_depth
            bounding_box_mesh = self.make_bb_box([mask_length / scale, mask_width / scale, mask_width / scale], matching_position)

            combined_mesh = combined_mesh + bounding_box_mesh

            bounding_box_mesh_list.append(bounding_box_mesh)

        return combined_mesh, mesh_rot, bounding_box_mesh_list, diff_img
    
    # for example use
    def predict_mask(self, input_mesh_path, unmod_image_path, mod_image_path, transforms_json_path, resolution=1024):
        '''
        Predict a 3D mask from the difference between unmodified and modified images. (including run(), this is designed for eaxmple_pipeline.py for easy use)

        Args:
            - input_mesh_path (str): Path to the input 3D mesh file.
            - unmod_image_path (str): Path to the unmodified image file.
            - mod_image_path (str): Path to the modified image file.
            - transforms_json_path (str): Path to the JSON file containing camera transforms.
            - resolution (int): Resolution for rendering and image processing.

        Returns:
            - bbox_mask (torch.Tensor): The predicted 3D mask as a voxel grid.
            - diff_img (PIL.Image): Image showing the differences between unmodified and modified images.'''
        white_background = self.white_background(unmod_image_path)
        white_background.save(unmod_image_path)

        _, mesh_rot, mask_box_mesh_list, diff_img = self.run(input_mesh_path, unmod_image_path, mod_image_path, transforms_json_path, res=resolution)
        max_extents = VoxelUtils.get_max_extent(mesh_rot)

        union_voxel = None
        for mask_box_mesh in mask_box_mesh_list:
            mask_voxel = VoxelUtils.voxelize_mesh(
                o3d_mesh=mask_box_mesh, max_extent=max_extents[0], box_center=max_extents[1]
            )
            mask_voxel_filled = VoxelUtils.fill_voxel(mask_voxel)

            mask_voxel_filled = (mask_voxel_filled > 0).float()
            if union_voxel is None:
                union_voxel = mask_voxel_filled

            else:
                union_voxel = torch.maximum(union_voxel, mask_voxel_filled)

        if union_voxel is not None:
            mask_downsampled = F.avg_pool3d(union_voxel, kernel_size=4, stride=4)
            mask_downsampled = (mask_downsampled > 0).float()
            mask_downsampled = mask_downsampled.repeat(1, 8, 1, 1, 1)
            nonzero_coords = (mask_downsampled > 0).nonzero(as_tuple=False)[:, -3:]

            zmin, ymin, xmin = nonzero_coords.min(dim=0).values.tolist()
            zmax, ymax, xmax = nonzero_coords.max(dim=0).values.tolist()
            bbox_mask = torch.zeros_like(mask_downsampled)
            bbox_mask[..., zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = 1.0

        else:
            bbox_mask = torch.zeros((1, 8, 16, 16, 16), dtype=torch.float32, device='cuda')
            
        return bbox_mask, diff_img


if __name__ == "__main__":
    example_name = "car-luggage-rack"  # TODO: Pick an example from assets/example_objects

    output_dir = root_dir / "assets" / "example_objects" / example_name
    input_mesh_path = output_dir / "unmodified.obj"
    unmod_image_path = output_dir / "explore_inpaint" / "unmodified.png"
    mod_image_path = output_dir / "explore_inpaint" / "modified.png"
    transforms_json_path = output_dir / "explore_inpaint" / "transforms.json"

    predictor = MaskPredictor()
    combined_mesh, _, _, diff_img = predictor.run(input_mesh_path, unmod_image_path, mod_image_path, transforms_json_path, res=1024, device='cuda')

    # Save input_mesh + the computed bounding box
    o3d.io.write_triangle_mesh(output_dir / "run_mask_box.obj", combined_mesh)
    # Save the ssim difference as grey-scale image
    diff_img.save(output_dir / "diff_image.png")
