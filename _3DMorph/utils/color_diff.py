import os
import sys
import tempfile
import logging
import numpy as np
import trimesh
import shutil  # for removing temp dir

from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir))

from _3DMorph.renderer.render_simple import BlenderRenderer

def obj_to_glb_with_trimesh(obj_path: str) -> str:

    stem = os.path.splitext(obj_path)[0]
    glb_path = stem + ".glb"

    scene = trimesh.load(obj_path, force='scene')
    scene.export(glb_path)
    return glb_path

def colorize_mesh_changes_to_obj(init_mesh_path: str,
                                 modified_mesh_path: str,
                                 color_rgb: tuple,
                                 out_dir: str,
                                 out_name: str = "colored_diff.obj",
                                 dist_threshold: float = 0.001) -> str:
    # helper: clip a triangle by scalar s vs 0 (keep s>=0 or s<=0)
    def _clip_triangle_by_scalar(tri_xyz, tri_s, keep_ge=True, eps=1e-12):
        # tri_xyz: (3,3), tri_s: (3,)
        poly_xyz = tri_xyz.tolist()
        poly_s   = tri_s.tolist()
        def inside(val):
            return (val >= -eps) if keep_ge else (val <= eps)
        if len(poly_xyz) < 3:
            return np.zeros((0,3), dtype=float)
        new_xyz, new_s = [], []
        n = len(poly_xyz)
        for i in range(n):
            A_xyz = np.array(poly_xyz[i], dtype=float)
            B_xyz = np.array(poly_xyz[(i+1)%n], dtype=float)
            A_s   = float(poly_s[i])
            B_s   = float(poly_s[(i+1)%n])
            Ain   = inside(A_s)
            Bin   = inside(B_s)
            if Ain:
                new_xyz.append(A_xyz)
                new_s.append(A_s)
            if Ain ^ Bin:
                t = (0.0 - A_s) / (B_s - A_s + 1e-20)
                P = A_xyz + t * (B_xyz - A_xyz)
                new_xyz.append(P)
                new_s.append(0.0)
        if len(new_xyz) >= 3:
            return np.array(new_xyz, dtype=float)
        return np.zeros((0,3), dtype=float)

    # helper: simple fan triangulation
    def _triangulate_fan(poly_xyz):
        # poly_xyz: (m,3), m>=3
        faces = []
        for k in range(1, len(poly_xyz)-1):
            faces.append([0, k, k+1])
        return np.array(faces, dtype=np.int64)

    # load meshes
    init_mesh = trimesh.load(init_mesh_path, force='mesh')
    mod_mesh  = trimesh.load(modified_mesh_path, force='mesh')
    if not isinstance(init_mesh, trimesh.Trimesh):
        init_mesh = init_mesh.dump(concatenate=True)
    if not isinstance(mod_mesh, trimesh.Trimesh):
        mod_mesh = mod_mesh.dump(concatenate=True)

    # distances per vertex (modified -> initial)
    _, distances, _ = trimesh.proximity.closest_point(init_mesh, mod_mesh.vertices)
    s_all = distances - float(dist_threshold)  # scalar field: >=0 changed, <0 unchanged

    V = mod_mesh.vertices
    F = mod_mesh.faces

    # collect new vertices/faces for two groups
    V_out = []
    F_gray = []   # unchanged
    F_red  = []   # changed

    # process per face with scalar clipping
    for fi in range(len(F)):
        tri_idx = F[fi]
        tri_xyz = V[tri_idx]                # (3,3)
        tri_s   = s_all[tri_idx]            # (3,)

        # part 1: unchanged (s <= 0)
        poly_u = _clip_triangle_by_scalar(tri_xyz, tri_s, keep_ge=False)
        if len(poly_u) >= 3:
            base = len(V_out)
            V_out.extend(poly_u.tolist())
            fan = _triangulate_fan(poly_u) + base
            F_gray.extend(fan.tolist())

        # part 2: changed (s >= 0)
        poly_c = _clip_triangle_by_scalar(tri_xyz, tri_s, keep_ge=True)
        if len(poly_c) >= 3:
            base = len(V_out)
            V_out.extend(poly_c.tolist())
            fan = _triangulate_fan(poly_c) + base
            F_red.extend(fan.tolist())

    V_out = np.array(V_out, dtype=float)
    F_gray = np.array(F_gray, dtype=np.int64) if len(F_gray) > 0 else np.zeros((0,3), dtype=np.int64)
    F_red  = np.array(F_red,  dtype=np.int64) if len(F_red)  > 0 else np.zeros((0,3), dtype=np.int64)

    # write MTL
    os.makedirs(out_dir, exist_ok=True)
    obj_path = os.path.join(out_dir, out_name)
    mtl_name = os.path.splitext(out_name)[0] + ".mtl"
    mtl_path = os.path.join(out_dir, mtl_name)

    with open(mtl_path, "w") as f:
        # gray
        f.write("newmtl mat_gray\n")
        f.write("Ka 0.6 0.6 0.6\n")  # Ka: Ambient color (how the material reflects ambient light)
        f.write("Kd 0.6 0.6 0.6\n")  # Kd: Diffuse color (how the material reflects direct light)
        f.write("Ks 0.0 0.0 0.0\n")  # Ks: Specular color (how shiny the material appears)
        f.write("Ns 300.0\n")
        f.write("illum 2\n")
        f.write("d 1.0\n\n")
        # red (custom color)
        ka_kd_values = tuple(channel / 255 for channel in color_rgb)
        f.write("newmtl mat_color\n")
        f.write(f"Ka {ka_kd_values[0]} {ka_kd_values[1]} {ka_kd_values[2]}\n")
        f.write(f"Kd {ka_kd_values[0]} {ka_kd_values[1]} {ka_kd_values[2]}\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("Ns 10.0\n")
        f.write("illum 2\n")
        f.write("d 1.0\n")

    # write OBJ (no vn to avoid mismatch)
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("o colored_diff\n")
        f.write("s off\n")
        # vertices
        for v in V_out:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # unchanged faces first
        if len(F_gray) > 0:
            f.write("g unchanged\n")
            f.write("usemtl mat_gray\n")
            for a, b, c in (F_gray + 1):
                f.write(f"f {a} {b} {c}\n")

        # changed faces
        if len(F_red) > 0:
            f.write("g changed\n")
            f.write("usemtl mat_color\n")
            for a, b, c in (F_red + 1):
                f.write(f"f {a} {b} {c}\n")

    return obj_path

def render_diff_with_blender(init_mesh_path: str,
                             modified_mesh_path: str,
                             output_dir: str = "./",
                             save_mesh: bool = True,
                             color_rgb: tuple = (255, 0, 0),
                             view_args: dict = {
                                                "offset": (-45, 30),
                                                "step": 60,
                                                "direction": 'right',
                                                "set_fov": 30
                                            },
                             n_views: int = 1,
                             dist_threshold: float = 0.001):
    workdir = tempfile.mkdtemp(prefix="colored_diff_")
    try:
        colored_obj = colorize_mesh_changes_to_obj(
            init_mesh_path=init_mesh_path,
            modified_mesh_path=modified_mesh_path,
            color_rgb=color_rgb,
            out_dir=workdir,
            out_name="colored_diff.obj",
            dist_threshold=dist_threshold
        )
        return render_mesh(colored_obj, ref_mesh_path=init_mesh_path, view_args=view_args, output_dir=output_dir, n_views=n_views)
    finally:
        try:
            if save_mesh:
                shutil.copy(colored_obj, os.path.join(output_dir, "colored_diff.obj"))
                shutil.copy(os.path.splitext(colored_obj)[0] + ".mtl", os.path.join(output_dir, "colored_diff.mtl"))
                obj_to_glb_with_trimesh(os.path.join(output_dir, "colored_diff.obj"))
                # os.remove(os.path.join(output_dir, "colored_diff.mtl"))
                # os.remove(os.path.join(output_dir, "colored_diff.obj"))
            shutil.rmtree(workdir)
        except Exception as e:
            logging.warning(f"Failed to remove temp dir {workdir}: {e}")

def render_mesh(input_mesh_path, ref_mesh_path, view_args, output_dir, n_views=1):
    renderer = BlenderRenderer(resolution=1024)
    os.makedirs(output_dir, exist_ok=True)
    images = renderer.render(
        input_mesh_path,
        ref_obj_path=ref_mesh_path,
        save_dir=output_dir,
        n_views=n_views,
        mode='linear',
        lin_view_args=view_args,
        geo_mode=False
    )
    return images

if __name__ == "__main__":
    img = render_diff_with_blender(
        init_mesh_path="",
        modified_mesh_path="",
        output_dir="",
        save_mesh=True,
        color_rgb=(255, 0, 0),
        n_views=1,
        dist_threshold=5e-2
    )
