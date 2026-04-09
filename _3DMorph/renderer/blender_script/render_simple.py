import bpy
import math
import hashlib
from mathutils import Vector
import mathutils
import json
import os
import argparse, sys, os, math, re, glob, tempfile, shutil

def init_scene(engine='CYCLES', resolution=518, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128 if not geo_mode else 32
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True
        
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def load_object(object_path):
    file_extension = object_path.split(".")[-1].lower()
    if file_extension == "obj":
        bpy.ops.import_scene.obj(filepath=object_path)
    elif file_extension in ["glb", "gltf"]:
        bpy.ops.import_scene.gltf(filepath=object_path)
    elif file_extension == "ply":
        bpy.ops.import_mesh.ply(filepath=object_path)
        for obj in bpy.context.selected_objects:
            obj.rotation_euler[0] += math.radians(90)
    else:
        raise ValueError(f"Unsupported file type: {object_path}") 
    
def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")    

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 800
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 6000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }

def scene_bbox(ref_obj=None):
    if ref_obj:
        target_obj = ref_obj
    else:
        objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        print(f"Found {len(objects)} mesh objects in the scene.")
        target_obj = objects[0]

    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    
    for coord in target_obj.bound_box:
        coord = Vector(coord)
        coord = target_obj.matrix_world @ coord
        bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
        bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))


    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene(ref_obj_path, mode='normal'):
    print(f"Normalizing scene with reference object: {ref_obj_path}")
    if not ref_obj_path == 'none':
        bpy.ops.import_scene.obj(filepath=ref_obj_path)
        ref_obj = bpy.context.scene.objects[-1]
    else:
        ref_obj = None

    bbox_min, bbox_max = scene_bbox(ref_obj)
    print(f"Original bounding box min: {bbox_min}, max: {bbox_max}")
    
    scale = 1 / max(bbox_max - bbox_min)
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.scale *= scale
    
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox(ref_obj)
    print(f"New bounding box: {bbox_min}, max: {bbox_max}")
    
    offset = -(bbox_min + bbox_max) / 2
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.location += offset

    if ref_obj is not None:
        bpy.data.objects.remove(ref_obj, do_unlink=True)

    bpy.context.view_layer.update()

    print(f"Scene normalized with scale: {scale} and offset: {offset}")
    print(f"New bounding box: {scene_bbox()}")
    return scale, offset


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def stable_hash(s: str, mod: int = 10**8):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % mod

def main(arg):
    # Setup
    init_scene(resolution=int(arg.resolution), geo_mode=arg.geo_mode.lower() == "true")
    load_object(arg.object)
    # split_mesh_normal()
    cam = init_camera()
    init_lighting()
    normalize_scene(arg.ref_object, arg.mode)

    views = json.loads(arg.views)

    output_folder = f"/dev/shm/blender_render_{stable_hash(arg.object)}"
    os.makedirs(output_folder, exist_ok=True)

    transforms = []
    images = []
    for i, view in enumerate(views):
        print(f"Rendering view {i + 1}/{len(views)}")
        cam.location = (
            view['radius'] * math.cos(view['yaw']) * math.cos(view['pitch']),
            view['radius'] * math.sin(view['yaw']) * math.cos(view['pitch']),
            view['radius'] * math.sin(view['pitch'])
        )
        cam.data.lens = 16 / math.tan(view['fov'] / 2)

        # Render
        bpy.context.scene.render.filepath = os.path.join(output_folder, f'render_{i:03d}.png')
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        # Save transform
        transforms.append({
            "file_path": f'render_{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam),
            "yaw": view['yaw'],
            "pitch": view['pitch']
        })

    # Save transforms
    with open(os.path.join(output_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--ref_object', type=str, help='Path to the reference 3D model file (optional).', default='none')
    parser.add_argument('--resolution', type=str, help='Resolution of the rendered images (default: 518)', default='518')
    parser.add_argument('--mode', type=str, help='Normalization method.', default='slat')
    parser.add_argument('--geo_mode', type=str, default='false', help='Enable or disable geo mode')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    main(args)
