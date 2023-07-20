import blenderproc as bproc
import numpy as np
import argparse
import random
import os
from pathlib import Path
import bpy
import json

parser = argparse.ArgumentParser()
# parser.add_argument('insert_mold', nargs='?', default="Cobot/model/insert_mold.obj", help="Path to the scene.obj file")
# parser.add_argument('mainshell', nargs='?', default="Cobot/model/mainshell.obj", help="Path to the scene.obj file")

# parser.add_argument('marble', nargs='?', default="Cobot/background/marble.obj", help="Path to the object file with the ground object")
parser.add_argument('output_dir', nargs='?', default="Cobot/output/dataset2_test3", help="Path to where the final files, will be saved")
parser.add_argument('image_dir', nargs='?', default="Cobot/background", help="Path to a folder with .jpg textures to be used in the sampling process")
args = parser.parse_args()

bproc.init()


# load the objects into the scene
obj_queue = []
for obj in Path("Cobot/model").rglob('*.obj'):
    for _ in range(5):
        parts = ['insert_mold','mainshell','topshell']
        obj_queue.append(bproc.loader.load_obj(str(obj)).pop())
        part = obj_queue[-1]
        idx = parts.index(obj.name[:-4])
        part.set_cp("category_id", idx + 1)
        part.set_name(parts[idx])
        print('category_id =', part.get_cp("category_id"))
        print('Part Name =', part.get_name())
# objs2 = bproc.loader.load_obj(args.mainshell)
# ground = bproc.loader.load_obj(args.ground_obj)[0]

for obj in Path("Cobot/background").rglob('*.obj'):
    ground = bproc.loader.load_obj(str(obj)).pop()
    pose = np.eye(4)
    pose[:, 3] = np.array([0, 0, 0-0.01, 1]).T
    ground.set_local2world_mat(pose)
    ground.set_scale([0.3,0.3,0.5])
    # ground.set_rotation_euler(np.array([-np.pi/2, 0, 0]))
    # ground.set_rotation_euler(np.array([0,0,0]))#([-np.pi/2, np.pi, np.random.uniform(0, 2*np.pi, 1).item()]))
    ground.set_cp("category_id", -1)
    ground.set_name("ground")
    # Go through all objects
    # Find all materials
    materials = bproc.material.collect_all()

    # # Find the material of the ground object
    # ground_material = bproc.filter.one_by_attr(materials, "name", "Material.001")
    # # Set its displacement based on its base color texture
    # ground_material.set_displacement_from_principled_shader_value("Base Color", multiply_factor=1.5)

    # Collect all jpg images in the specified directory
    images = list(Path(args.image_dir).absolute().rglob("texture.jpg"))
    
    # Load one random image
    image = bpy.data.images.load(filepath=str(random.choice(images)))
    # Set it as base color of the current material
    for m in ground.get_materials():
            m.set_principled_shader_value('Base Color', image)

   
# Set some category ids for loaded objects
# for j, obj in enumerate(objs):
#     obj.set_cp("category_id", j + 1)
#     obj.set_name("insert_mold")
#     print('category_id =', obj.get_cp("category_id"))
#     print('Part Name =', obj.get_name())


# for i, obj2 in enumerate(objs2):
#     obj2.set_cp("category_id", i + 1)
#     obj2.set_name("mainshell")
#     print('category_id =', obj2.get_cp("category_id"))
#     print('Part Name =', obj2.get_name())




# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
# light.set_location([5, -5, 5])
# light.set_energy(1000)
light.set_location([0,0, random.uniform(1, 4)])
light.set_energy(np.random.uniform(50,100,1).item())

# Find point of interest, all cam poses should look towards it
poi = bproc.object.compute_poi(obj_queue)
# Set camera resolution:
bproc.camera.set_resolution(image_width=640, image_height=640)
# Sample 5 camera poses
for i in range(5):
    # Sample random camera location above objects
    location = [random.uniform(-0.2,0.2), random.uniform(-0.2, 0.2), random.uniform(0.3,0.5)]#np.random.uniform([-10, -10, 8], [10, 10, 12])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=0)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

# Define a function that samples the pose of a given sphere
def sample_pose(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([-0.08, -0.08, 0.015], [0.08, 0.08, 0.015]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# Sample the poses of all usb2 above the ground without any collisions in-between
bproc.object.sample_poses(
    obj_queue,
    sample_pose_func=sample_pose
)

# bproc.object.sample_poses(
#     objs2,
#     sample_pose_func=sample_pose
# )

# Make all usb2 actively participate in the simulation
for part in obj_queue:
    part.enable_rigidbody(active=True, collision_shape="COMPOUND", mass=0.01)
    part.build_convex_decomposition_collision_shape('blenderproc_resources/vhacd')

# for obj2 in objs2:
#     obj2.enable_rigidbody(active=False)
# The ground should only act as an obstacle and is therefore marked passive.
# To let the usb2 fall into the valleys of the ground, make the collision shape MESH instead of CONVEX_HULL.
ground.enable_rigidbody(active=False, collision_shape="CONVEX_HULL", mass = 1)

# Run the simulation and fix the poses of the usb2 at the end
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=20, check_object_interval=1)

# activate normal rendering
bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(map_by=["category_id","instance", "name"])

# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)

#Write data to coco file
bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    mask_encoding_format='polygon',
                                    color_file_format="PNG", 
                                    append_to_existing_output=True)


