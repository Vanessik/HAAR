import os
import sys
import shutil

import numpy as np

sys.path.append(os.getcwd())

import bpy

sys.path.append('/home/vsklyarova/miniconda3/envs/pytorch3d/lib/python3.9/site-packages/')
sys.path.append('/home/vsklyarova/snap/firefox/common/Downloads/ls/lib/python3.10/site-packages/')
import trimesh

argv = sys.argv
argv = argv[argv.index("--args") + 1:] 

# camera views
PHIS = [-0.08726646259971649, 0.34906585039886584]
THETAS = [ 4.71238898038469, 1.5707963267948966]

def enable_gpus():
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = list(cycles_preferences.devices)[:2]

    activated_gpus = []

    for device in devices:
        device.use = True
        activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = 'OPTIX'
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.use_persistent_data = True

    return activated_gpus

def create_folder(folder):
    try:
        os.mkdir(folder)
    except OSError as error:
        pass

def create_hair_material(name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    
    mat.node_tree.links.clear()
    mat.node_tree.nodes.clear()

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    output = nodes.new(type='ShaderNodeOutputMaterial')
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    links.new(shader.outputs[0], output.inputs[0])
    
    return bpy.data.materials[name]
    
    
def create_hair(name: str, block_id: int):

    def create_points(curveData, coords, index=0):
        polyline = curveData.splines.new('POLY')
        polyline.points.add(len(coords)-1)
        for i, coord in enumerate(coords):
            x, y, z = coord
            polyline.points[index].co = (x, y, z, i)
            index += 1

    hair_sample = hair[block_id * (n_strands // hair_blocks): (block_id + 1) * (n_strands // hair_blocks)]
    
    curveData = bpy.data.curves.new('hair', type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 1

    for i in range(len(hair_sample)):
        index = 0
        create_points(curveData, hair_sample[i], index=index)

    curveOB = bpy.data.objects.new(name, curveData)
    bpy.data.scenes[0].collection.objects.link(curveOB)
    
    return bpy.data.objects[name]


if __name__ == '__main__':
    
    enable_gpus()

    fol = argv[0]
    pc = sorted(os.listdir(fol))
    pc = [i for i in pc if i.split('.')[-1]=='ply']

    for m in range(len(pc)):
        
        temp_folder = argv[1] + f'{m:05d}_temp'
        output_folder_ds = argv[1] + f'{m:05d}'
    
        strands = np.array(trimesh.load(os.path.join(fol, pc[m])).vertices)

        hair_blocks = 1
        hairs = strands.reshape(-1, 100, 3)


        hair = np.zeros((hairs.shape[0], 100, 3))

        hair[:, :, 0] = hairs[:, :, 0]
        hair[:, :, 1] = -hairs[:, :, 2]
        hair[:, :, 2] = hairs[:, :, 1]

        n_strands = len(hair)

        hair_objects = create_hair('Hair', 0)

        # Scene config
        center = (0.0, 0.0, 1.6)
        radius = 1.2

        n_flights = 4
        n_frames_per_flight = 4

        min_elevation = - np.pi / 6
        max_elevation = np.pi / 4

        bpy.data.objects['Target'].location = center


        # Rendering config
        resolution = (int(argv[2]), int(argv[2]))
        z_near = 0.1
        z_far = 100.0
        samples = int(argv[4])

        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = resolution

        bpy.data.objects['Camera'].data.clip_start = z_near
        bpy.data.objects['Camera'].data.clip_end = z_far


        # Shader config
        melanin_value = 0.8


        mat = create_hair_material("Hair")

        bpy.data.objects['Hair'].data.extrude =  float(argv[3])


        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 0.0141, 0.8, 1)
        mat.node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.0

        bpy.context.object.scale[2] = 1.

        bpy.data.collections['Collection'].objects.link(bpy.data.objects["Hair"])

        # Lights config
        power = 0.6
        light_sphere_radius = radius

        bpy.data.materials["Emission Material"].node_tree.nodes["Emission"].inputs[1].default_value = power
        bpy.data.objects["Light Sphere"].location = center
        bpy.data.objects["Light Sphere"].scale = (light_sphere_radius, ) * 3


        create_folder(temp_folder)

        i = 0

        for i in range(len(PHIS)):
            phi = PHIS[i]
            theta = THETAS[i]
            bpy.data.objects['Camera'].location = (
                center[0] + radius * np.cos(theta) * np.cos(phi),
                center[1] + radius * np.sin(theta) * np.cos(phi),
                center[2] + radius * np.sin(phi)
            )

            bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = os.path.join(temp_folder, str(i))
            bpy.ops.render.render(write_still=True)

            i += 1

        create_folder(output_folder_ds)
        create_folder(os.path.join(output_folder_ds, 'image'))

        # Dataset format
        projection = []
        for i in range(len(PHIS)):
            shutil.copyfile(
                os.path.join(temp_folder, str(i), 'image0001.png'),
                os.path.join(output_folder_ds, 'image', f'img_{i:04d}.png'))

            shutil.rmtree(os.path.join(temp_folder, str(i))) 
            
        object_to_delete = bpy.data.objects['Hair']
        bpy.data.objects.remove(object_to_delete, do_unlink=True)
