#  Copyright (c) 2023 Max Planck Society

import os
import os.path as osp

import bpy
import numpy as np
import trimesh

import argparse


def clean_scene():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='CURVE')
    bpy.ops.object.delete()


def create_hair(name: str, strands: np.ndarray, hair_blocks: int, block_id: int):
    def create_points(curveData, coords, index=0):
        polyline = curveData.splines.new('POLY')
        polyline.points.add(len(coords) - 1)
        for i, coord in enumerate(coords):
            x, y, z = coord
            polyline.points[index].co = (x, y, z, i)
            index += 1

    n_strands = strands.shape[0]

    hair_sample = strands[block_id * (n_strands // hair_blocks): (block_id + 1) * (n_strands // hair_blocks)]

    curveData = bpy.data.curves.new('hair', type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 1

    for i in range(len(hair_sample)):
        index = 0
        create_points(curveData, hair_sample[i], index=index)

    curveOB = bpy.data.objects.new(name, curveData)
    bpy.data.scenes[0].collection.objects.link(curveOB)

    return bpy.data.objects[name]


def import_hair_from_ply(ply_fname, n_strands_target):
    strands = np.array(trimesh.load(ply_fname).vertices)
    print(strands.shape)

    strands = strands.reshape(-1, 100, 3)

    n_strands, n_vertices = strands.shape[0], strands.shape[1]

    if n_strands > n_strands_target:
        sampled_idx = np.random.choice(strands.shape[0], n_strands_target, replace=False)
        strands = strands[sampled_idx, ...]

    strands = strands[:, :, [0, 2, 1]]
    strands[:, :, 1] *= -1

    hair_objects = [create_hair(f'Hair', strands, hair_blocks=1, block_id=0)]
    return hair_objects


def export_hair_object_as_alembic_for_unreal(hair_object_name, out_fname):
    new_active_object = bpy.data.objects[hair_object_name]

    bpy.context.view_layer.objects.active = new_active_object

    obj = bpy.context.active_object

    attr_name = 'groom_group_id'
    obj[attr_name] = 0
    obj[attr_name + '_AbcGeomScope'] = 'con'

    os.makedirs(osp.dirname(out_fname), exist_ok=True)
    bpy.ops.wm.alembic_export(filepath=out_fname, start=1, end=1, export_hair=False, export_particles=False,
                              orcos=False, xsamples=1, gsamples=1, sh_open=0.0, sh_close=1.0, selected=False,
                              visible_objects_only=False, flatten=False, uvs=False, packuv=False, normals=False,
                              vcolors=False, face_sets=False, subdiv_schema=False, apply_subdiv=False,
                              curves_as_mesh=False, use_instancing=True, global_scale=1.0, triangulate=False,
                              quad_method='SHORTEST_DIAGONAL', ngon_method='BEAUTY', export_custom_properties=True,
                              as_background_job=False, evaluation_mode='RENDER', init_scene_frame_range=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_fname', type=str, required=True)
    parser.add_argument('--out_abc_fname', type=str, required=True)
    parser.add_argument('--n_strands_max', type=int, default=120000, required=False)
    args = parser.parse_args()

    clean_scene()
    import_hair_from_ply(args.ply_fname, n_strands_target=args.n_strands_max)
    export_hair_object_as_alembic_for_unreal(hair_object_name='Hair', out_fname=args.out_abc_fname)


if __name__ == '__main__':
    main()
