import os
import argparse
import torch
import trimesh
import numpy as np

from network import VolumeTexture


argparser = argparse.ArgumentParser()
argparser.add_argument("--mesh_path", type=str, default="xxx")
argparser.add_argument("--save_path", type=str, default="xxx")
argparser.add_argument("--ckpt_path", type=str, default="xxx")
opt = argparser.parse_args()


def load_geometry(mesh_uv_path):
    mesh = trimesh.load_mesh(mesh_uv_path)
    vertices = torch.from_numpy(mesh.vertices).cuda().float()  # [v,3]

    # fit vertices in [0,1] bounding box
    offset = torch.mean(vertices, dim=0, keepdim=False)  # [3]
    can_vertices = vertices - offset
    scale = torch.max(can_vertices, dim=0)[0] - torch.min(can_vertices, dim=0)[0]
    scale = torch.max(scale)
    scale = 2 / scale
    
    can_vertices = can_vertices * scale
    return can_vertices, mesh


network = VolumeTexture().cuda()
network.load_state_dict(
    torch.load(opt.ckpt_path)
)

can_vertices, mesh = load_geometry(opt.mesh_path)
can_vertices = (can_vertices + 1) / 2

network = network.eval()

with torch.no_grad():
    color = network(can_vertices).cpu().numpy()  # [v,3]
    color = np.clip(color, 0, 1)

trimesh.Trimesh(
    vertices=mesh.vertices,
    faces=mesh.faces,
    vertex_colors=(color * 255).astype(np.uint8),
).export(opt.save_path)
