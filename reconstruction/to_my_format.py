import os
import argparse
import json
import trimesh
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="xxx")
opt = parser.parse_args()


mesh_path = os.path.join(opt.data_root, "recon/train/ours_30000/fuse_post.ply")
mesh = trimesh.load(mesh_path)
split_all = mesh.split(only_watertight=False)
mesh = sorted(split_all, key=lambda x: len(x.faces))[-1]
mesh.export(os.path.join(opt.data_root, "2dgs_recon.obj"))

cam_path = os.path.join(opt.data_root, "recon/cameras.json")

with open(cam_path, 'r') as f:
    cam_info_list = json.load(f)

save_info = {
    "fl_x": cam_info_list[0]["fx"],
    "fl_y": cam_info_list[0]["fy"],
    "w": cam_info_list[0]["width"],
    "h": cam_info_list[0]["height"],
}
save_info["cx"] = save_info["w"] / 2
save_info["cy"] = save_info["h"] / 2
frames = []

for cam_info in cam_info_list:
    trans = cam_info["position"]
    rot = cam_info["rotation"]

    cur_frame = {
        "file_path": os.path.join(opt.data_root, "raw_frames", "%s.png" % cam_info["img_name"])
    }
    transforms = np.eye(4)
    transforms[:3, :3] = rot
    transforms[:3, 3] = trans
    cur_frame["transform_matrix"] = transforms.tolist()

    frames.append(cur_frame)

save_info["frames"] = frames
with open(os.path.join(opt.data_root, "transforms.json"), 'w') as f:
    json.dump(save_info, f, indent=4)
