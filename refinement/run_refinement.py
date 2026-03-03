import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str)
args = parser.parse_args()


# setup path
args.img_root = os.path.join(args.data_root, "raw_frames")
args.mesh_path = os.path.join(args.data_root, "2dgs_recon.obj")
args.cam_path = os.path.join(args.data_root, "transforms.json")
args.save_root = os.path.join(args.data_root, "refinement")


os.makedirs(args.save_root, exist_ok=True)

sample_data_root = os.path.join(args.save_root, "sample")
tex_log_root = os.path.join(args.save_root, "texture")

os.makedirs(sample_data_root, exist_ok=True)


# sample 16 sharp frames from all the raw frames
# these sampled frames are used to refine the texture
if True:
    cmd = """
        python select_frame/compute_sharpness.py \
        --img_root %s \
        --save_root %s
    """
    os.system(
        cmd % (
            args.img_root, sample_data_root
        )
    )
    
    cmd = """
        python select_frame/sample_by_sharpness.py \
        --img_root %s \
        --cam_path %s \
        --save_root %s \
        --num_view 16
    """
    os.system(
        cmd % (
            args.img_root, args.cam_path, sample_data_root
        )
    )


# render screen-space position maps
if True:    
    cmd = """
        python render_position_map.py \
        --mesh_path %s \
        --cam_path %s \
        --img_root %s \
        --save_root %s
    """
    
    os.system(
        cmd % (
            args.mesh_path, 
            os.path.join(sample_data_root, "select_sharp.json"), 
            args.img_root, 
            sample_data_root,
        )
    )


# optimize a 3D texture volume
if True:
    cmd = """
        python build_texture.py \
        --img_root %s \
        --pointmap_root %s \
        --mask_root %s \
        --save_root %s \
    """

    os.system(
        cmd % (
            os.path.join(sample_data_root, "image"),
            os.path.join(sample_data_root, "pointmap"),
            os.path.join(sample_data_root, "pointmap_mask"),
            tex_log_root,
        )
    )


# save the textured mesh after refinement
if True:
    cmd = """
        python add_texture_to_mesh.py \
        --mesh_path %s \
        --save_path %s \
        --ckpt_path %s
    """
    
    os.system(
        cmd % (
            args.mesh_path, 
            os.path.join(args.save_root, "textured_mesh.obj"),
            os.path.join(tex_log_root, "latest.pth"),
        )
    )
