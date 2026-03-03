import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--ldm_type', type=str, default="eyelid", choices=["ibug", "eyelid", "interp"])
parser.add_argument('--close_eye', type=int, default=1)
opt = parser.parse_args()


opt.img_root = os.path.join(opt.data_root, "raw_frames")
opt.mesh_path = os.path.join(opt.data_root, "refinement", "textured_mesh.obj")
opt.cam_path = os.path.join(opt.data_root, "transforms.json")
opt.save_root = os.path.join(opt.data_root, "register")

code_root = os.path.dirname(os.path.abspath(__file__))
fit_code_root = os.path.join(code_root, "align/AlbedoMMFitting")
wrap_root = os.path.join(code_root, "wrap")

coarse_save_root = os.path.join(opt.save_root, "coarse_align")
fine_save_root = os.path.join(opt.save_root, "fine_align")
wrap_save_root = os.path.join(opt.save_root, "wrap")


# coarse alignment
if True:
    cmd = """
        python align/align_coarse.py \
        --img_root %s \
        --mesh_path %s \
        --cam_path %s \
        --save_root %s
    """

    os.system(cmd % (
        opt.img_root,
        opt.mesh_path,
        opt.cam_path,
        coarse_save_root,
    ))


# fine alignment
if True:
    os.chdir(fit_code_root)

    cmd = """
        python fitting.py \
        --img_root %s \
        --coarse_fitting_root %s \
        --save_root %s
    """

    os.system(cmd % (
        opt.img_root,
        coarse_save_root,
        fine_save_root,
    ))

    cmd = """
        python to_canonical.py \
        --coarse_fitting_root %s \
        --save_root %s
    """
    os.system(cmd % (
        coarse_save_root,
        fine_save_root,
    ))

    os.chdir(code_root)


# build correspondences
if True:
    cmd = """
        python to_wrap.py \
        --data_root %s \
        --save_root %s \
        --ldm_type %s \
        --close_eye %d
    """
    os.system(cmd % (
        fine_save_root,
        wrap_save_root,
        opt.ldm_type,
        opt.close_eye
    ))


# run wrap for registration
if True:
    os.chdir(wrap_root)
    cmd = """
        xvfb-run --auto-servernum ./WrapCmd compute %s
    """
    os.system(cmd % (os.path.join(wrap_save_root, "wrap.wrap")))

    os.chdir(code_root)
    os.system("python build_dataset.py --data_root %s" % opt.data_root)
