import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="xxx")
opt = parser.parse_args()


opt.data_root = os.path.abspath(opt.data_root)

# convert to 2DGS format
if True:
    os.system(
        "python to_2dgs_format.py --data_root %s" % (
            opt.data_root
        )
    )


# calibrate camera parameters using colmap
if True:
    os.system(
        "python run_colmap.py --data_root %s" % (
            opt.data_root
        )
    )


# run 2dgs
if True:
    code_root = os.path.dirname(os.path.abspath(__file__))
    gs_code_root = os.path.join(code_root, "2d-gaussian-splatting")
    os.chdir(gs_code_root)

    recon_root = os.path.join(opt.data_root, "recon")
    os.system(
        "python train.py -s %s -m %s" % (
            opt.data_root, recon_root,
        )
    )

    os.system(
        "python render.py -s %s -m %s --mesh_res 1024" % (
            opt.data_root, recon_root,
        )
    )

    os.chdir(code_root)


# to my format
if True:
    os.system(
        "python to_my_format.py --data_root %s" % (
            opt.data_root
        )
    )
