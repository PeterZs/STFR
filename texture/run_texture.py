import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=None)
opt = parser.parse_args()


os.system("python render_gbuffer.py --data_root %s" % opt.data_root)

os.system("python build_texture.py --data_root %s" % opt.data_root)
