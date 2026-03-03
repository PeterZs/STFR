import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="xxx")
args = parser.parse_args()
data_root = args.data_root


img_root = os.path.join(data_root, "images")
mask_root = os.path.join(data_root, "mask")
database_root = os.path.join(data_root, "database.db")
sparse_root = os.path.join(data_root, "sparse")
recon_root = os.path.join(sparse_root, "0")
camera_model="PINHOLE"


feat_extract = """
    colmap feature_extractor \
    --database_path %s \
    --image_path %s \
    --ImageReader.camera_model %s \
    --ImageReader.single_camera 1 \
    --SiftExtraction.max_image_size 4000 \
    --FeatureExtraction.use_gpu 1 \
    --ImageReader.mask_path %s
"""

feat_match = """
    colmap exhaustive_matcher \
    --database_path %s \
    --FeatureMatching.guided_matching 1 \
    --FeatureMatching.use_gpu 1
"""

sfm = """
    colmap mapper \
    --database_path %s \
    --image_path %s \
    --output_path %s
"""

ba = """
    colmap bundle_adjuster \
    --input_path %s \
    --output_path %s \
    --BundleAdjustment.max_num_iterations 100
"""

to_txt = """
    colmap model_converter \
    --input_path %s \
    --output_path %s \
    --output_type TXT
"""

os.system(feat_extract % (
    database_root, img_root, camera_model, mask_root
))

os.system(feat_match % (
    database_root,
))

os.makedirs(sparse_root, exist_ok=True)
os.system(sfm % (
    database_root, img_root, sparse_root
))

os.system(ba % (
    recon_root, recon_root
))

os.system(to_txt % (
    recon_root, recon_root
))
