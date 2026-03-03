import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default="xxx")
opt = parser.parse_args()

img_root = os.path.join(opt.data_root, "raw_frames")
mask_root = os.path.join(opt.data_root, "mask")
save_root = os.path.join(opt.data_root, "images")
os.makedirs(save_root, exist_ok=True)

for pth in tqdm(sorted(os.listdir(img_root))):
    img = transforms.ToTensor()(Image.open(os.path.join(img_root, pth)))
    mask = transforms.ToTensor()(Image.open(os.path.join(mask_root, pth)))[:1]
    comp = torch.cat([img, mask], dim=0)
    save_image(comp, os.path.join(save_root, pth))
