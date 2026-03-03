import os
import argparse
import numpy as np
from tqdm import tqdm
import lpips
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision import transforms
import kornia

from network import VolumeTexture


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, default="xxx")
parser.add_argument('--pointmap_root', type=str, default="xxx")
parser.add_argument('--save_root', type=str, default="xxx")
parser.add_argument('--mask_root', type=str, default="xxx")
parser.add_argument('--vis_freq', type=int, default=25)
opt = parser.parse_args()


class PointMapDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.img_root = opt.img_root
        self.pointmap_root = opt.pointmap_root
        self.mask_root = opt.mask_root
        self.images, self.pointmaps, self.masks = [], [], []
        for pth in sorted(os.listdir(self.img_root)):
            img = transforms.ToTensor()(
                Image.open(os.path.join(self.img_root, pth))
            )
            self.images.append(img)
            mask = transforms.ToTensor()(
                Image.open(os.path.join(self.mask_root, pth))
            )
            self.masks.append(mask)
            img_name = os.path.splitext(pth)[0]
            self.pointmaps.append(
                torch.load(
                    os.path.join(self.pointmap_root, "%s.pkl" % img_name),
                    map_location="cpu",
                )
            )
        
        self.images = torch.stack(self.images, dim=0).cuda()
        self.pointmaps = torch.cat(self.pointmaps, dim=0).cuda()
        self.masks = torch.stack(self.masks, dim=0).cuda()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            "img": self.images[idx],
            "pointmap": self.pointmaps[idx],
            "mask": self.masks[idx],
        }


class TextureOptimizer:
    def __init__(self):
        self.network = VolumeTexture()
        self.network = self.network.cuda()
        self.dataset = PointMapDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        self.val_dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.lpips_loss = lpips.LPIPS(net="vgg").cuda()
        self.lpips_loss.requires_grad_(False)
    
    def optimize(self):
        params = [{"params": self.network.parameters()}]
        lr = 0.01
        optimizer = torch.optim.Adam(params=params, lr=lr)
        grad_scaler = torch.cuda.amp.GradScaler(2 ** 10)
        self.log_dir = opt.save_root
        os.makedirs(self.log_dir, exist_ok=True)
        writer = SummaryWriter(self.log_dir)
        
        for i in tqdm(range(151)):
            for data in self.dataloader:
                # forward
                point = data["pointmap"]  # [b,3,h,w]
                b, _, h, w = point.shape
                point = point.permute(0, 2, 3, 1).reshape(b * h * w, 3)
                point = (point + 1) / 2
                render = self.network(point)
                pred = render.reshape(b, h, w, 3).permute(0, 3, 1, 2).contiguous()
                
                mask = data["mask"]
                gt = data["img"]

                img_pred = pred * mask
                img_gt = gt * mask
                
                
                loss = F.l1_loss(img_pred, img_gt)
                writer.add_scalar('L1 Loss', loss.item(), i)
                if i > 100:
                    loss_lpips = self.lpips_loss(img_pred, img_gt, normalize=True).mean()
                    loss = loss + 0.1 * loss_lpips
                    writer.add_scalar('LPIPS Loss', loss.item(), i)

                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            if i % opt.vis_freq == 0:
                torch.save(
                    self.network.state_dict(), os.path.join(self.log_dir, "latest.pth")
                )
                cur_log_dir = os.path.join(self.log_dir, "%05d" % i)
                os.makedirs(cur_log_dir, exist_ok=True)
                with torch.no_grad():
                    render_list = []
                    gt_list = []
                    psnr_list = []
                    ssim_list = []
                    lpips_list = []
                    img_name_list = []
                    for data in self.val_dataloader:
                        # forward
                        point = data["pointmap"]  # [b,3,h,w]
                        b, _, h, w = point.shape
                        point = point.permute(0, 2, 3, 1).reshape(b * h * w, 3)
                        point = (point + 1) / 2
                        render = self.network(point)
                        pred = render.reshape(b, h, w, 3).permute(0, 3, 1, 2).contiguous()
                        
                        mask = data["mask"]
                        gt = data["img"]
                        
                        img_pred = pred * mask
                        img_gt = gt * mask
                        gt_list.append(img_gt)
                        render_list.append(img_pred)
                        cur_psnr = kornia.metrics.psnr(img_pred, img_gt, max_val=1.).item()
                        cur_ssim = kornia.metrics.ssim(img_pred, img_gt, window_size=3).mean().item()
                        cur_lpips = self.lpips_loss(img_pred, img_gt, normalize=True).mean()
                        psnr_list.append(cur_psnr)
                        ssim_list.append(cur_ssim)
                        lpips_list.append(cur_lpips)
                    
                    render_list = torch.cat(render_list, dim=0)
                    gt_list = torch.cat(gt_list, dim=0)
                    save_image(render_list, os.path.join(cur_log_dir, "render.jpg"))
                    save_image(gt_list, os.path.join(cur_log_dir, "gt.jpg"))

                    psnr_score = sum(psnr_list) / len(psnr_list)
                    ssim_score = sum(ssim_list) / len(ssim_list)
                    lpips_score = sum(lpips_list) / len(lpips_list)
                    writer.add_scalar('ssim', ssim_score, i)
                    writer.add_scalar('psnr', psnr_score, i)
                    writer.add_scalar('lpips', lpips_score, i)
                    print("[iter %05d][PSNR %.4f][SSIM %.4f][LPIPS %.4f]" % (
                        i, psnr_score, ssim_score, lpips_score,
                    ))


if __name__ == "__main__":
    optimizer = TextureOptimizer()
    optimizer.optimize()
