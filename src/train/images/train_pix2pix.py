import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# 평가 메트릭
import lpips
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from pix2pix_model import UnetGenerator, NLayerDiscriminator, weights_init_normal

# ------------------------------
# 1. Dataset
# ------------------------------
class PairedDataset(Dataset):
    def __init__(self, root_dir, split='train', size=512):
        self.root_dir = Path(root_dir) / split
        self.size = size
        
        self.mask_dir = self.root_dir / 'masks'
        self.image_dir = self.root_dir / 'images'
        self.filenames = sorted([f.name for f in self.mask_dir.iterdir() if f.suffix in ['.jpg', '.png', '.jpeg']])
        
        # 512x512 Resize
        self.transform = transforms.Compose([
            transforms.Resize((size, size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        mask = Image.open(self.mask_dir / fname).convert('RGB')
        image = Image.open(self.image_dir / fname).convert('RGB')
        return {'A': self.transform(mask), 'B': self.transform(image), 'path': fname}

# ------------------------------
# 2. Training Script
# ------------------------------
def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    train_dataset = PairedDataset(args.data_dir, split='train', size=args.size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_dataset = PairedDataset(args.data_dir, split='val', size=args.size)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    print(f"🚀 Pix2Pix (512x512) 학습 시작")
    print(f" - Train: {len(train_dataset)}장, Val: {len(val_dataset)}장")

    # --- 모델 초기화 ---
    # 중요: 512x512는 2^9승 이므로 num_downs=9로 설정
    netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=9, ngf=64).to(device)
    netD = NLayerDiscriminator(input_nc=6, ndf=64, n_layers=3).to(device)

    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    criterionGAN = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        netG.train()
        netD.train()
        
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in tqdm_bar:
            real_A = batch['A'].to(device) # Mask
            real_B = batch['B'].to(device) # Real

            # --- Discriminator ---
            optimizer_D.zero_grad()
            fake_B = netG(real_A)

            # Real Loss
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))

            # Fake Loss
            fake_AB = torch.cat((real_A, fake_B.detach()), 1)
            pred_fake = netD(fake_AB)
            loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --- Generator ---
            optimizer_G.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = netD(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterionL1(fake_B, real_B) * args.lambda_L1

            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            tqdm_bar.set_postfix({'D': loss_D.item(), 'G': loss_G.item()})

        # --- Validation ---
        if (epoch + 1) % args.val_interval == 0:
            evaluate(netG, val_loader, loss_fn_lpips, device, sample_dir, epoch)
            torch.save(netG.state_dict(), os.path.join(args.output_dir, f'netG_epoch_{epoch+1}.pth'))

@torch.no_grad()
def evaluate(netG, val_loader, loss_fn_lpips, device, sample_dir, epoch):
    netG.eval()
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    
    for i, batch in enumerate(val_loader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        fake_B = netG(real_A)
        
        fake_B_norm = (fake_B + 1) / 2
        real_B_norm = (real_B + 1) / 2
        
        total_psnr += psnr(fake_B_norm, real_B_norm, data_range=1.0).item()
        total_ssim += ssim(fake_B_norm, real_B_norm, data_range=1.0).item()
        total_lpips += loss_fn_lpips(fake_B, real_B).item()
        
        if i == 0:
            vis = torch.cat((real_A, real_B, fake_B), dim=3)
            vis = (vis + 1) / 2.0
            save_image(vis, os.path.join(sample_dir, f'val_epoch_{epoch+1}.png'))
            
    n = len(val_loader)
    print(f"\n📊 Epoch {epoch+1} | PSNR: {total_psnr/n:.2f} | SSIM: {total_ssim/n:.4f} | LPIPS: {total_lpips/n:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_pix2pix_512')
    parser.add_argument('--epochs', type=int, default=200)
    # 512px은 메모리를 많이 차지하므로 batch_size 1이 안전함
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--size', type=int, default=512, help="이미지 해상도")
    parser.add_argument('--lambda_L1', type=float, default=100.0)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    train(args)