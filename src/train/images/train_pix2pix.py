import os
import argparse
import numpy as np
import re
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

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
        
        valid_exts = ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']
        self.filenames = sorted([f.name for f in self.mask_dir.iterdir() if f.suffix.lower() in valid_exts])
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
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
    
    best_sample_dir = os.path.join(args.output_dir, 'best_samples')
    os.makedirs(best_sample_dir, exist_ok=True)
    
    # 체크포인트 샘플 저장용 폴더 (선택사항, 여기서는 output_dir에 바로 저장)
    # ckpt_sample_dir = os.path.join(args.output_dir, 'checkpoint_samples')
    # os.makedirs(ckpt_sample_dir, exist_ok=True)

    # 데이터셋 로드
    train_dataset = PairedDataset(args.data_dir, split='train', size=args.size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    try:
        val_dataset = PairedDataset(args.data_dir, split='val', size=args.size)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        print(f"Pix2Pix (512x512) Training Start")
        print(f" - Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    except Exception as e:
        print(f"Warning: Validation dataset not found. {e}")
        val_loader = None
        print(f"Pix2Pix (512x512) Training Start")
        print(f" - Train: {len(train_dataset)} images")

    # --- 모델 초기화 ---
    netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=9, ngf=64).to(device)
    netD = NLayerDiscriminator(input_nc=6, ndf=64, n_layers=3).to(device)

    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    criterionGAN = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # --- Resume ---
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading Generator checkpoint: {args.resume}")
            netG.load_state_dict(torch.load(args.resume, map_location=device))
            
            match = re.search(r'epoch_(\d+)', os.path.basename(args.resume))
            if match:
                start_epoch = int(match.group(1))
                print(f" -> Resuming from epoch {start_epoch + 1}")
            
            resume_d = args.resume.replace('netG', 'netD')
            if os.path.isfile(resume_d):
                print(f"Loading Discriminator checkpoint: {resume_d}")
                netD.load_state_dict(torch.load(resume_d, map_location=device))
            else:
                print("Warning: Discriminator checkpoint not found. Starting D from scratch.")
        else:
            print(f"Checkpoint file not found: {args.resume}")

    print_freq = 10
    best_psnr = 0.0 

    for epoch in range(start_epoch, args.epochs):
        netG.train()
        netD.train()
        
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Start")
        
        for i, batch in enumerate(train_loader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # Update D
            optimizer_D.zero_grad()
            fake_B = netG(real_A)

            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))

            fake_AB = torch.cat((real_A, fake_B.detach()), 1)
            pred_fake = netD(fake_AB)
            loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # Update G
            optimizer_G.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = netD(fake_AB)
            
            loss_G_GAN = criterionGAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterionL1(fake_B, real_B) * args.lambda_L1

            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            if (i + 1) % print_freq == 0:
                print(f"  [Epoch {epoch+1}][Batch {i+1}/{len(train_loader)}] "
                      f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

        # --- Validation & Best Save ---
        if val_loader and (epoch + 1) % args.val_interval == 0:
            print(f"  Running Validation...")
            
            avg_psnr, avg_ssim, avg_lpips, vis_img = evaluate(netG, val_loader, loss_fn_lpips, device)
            
            print(f"  [Val Epoch {epoch+1}] PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")

            if avg_psnr > best_psnr:
                print(f"  ★ New Best PSNR! ({best_psnr:.2f} -> {avg_psnr:.2f}) Saving Model & Sample...")
                best_psnr = avg_psnr
                
                torch.save(netG.state_dict(), os.path.join(args.output_dir, 'best_pix2pix_netG.pth'))
                torch.save(netD.state_dict(), os.path.join(args.output_dir, 'best_pix2pix_netD.pth'))
                save_image(vis_img, os.path.join(best_sample_dir, f'best_psnr{best_psnr:.2f}_epoch{epoch+1}.png'))

        # --- 주기적 저장 (save_interval) & 샘플 생성 ---
        if (epoch + 1) % args.save_interval == 0:
            # 1. 모델 저장
            save_path_G = os.path.join(args.output_dir, f'netG_epoch_{epoch+1}.pth')
            save_path_D = os.path.join(args.output_dir, f'netD_epoch_{epoch+1}.pth')
            torch.save(netG.state_dict(), save_path_G)
            torch.save(netD.state_dict(), save_path_D)
            
            # 2. 체크포인트 샘플 이미지 생성 및 저장 (추가된 부분)
            netG.eval()
            with torch.no_grad():
                # Val 셋이 있으면 Val에서, 없으면 Train에서 하나 가져오기
                loader_for_sample = val_loader if val_loader else train_loader
                sample_batch = next(iter(loader_for_sample))
                
                real_A = sample_batch['A'].to(device)
                real_B = sample_batch['B'].to(device)
                fake_B = netG(real_A)
                
                # [Mask | Ground Truth | Generated]
                vis = torch.cat((real_A, real_B, fake_B), dim=3)
                vis = (vis + 1) / 2.0 # Denormalize
                
                sample_path = os.path.join(args.output_dir, f'sample_epoch_{epoch+1}.png')
                save_image(vis, sample_path)
            
            netG.train() # 다시 학습 모드로 복귀
            print(f"  Checkpoint & Sample saved: {save_path_G}")

@torch.no_grad()
def evaluate(netG, val_loader, loss_fn_lpips, device):
    netG.eval()
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    first_vis_img = None
    
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
            first_vis_img = (vis + 1) / 2.0
            
    n = len(val_loader)
    return total_psnr/n, total_ssim/n, total_lpips/n, first_vis_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_pix2pix_512')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--lambda_L1', type=float, default=100.0)
    
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    train(args)
