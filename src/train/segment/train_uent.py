from __future__ import annotations
"""
UNet training (segmentation) — Refactored
- CLI 옵션으로 경로/하이퍼파라미터 제어
- 모든 이미지 확장자 지원 + 파일명(stem) 기준 이미지/마스크 매칭(*_mask 허용)
- tqdm 제거 → print 기반 진행 로그(에포크 내 20% 간격)
- 체크포인트/샘플 주기: --save N (N 에포크마다 저장)
- 재개: --resume (최신), --resume-from (특정 에포크/경로)

Usage
-----
python train_unet.py \
  -i ./data/processed/train/images \
  -m ./data/processed/train/masks \
  -v ./data/processed/val/images \
  -w ./data/processed/val/masks \
  -o ./result/unet_run1 \
  --epochs 300 --batchsize 8 --numworker 8 --lr 1e-4 --image-size 512 \
  --save 25 --resume
"""

import os
import re
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models import UNet

# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train U-Net for binary segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--train-images', type=Path, required=True, help='학습 이미지 디렉터리')
    p.add_argument('-m', '--train-masks', type=Path, required=True, help='학습 마스크 디렉터리')
    p.add_argument('-v', '--val-images', type=Path, required=True, help='검증 이미지 디렉터리')
    p.add_argument('-w', '--val-masks', type=Path, required=True, help='검증 마스크 디렉터리')
    p.add_argument('-o', '--output-dir', type=Path, required=True, help='출력 루트(models/, samples/)')
    p.add_argument('--epochs', type=int, default=100, help='학습 epoch 수')
    p.add_argument('--batchsize', type=int, default=4, help='배치 크기')
    p.add_argument('--numworker', type=int, default=4, help='DataLoader num_workers')
    p.add_argument('--lr', type=float, default=1e-4, help='학습률')
    p.add_argument('--image-size', type=int, default=512, help='입력 리사이즈(정방형)')
    p.add_argument('--save', type=int, default=50, help='N 에포크마다 체크포인트 저장')
    p.add_argument('--resume', action='store_true', help='models/ 에서 최신 체크포인트 재개')
    p.add_argument('--resume-from', type=str, default='', help='특정 체크포인트 재개(번호 또는 경로)')
    p.add_argument('--gpu', type=int, default=0, help='사용할 GPU 인덱스(0,1,2). -1이면 CPU')
    return p.parse_args()

def pick_device(gpu_idx: int) -> torch.device:
    if gpu_idx < 0 or not torch.cuda.is_available():
        print("[info] Using CPU")
        return torch.device("cpu")
    num = torch.cuda.device_count()
    if gpu_idx >= num:
        print(f"[warn] GPU index {gpu_idx} out of range (0~{num-1}). Fallback to cuda:0")
        gpu_idx = 0
    torch.cuda.set_device(gpu_idx)  # 선택한 카드 활성화
    print(f"[info] Using GPU: cuda:{gpu_idx}")
    return torch.device(f"cuda:{gpu_idx}")


# ------------------------------
# Dataset & pairing
# ------------------------------
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}

def _list_images(d: Path) -> List[Path]:
    return sorted([p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _build_mask_index(mask_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in _list_images(mask_dir):
        stem = p.stem
        idx[stem] = p
        if stem.endswith('_mask'):
            idx[stem[:-5]] = p
    return idx


class SegDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, image_size: int):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.images = _list_images(self.img_dir)
        self.mask_index = _build_mask_index(self.mask_dir)

        # transform: grayscale to 1ch tensor for both
        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0,1]
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        pairs: List[Tuple[Path, Path]] = []
        skipped = 0
        for im in self.images:
            m = self.mask_index.get(im.stem) or self.mask_index.get(im.stem + '_mask')
            if m is not None:
                pairs.append((im, m))
            else:
                skipped += 1
        self.pairs = pairs
        if skipped:
            print(f"[dataset] masks not found for {skipped} images -> skipped")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        # convert to single channel tensors
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        img_t = self.img_tf(img)  # (1,H,W) float32 0..1
        mask_t = self.mask_tf(mask)
        # binarize mask (assume foreground=255 or >0.5)
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t


# ------------------------------
# Loss & metrics
# ------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, inputs, targets, smooth: float = 1.0):
        bce = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)
        inter = (probs * targets).sum()
        dice = 1 - (2 * inter + smooth) / (probs.sum() + targets.sum() + smooth)
        return bce + dice


def evaluate(loader: DataLoader, model: nn.Module, device: str) -> Tuple[float, float]:
    model.eval()
    dice_sum = 0.0
    iou_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            inter = (preds * y).sum()
            union = preds.sum() + y.sum()
            dice = (2 * inter) / (union + 1e-8)
            iou = inter / (union - inter + 1e-8)
            dice_sum += dice.item()
            iou_sum += iou.item()
    n = max(1, len(loader))
    model.train()
    return dice_sum / n, iou_sum / n


# ------------------------------
# Checkpoint helpers
# ------------------------------

def _last_epoch(models_dir: Path) -> Optional[int]:
    if not models_dir.exists():
        return None
    ep = -1
    pat = re.compile(r"_(?:epoch_)(\d+)(?:\.pth)?$")
    for p in models_dir.iterdir():
        m = pat.search(p.name)
        if m:
            try:
                ep = max(ep, int(m.group(1)))
            except ValueError:
                pass
    return ep if ep >= 0 else None


def _resolve_resume(models_dir: Path, resume_from: str) -> Tuple[Optional[Path], Optional[int]]:
    if not resume_from:
        return models_dir, None
    if resume_from.isdigit():
        return models_dir, int(resume_from)
    p = Path(resume_from)
    if not p.exists():
        print(f"[resume] 경로가 없습니다: {p}")
        return None, None
    m = re.search(r'epoch_(\d+)', p.name)
    if m:
        return p.parent, int(m.group(1))
    if p.is_dir():
        ep = _last_epoch(p)
        return (p, ep) if ep is not None else (None, None)
    return None, None


# ------------------------------
# Train loop
# ------------------------------

def main():
    args = parse_args()
    device = pick_device(args.gpu)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    out_root: Path = args.output_dir
    models_dir = out_root / 'models'
    samples_dir = out_root / 'samples'
    models_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # data
    train_ds = SegDataset(args.train_images, args.train_masks, args.image_size)
    val_ds = SegDataset(args.val_images, args.val_masks, args.image_size)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("No valid pairs in train/val datasets.")
    train_loader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True,
                              num_workers=args.numworker, pin_memory=(device=='cuda'),
                              persistent_workers=(args.numworker>0))
    val_loader = DataLoader(val_ds, batch_size=args.batchsize, shuffle=False,
                            num_workers=args.numworker, pin_memory=(device=='cuda'),
                            persistent_workers=(args.numworker>0))

    # model + opt
    model = UNet(n_channels=1, n_classes=1).to(device)
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # resume
    start_epoch = 0
    if args.resume_from:
        mdir, ep = _resolve_resume(models_dir, args.resume_from)
        if mdir and ep is not None:
            ckpt = mdir / f'unet_epoch_{ep}.pth'
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=device))
                start_epoch = ep
                print(f"[resume-from] loaded epoch {ep}")
            else:
                print(f"[resume-from] 체크포인트가 없습니다: {ckpt}")
        else:
            print('[resume-from] 유효한 체크포인트를 찾지 못했습니다.')
    elif args.resume:
        last = _last_epoch(models_dir)
        if last is not None:
            ckpt = models_dir / f'unet_epoch_{last}.pth'
            try:
                model.load_state_dict(torch.load(ckpt, map_location=device))
                start_epoch = last
                print(f"[resume] loaded epoch {last}")
            except Exception as e:
                print(f"[resume] load failed: {e}")
        else:
            print('[resume] no checkpoints found.')

    # train
    steps_per_epoch = math.ceil(len(train_ds) / args.batchsize)
    best_dice = -1.0
    save_every = max(1, int(args.save))

    print(f"[start] train={len(train_ds)} val={len(val_ds)} batch={args.batchsize} epochs={args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        if device == 'cuda':
            torch.cuda.empty_cache()
        model.train()
        total_loss = 0.0
        print(f"\n[epoch {epoch+1}/{args.epochs}] start")

        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % max(1, steps_per_epoch // 5) == 0 or step == steps_per_epoch:
                pct = int(step / steps_per_epoch * 100)
                print(f"  - {step}/{steps_per_epoch} ({pct}%) loss={loss.item():.4f}")

        avg_loss = total_loss / steps_per_epoch
        dice, iou = evaluate(val_loader, model, device)
        print(f"[epoch {epoch+1}] done | TrainLoss={avg_loss:.4f} ValDice={dice:.4f} ValIoU={iou:.4f}")

        # best model
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), models_dir / 'unet_best.pth')
            print("  -> saved best model: unet_best.pth")

        # periodic save
        e = epoch + 1
        if e % save_every == 0:
            torch.save(model.state_dict(), models_dir / f'unet_epoch_{e}.pth')
            print(f"  -> saved checkpoint @ epoch {e}")

    # final save
    torch.save(model.state_dict(), models_dir / 'unet_final.pth')
    print('[done] Final model saved: unet_final.pth')


if __name__ == '__main__':
    main()
