import os
import argparse
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import config


# ── Multi-class Dice Loss ─────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Dice Loss untuk multi-class segmentation.
    Menghitung Dice per kelas lalu rata-rata (macro average)."""
    def __init__(self, smooth=1.0, num_classes=3):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = torch.softmax(logits.float(), dim=1)  # (B, C, H, W)
        dice_sum = 0.0
        for cls in range(self.num_classes):
            prob_cls = probs[:, cls]  # (B, H, W)
            target_cls = (targets == cls).float()  # (B, H, W)
            intersection = (prob_cls * target_cls).sum()
            dice = (2. * intersection + self.smooth) / (prob_cls.sum() + target_cls.sum() + self.smooth)
            dice_sum += dice
        return 1 - (dice_sum / self.num_classes)


class CombinedLoss(nn.Module):
    """Gabungan CrossEntropy + Dice Loss untuk 3-class building segmentation."""
    def __init__(self, class_weights, dice_weight=0.5, num_classes=3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes=num_classes)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


# ── Dataset ───────────────────────────────────────────────────────────────────
class BuildingDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, augment=False):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
        
        # Color jitter hanya untuk training (TIDAK diterapkan ke mask)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05
        ) if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = Path(row['image_path']).name
        mask_name = Path(row['mask_path']).name
        
        img_path = self.img_dir / img_name
        mask_path = self.mask_dir / mask_name
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')  # Grayscale: 0, 1, 2
        
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
                
            # Random vertical flip
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
                
            # Random rotation (90/180/270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
            
            # Random scale jitter (crop lalu resize ±10-15%)
            if random.random() > 0.5:
                w, h = img.size
                scale = random.uniform(0.85, 1.15)
                new_w, new_h = int(w * scale), int(h * scale)
                
                img = TF.resize(img, [new_h, new_w])
                mask = TF.resize(mask, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)
                
                # Center crop back ke ukuran asli
                img = TF.center_crop(img, [h, w])
                mask = TF.center_crop(mask, [h, w])
            
            # Color jitter (HANYA untuk image, BUKAN mask)
            if self.color_jitter:
                img = self.color_jitter(img)

        img_tensor = self.img_transform(img)
        
        # Mask 3-class: baca langsung nilai 0, 1, 2 (JANGAN konversi binary)
        mask_np = np.array(mask).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()

        return {"pixel_values": img_tensor, "labels": mask_tensor}


# ── IoU Calculator ────────────────────────────────────────────────────────────
def calculate_iou(preds, labels, num_classes=3):
    """Hitung IoU per kelas dan mIoU. Return (mIoU, dict per-class IoU)."""
    preds = preds.argmax(dim=1).flatten()
    labels = labels.flatten()
    
    class_names = {0: 'bg', 1: 'interior', 2: 'border'}
    ious = {}
    
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            ious[class_names[cls]] = float('nan')
        else:
            ious[class_names[cls]] = intersection / union
    
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    return miou, ious


# ── Compute class weights from dataset ────────────────────────────────────────
def compute_class_weights(csv_file, mask_dir, num_samples=200):
    """Hitung inverse-frequency class weights dari sampel dataset."""
    df = pd.read_csv(csv_file)
    counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        mask_name = Path(row['mask_path']).name
        mask_path = Path(mask_dir) / mask_name
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'))
            for cls in range(config.NUM_CLASSES):
                counts[cls] += np.sum(mask == cls)
    
    total = counts.sum()
    if total == 0:
        return torch.ones(config.NUM_CLASSES)
    
    # Inverse frequency: w_i = total / (num_classes * count_i)
    inv_freq = total / (config.NUM_CLASSES * np.maximum(counts, 1))
    # Normalize so background = 1.0
    inv_freq = inv_freq / inv_freq[0]
    
    print(f"  Class pixel counts: bg={int(counts[0]):,}, interior={int(counts[1]):,}, border={int(counts[2]):,}")
    print(f"  Computed weights:   [{inv_freq[0]:.2f}, {inv_freq[1]:.2f}, {inv_freq[2]:.2f}]")
    
    return torch.tensor(inv_freq, dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # Dataset
    train_dir = Path(config.DATASET_DIR) / "train"
    val_dir = Path(config.DATASET_DIR) / "val"
    
    train_csv = Path(config.DATASET_DIR) / "train.csv"
    val_csv = Path(config.DATASET_DIR) / "val.csv"
    
    train_dataset = BuildingDataset(
        csv_file=train_csv,
        img_dir=train_dir / "images",
        mask_dir=train_dir / "masks",
        augment=config.AUGMENT
    )
    
    val_dataset = BuildingDataset(
        csv_file=val_csv,
        img_dir=val_dir / "images",
        mask_dir=val_dir / "masks",
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume:
        print(f"Melanjutkan training dari {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_miou = checkpoint['best_miou']
        
    Path(config.MODEL_DIR).mkdir(exist_ok=True)
    log_file = open(Path(config.MODEL_DIR) / "training_log.csv", "w")
    log_file.write("epoch,train_loss,val_loss,val_miou,bg_iou,interior_iou,border_iou\n")
    
    # Compute class weights from actual data
    print("\nMenghitung class weights dari dataset...")
    class_weights = compute_class_weights(
        train_csv, train_dir / "masks"
    ).to(device)
    
    criterion = CombinedLoss(
        class_weights=class_weights,
        dice_weight=0.5,
        num_classes=config.NUM_CLASSES
    )

    # Training Loop
    print("\nMulai Training...")
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            
            loss = criterion(logits, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss = train_loss / max(train_batches, 1)
        
        # VAL
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_miou_total = 0.0
        val_ious_sum = {'bg': 0.0, 'interior': 0.0, 'border': 0.0}
        val_ious_count = {'bg': 0, 'interior': 0, 'border': 0}
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]  ")
            for batch in pbar_val:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(pixel_values=pixel_values)
                logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                logits = logits.float()
                
                loss = criterion(logits, labels)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batches += 1
                
                miou, per_class = calculate_iou(logits, labels, config.NUM_CLASSES)
                val_miou_total += miou
                for k, v in per_class.items():
                    if not np.isnan(v):
                        val_ious_sum[k] += v
                        val_ious_count[k] += 1
                
        val_loss = val_loss / max(val_batches, 1)
        val_miou = val_miou_total / len(val_loader)
        
        # Per-class IoU averages
        avg_ious = {}
        for k in ['bg', 'interior', 'border']:
            avg_ious[k] = val_ious_sum[k] / max(val_ious_count[k], 1)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        print(f"   IoU: bg={avg_ious['bg']:.4f} | interior={avg_ious['interior']:.4f} | border={avg_ious['border']:.4f}")
        log_file.write(f"{epoch+1},{train_loss},{val_loss},{val_miou},{avg_ious['bg']:.4f},{avg_ious['interior']:.4f},{avg_ious['border']:.4f}\n")
        log_file.flush()
        
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, Path(config.MODEL_DIR) / "checkpoint_best.pth")
            print(f"Best model saved! (mIoU={best_miou:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   LR sekarang: {current_lr:.2e}")
                
        torch.cuda.empty_cache()
        
    log_file.close()

if __name__ == "__main__":
    train()
