import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import config

import importlib
train_mod = importlib.import_module("04_train_segformer")
BuildingDataset = train_mod.BuildingDataset
calculate_iou = train_mod.calculate_iou


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    
    test_dir = Path(config.DATASET_DIR) / "test"
    test_dataset = BuildingDataset(
        csv_file=Path(config.DATASET_DIR) / "test.csv",
        img_dir=test_dir / "images",
        mask_dir=test_dir / "masks",
        augment=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    ckpt_path = Path(config.MODEL_DIR) / "checkpoint_best.pth"
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Mengevaluasi Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # 3-class confusion matrix
    class_names = ['Background', 'Interior', 'Border']
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    print("\n=== Final Evaluation on Test Set (3-Class) ===")
    print(f"\nConfusion Matrix:")
    print(f"{'':>12} | {'Pred BG':>8} | {'Pred Int':>8} | {'Pred Brd':>8}")
    print("-" * 48)
    for i, name in enumerate(class_names):
        print(f"{'True '+name:>12} | {cm[i,0]:>8} | {cm[i,1]:>8} | {cm[i,2]:>8}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':>12} | {'IoU':>8} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
    print("-" * 58)
    
    ious = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        ious.append(iou)
        
        print(f"{name:>12} | {iou:>8.4f} | {precision:>10.4f} | {recall:>8.4f} | {f1:>8.4f}")
    
    miou = sum(ious) / len(ious)
    print(f"\n{'Mean IoU':>12} | {miou:>8.4f}")
    
    # Save report
    with open(Path(config.MODEL_DIR) / "evaluation_report.txt", "w") as f:
        f.write("=== Final Evaluation on Test Set (3-Class) ===\n\n")
        for i, name in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            f.write(f"{name}: IoU={iou:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}\n")
        f.write(f"\nMean IoU: {miou:.4f}\n")
    
    print(f"\nReport saved to {Path(config.MODEL_DIR) / 'evaluation_report.txt'}")


if __name__ == "__main__":
    evaluate()
