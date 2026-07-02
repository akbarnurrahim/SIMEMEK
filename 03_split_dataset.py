import os
import argparse
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import config


def split_dataset(zoom_level=19):
    out_dir = Path(config.DATASET_DIR)
    csv_path = out_dir / "pairs.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} tidak ditemukan!")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Total tile di pairs.csv: {len(df)}")
    
    # Filter zoom level
    if 'tile_z' in df.columns:
        before = len(df)
        df = df[df['tile_z'] == zoom_level]
        print(f"Filter zoom {zoom_level}: {before} -> {len(df)} tile")
    
    # Filter hanya yang memiliki bangunan sesuai minimum pixel
    valid_df = df[df['building_pixel_count'] >= config.MIN_BUILDING_PIXELS].copy()
    
    if len(valid_df) == 0:
        print("Error: Tidak ada tile yang memiliki cukup bangunan untuk ditraining.")
        return
        
    print(f"Menggunakan {len(valid_df)} tile valid dari total {len(df)} tile.")
    
    # Bining coverage untuk stratified split
    bins = np.linspace(0, 100, 10)
    valid_df['cov_bin'] = pd.cut(valid_df['building_coverage_pct'], bins=bins, labels=False, include_lowest=True)
    
    # Grouping bin kecil agar stratify tidak gagal
    cov_counts = valid_df['cov_bin'].value_counts()
    valid_bins = cov_counts[cov_counts >= 3].index
    valid_df['cov_bin'] = valid_df['cov_bin'].apply(lambda x: x if x in valid_bins else -1)
    
    train_ratio = config.TRAIN_RATIO
    val_test_ratio = config.VAL_RATIO + config.TEST_RATIO
    
    # Stratified split 1: train vs (val+test)
    train_df, temp_df = train_test_split(
        valid_df, 
        test_size=val_test_ratio, 
        random_state=42,
        stratify=valid_df['cov_bin']
    )
    
    val_ratio_adj = config.VAL_RATIO / val_test_ratio
    
    # Stratified split 2: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_adj),
        random_state=42,
        stratify=temp_df['cov_bin']
    )
    
    print(f"\nDistribusi Split:")
    print(f"  Train : {len(train_df)} ({len(train_df)/len(valid_df)*100:.1f}%)")
    print(f"  Val   : {len(val_df)} ({len(val_df)/len(valid_df)*100:.1f}%)")
    print(f"  Test  : {len(test_df)} ({len(test_df)/len(valid_df)*100:.1f}%)")
    
    # Buat direktori dan copy file
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    
    for split_name, split_df in splits.items():
        split_dir = out_dir / split_name
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)
        
        # Simpan CSV split (tanpa kolom cov_bin)
        split_df = split_df.drop(columns=['cov_bin'])
        split_df.to_csv(out_dir / f"{split_name}.csv", index=False)
        
        print(f"Mengkopi file untuk {split_name}...")
        for _, row in split_df.iterrows():
            img_src = Path(row['image_path'])
            mask_src = Path(row['mask_path'])
            
            if img_src.exists():
                shutil.copy(img_src, split_dir / "images" / img_src.name)
            if mask_src.exists():
                shutil.copy(mask_src, split_dir / "masks" / mask_src.name)
            
    print("\nDataset berhasil di-split dan siap untuk training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset ke train/val/test")
    parser.add_argument("--zoom-level", type=int, default=19,
                        help="Filter tile berdasarkan zoom level (default: 19)")
    args = parser.parse_args()
    split_dataset(zoom_level=args.zoom_level)
