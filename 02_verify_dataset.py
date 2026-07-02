import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import config


def verify_dataset():
    out_dir = Path(config.DATASET_DIR)
    csv_path = out_dir / "pairs.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} tidak ditemukan!")
        return
        
    df = pd.read_csv(csv_path)
    
    print("=" * 50)
    print(" Verifikasi Dataset SegFormer (3-Class)")
    print("=" * 50)
    
    # === Laporan distribusi zoom ===
    if 'tile_z' in df.columns:
        zoom_dist = df['tile_z'].value_counts().sort_index()
        print("\nDistribusi Zoom Level:")
        for z, count in zoom_dist.items():
            pct = count / len(df) * 100
            marker = " <<<" if z == 19 else ""
            print(f"  Zoom {z}: {count:>6} tile ({pct:.1f}%){marker}")
        
        non_19 = df[df['tile_z'] != 19]
        if len(non_19) > 0.1 * len(df):
            print(f"\n  PERINGATAN: {len(non_19)} tile ({len(non_19)/len(df)*100:.1f}%) bukan zoom 19!")
            print("  Gunakan --zoom-level 19 di 03_split_dataset.py untuk filter.")
    
    total_pairs = len(df)
    valid_pairs = 0
    missing_files = 0
    wrong_size = 0
    has_building = 0
    
    # Statistik per-kelas
    class_pixels = {0: 0, 1: 0, 2: 0}
    total_pixels = 0
    
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    
    print("\nSedang memverifikasi file...")
    
    valid_indices = []
    
    for idx, row in df.iterrows():
        img_p = Path(row['image_path'])
        mask_p = Path(row['mask_path'])
        
        if not img_p.exists() or not mask_p.exists():
            missing_files += 1
            continue
            
        with Image.open(img_p) as img, Image.open(mask_p) as mask:
            if img.size != (config.TILE_SIZE, config.TILE_SIZE) or mask.size != (config.TILE_SIZE, config.TILE_SIZE):
                wrong_size += 1
                continue
                
            mask_arr = np.array(mask)
            
            # Validasi: mask hanya boleh berisi 0, 1, 2
            valid_values = {0, 1, 2}
            unique_vals = set(np.unique(mask_arr))
            if not unique_vals.issubset(valid_values):
                print(f"  Peringatan: Mask {mask_p.name} mengandung nilai tidak valid: {unique_vals - valid_values}")
            
            # Statistik per-kelas
            for cls in [0, 1, 2]:
                class_pixels[cls] += np.sum(mask_arr == cls)
            total_pixels += mask_arr.size
                
            if np.sum(mask_arr > 0) > 0:
                has_building += 1
                valid_pairs += 1
                
                # Buat preview untuk 200 sampel pertama
                if valid_pairs <= 200:
                    create_preview_3class(img, mask_arr, preview_dir / f"sample_{valid_pairs:04d}.png")
            else:
                valid_pairs += 1
                
            valid_indices.append(idx)

    # Bersihkan CSV jika ada masalah
    if missing_files > 0 or wrong_size > 0:
        print("\nMembersihkan file yang bermasalah dari dataset...")
        df_cleaned = df.loc[valid_indices]
        df_cleaned.to_csv(csv_path, index=False)
        print(f"Dataset berhasil dibersihkan! Tersisa {len(df_cleaned)} tile valid.")

    print(f"\nHasil Verifikasi:")
    print(f"  Total entries di CSV   : {total_pairs}")
    print(f"  File hilang            : {missing_files}")
    print(f"  Ukuran tidak sesuai    : {wrong_size}")
    print(f"  Pair valid             : {valid_pairs}")
    if valid_pairs:
        print(f"  Pair memiliki bangunan : {has_building} ({has_building/valid_pairs*100:.1f}%)")
    
    # Statistik kelas
    if total_pixels > 0:
        print(f"\nStatistik Kelas (3-Class):")
        labels = {0: "Background", 1: "Interior", 2: "Border"}
        for cls in [0, 1, 2]:
            pct = class_pixels[cls] / total_pixels * 100
            print(f"  Kelas {cls} ({labels[cls]:>10}): {class_pixels[cls]:>12,} piksel ({pct:.2f}%)")
        
        # Recommended weights (inverse-frequency)
        counts = np.array([class_pixels[0], class_pixels[1], class_pixels[2]], dtype=np.float64)
        counts = np.maximum(counts, 1)
        inv_freq = total_pixels / (3 * counts)
        inv_freq = inv_freq / inv_freq[0]
        print(f"\n  Recommended class_weights: [{inv_freq[0]:.2f}, {inv_freq[1]:.2f}, {inv_freq[2]:.2f}]")
    
    if missing_files == 0 and wrong_size == 0 and has_building > 0:
        print(f"\nDATASET SIAP UNTUK TRAINING (Lanjut ke step 03)")
    else:
        print(f"\nDATASET MEMILIKI MASALAH. Harap periksa log di atas.")


def create_preview_3class(img, mask_arr, out_path):
    """Buat preview visual untuk mask 3-class:
    hijau = interior, merah = border, transparan = background"""
    img_np = np.array(img.convert("RGB"))
    
    # Overlay dengan warna per-kelas
    overlay = img_np.copy().astype(np.float32)
    
    # Interior → hijau semi-transparan
    interior = mask_arr == 1
    overlay[interior, 0] *= 0.4
    overlay[interior, 1] = overlay[interior, 1] * 0.4 + 150
    overlay[interior, 2] *= 0.4
    
    # Border → merah semi-transparan
    border = mask_arr == 2
    overlay[border, 0] = overlay[border, 0] * 0.4 + 200
    overlay[border, 1] *= 0.3
    overlay[border, 2] *= 0.3
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Mask visual: 0=hitam, 1=abu gelap (127), 2=putih (255)
    mask_visual = np.zeros_like(mask_arr, dtype=np.uint8)
    mask_visual[mask_arr == 1] = 127
    mask_visual[mask_arr == 2] = 255
    mask_visual_rgb = np.stack((mask_visual, mask_visual, mask_visual), axis=-1)
    
    # Gabung 3 panel
    combined = np.concatenate((img_np, mask_visual_rgb, overlay), axis=1)
    Image.fromarray(combined).save(out_path)


if __name__ == "__main__":
    verify_dataset()
