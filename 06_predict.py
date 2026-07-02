"""
Prediksi 3-class SegFormer + Instance Separation via Watershed.

Pipeline:
1. Model prediksi 3 kelas: background=0, interior=1, border=2
2. Interior pixels (setelah morphological cleaning) → seed/marker untuk 
   cv2.connectedComponents(), setiap bangunan mendapat ID unik
3. Area border + sisa unknown di antara interior dan full building mask 
   → ditandai sebagai zona "unknown" untuk watershed
4. cv2.watershed() meng-grow tiap seed interior ke zona unknown tersebut, 
   dibatasi oleh distance transform dari full_building sebagai elevation 
   map, sehingga piksel ambigu jatuh ke seed yang paling dekat secara 
   spasial (bukan sekadar ID terbesar)
5. Hasil watershed dibersihkan (boundary line -1 dan background dijadikan 
   0) → instance_mask final dengan ID unik per bangunan
"""
import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from transformers import SegformerForSemanticSegmentation
import config


def instance_separation(pred_mask):
    """
    Pisahkan bangunan individual menggunakan prediksi 3-class.
    
    Args:
        pred_mask: numpy array (H, W) dengan nilai 0=bg, 1=interior, 2=border
    
    Returns:
        instance_mask: numpy array (H, W) dengan ID unik per bangunan (0=bg)
        num_buildings: jumlah bangunan terdeteksi
        building_mask: binary mask seluruh bangunan (untuk visualisasi)
    """
    interior = (pred_mask == 1).astype(np.uint8)
    border = (pred_mask == 2).astype(np.uint8)
    
    # Morphological opening kecil untuk hapus noise interior
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    interior_clean = cv2.morphologyEx(interior, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Connected components pada interior → setiap bangunan mendapat ID unik
    num_labels, markers = cv2.connectedComponents(interior_clean)
    num_buildings = num_labels - 1  # Label 0 = background
    
    # Gunakan watershed untuk ekspansi ke area border
    # OpenCV watershed butuh markers: 
    #   0 = unknown area (akan diisi oleh watershed)
    #   1 = sure background (kita pakai 1 karena 0 untuk unknown)
    #   >1 = seeds untuk instance bangunan
    
    markers = markers + 1 # Shift label: 0 jadi 1 (bg), 1 jadi 2 (seed_1), dst.
    
    # Tentukan area unknown: semua yg diprediksi sbg bangunan (interior/border) 
    # TAPI belum jadi seed (interior_clean)
    full_building = (pred_mask > 0).astype(np.uint8)
    unknown = cv2.subtract(full_building, interior_clean)
    markers[unknown == 1] = 0
    
    # Area background yg PASTI (bukan bangunan)
    markers[full_building == 0] = 1
    
    # Dummy 3-channel image sebagai elevation map untuk watershed.
    # Kita buat dari distance transform FULL BUILDING (interior+border), lalu di-invert,
    # supaya piksel dekat tepi luar bangunan punya elevation TINGGI,
    # dan piksel di tengah bangunan (dekat seed) punya elevation RENDAH.
    # Ini memberi gradien mengalir dari luar ke dalam untuk memisahkan instance menempel.
    dist_transform = cv2.distanceTransform(full_building, cv2.DIST_L2, 3)
    # Invert agar batas antar bangunan jadi puncak (watershed ridge)
    max_dist = dist_transform.max() if dist_transform.max() > 0 else 1
    dist_transform = max_dist - dist_transform
    
    cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
    dummy_img = cv2.cvtColor(dist_transform.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Eksekusi watershed
    markers = cv2.watershed(dummy_img, markers)
    
    # Bersihkan hasil: -1 (watershed boundary) dan 1 (background) jadi 0
    instance_mask = markers.copy()
    instance_mask[instance_mask == -1] = 0
    instance_mask[instance_mask == 1] = 0
    
    # Kembalikan ID mulai dari 1, 2, ...
    instance_mask[instance_mask > 1] -= 1
    
    # Building mask = semua area yang punya instance ID > 0
    building_mask = (instance_mask > 0).astype(np.uint8) * 255
    
    return instance_mask, num_buildings, building_mask


def predict(image_path, model, device, transform, out_dir):
    img = Image.open(image_path).convert("RGB")
    
    img_resized = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.LANCZOS)
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = outputs.logits
        
        logits = nn.functional.interpolate(
            logits, size=(config.IMAGE_SIZE, config.IMAGE_SIZE), mode="bilinear", align_corners=False
        )
        # Prediksi 3-class: 0=background, 1=interior, 2=border
        raw_preds = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    # Instance separation menggunakan border barriers
    instance_mask, num_buildings, _ = instance_separation(raw_preds)
    
    # === Buat gambar output 3 panel: IMAGE | 3-CLASS MASK | OVERLAY ===
    img_arr = np.array(img_resized)
    
    # Panel 2: Mask 3-class visual (hitam=bg, hijau=interior, merah=border)
    mask_visual = np.zeros((*raw_preds.shape, 3), dtype=np.uint8)
    mask_visual[raw_preds == 1] = [0, 200, 0]    # Interior = hijau
    mask_visual[raw_preds == 2] = [200, 50, 50]   # Border = merah
    
    # Panel 3: Overlay pada foto satelit
    overlay = img_arr.copy().astype(np.float32)
    
    # Interior → hijau semi-transparan
    interior_px = raw_preds == 1
    overlay[interior_px, 0] *= 0.4
    overlay[interior_px, 1] = overlay[interior_px, 1] * 0.4 + 150
    overlay[interior_px, 2] *= 0.4
    
    # Border → merah semi-transparan
    border_px = raw_preds == 2
    overlay[border_px, 0] = overlay[border_px, 0] * 0.4 + 180
    overlay[border_px, 1] *= 0.3
    overlay[border_px, 2] *= 0.3
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Gambar garis instance boundary (tiap bangunan warna berbeda)
    for inst_id in range(1, num_buildings + 1):
        inst_binary = (instance_mask == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(inst_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Warna unik per instance
        color = (
            int(50 + (inst_id * 67) % 200),
            int(50 + (inst_id * 131) % 200),
            int(50 + (inst_id * 197) % 200),
        )
        # BUG FIX: gambar langsung ke overlay asli
        cv2.drawContours(overlay, contours, -1, color, 1)
    
    # Gabungkan 3 panel
    combined = np.concatenate((img_arr, mask_visual, overlay), axis=1)
    
    out_path = out_dir / f"pred_{Path(image_path).name}"
    Image.fromarray(combined).save(out_path)
    
    # Simpan mask instance terpisah
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    # Instance mask: setiap bangunan punya ID unik (1, 2, 3, ...)
    Image.fromarray(instance_mask.astype(np.uint16)).save(mask_dir / f"instance_{Path(image_path).name}")
    # Raw 3-class mask
    Image.fromarray(raw_preds).save(mask_dir / f"mask3c_{Path(image_path).name}")
    
    print(f"  Prediksi disimpan: {out_path} ({num_buildings} bangunan terdeteksi)")
    return num_buildings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path ke gambar")
    parser.add_argument("--folder", type=str, help="Path ke folder gambar")
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        print("Harap berikan argumen --image atau --folder")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = Path(config.MODEL_DIR) / "checkpoint_best.pth"
    if not ckpt_path.exists():
        print(f"Error: Model belum di-training! ({ckpt_path} tidak ada)")
        return
    
    print(f"Load model dari {ckpt_path}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)
    
    total_buildings = 0
    
    if args.image:
        total_buildings += predict(args.image, model, device, transform, out_dir)
    
    if args.folder:
        folder = Path(args.folder)
        images = sorted(folder.glob("*.png"))
        print(f"\nMemproses {len(images)} gambar...")
        for i, img_path in enumerate(images):
            print(f"[{i+1}/{len(images)}]", end="")
            total_buildings += predict(img_path, model, device, transform, out_dir)
    
    print(f"\n{'='*50}")
    print(f"  SELESAI! Total bangunan terdeteksi: {total_buildings}")
    print(f"  Output tersimpan di: {out_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
