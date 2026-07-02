"""
07_vectorize.py — Konversi mask instance SegFormer menjadi GeoJSON polygon bangunan.

Output: file GeoJSON berisi polygon yang di-aproksimasi (preserves bentuk asli)
beserta area (dalam m2 yang akurat via CRS transformasi UTM), 
instance_id, dan jumlah vertices.

Cara pakai:
    python 07_vectorize.py
    python 07_vectorize.py --output hasil_bangunan.geojson
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import json
import mercantile
from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon, mapping
import pyproj
from shapely.ops import transform
import config


def pixel_to_geo(point_list, tile_bounds, tile_size):
    """
    Konversi list of points (dari OpenCV contours) ke geografis (lon, lat).
    """
    west, south, east, north = tile_bounds.west, tile_bounds.south, tile_bounds.east, tile_bounds.north
    
    geo_points = []
    for point in point_list:
        px, py = point[0]  # OpenCV returns points as [[x, y]]
        lon = west + (px / tile_size) * (east - west)
        lat = north - (py / tile_size) * (north - south)  # Y terbalik
        geo_points.append((lon, lat))
    
    return geo_points


def extract_buildings_from_instance_mask(mask_path, tile_bounds, tile_size):
    """
    Ekstrak polygon bangunan langsung dari instance_mask (uint16)
    yang dihasilkan 06_predict.py via watershed.
    """
    # Instance mask berbentuk uint16 (0 = background, >0 = instance ID)
    mask = np.array(Image.open(mask_path))
    
    if mask.max() == 0:
        return []
    
    polygons = []
    
    # Ambil list ID unik bangunan
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]  # Abaikan 0 (background)
    
    for inst_id in instance_ids:
        # Ekstrak mask binary HANYA untuk bangunan ini
        inst_binary = (mask == inst_id).astype(np.uint8) * 255
        
        # Ekstrak kontur
        contours, _ = cv2.findContours(inst_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter bangunan yang sangat kecil (noise)
            if area < 20:
                continue
                
            # Simplifikasi polygon (TIDAK LAGI minAreaRect, agar bentuk asli terjaga)
            # epsilon yang lebih besar akan lebih menyederhanakan, lebih kecil lebih detail
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Konversi piksel → geo
            geo_points = pixel_to_geo(approx, tile_bounds, tile_size)
            
            # Buat Shapely Polygon
            if len(geo_points) >= 3:
                poly = Polygon(geo_points)
                if poly.is_valid and poly.area > 0:
                    polygons.append({
                        "polygon": poly,
                        "instance_id": int(inst_id),
                        "num_vertices": len(geo_points)
                    })
    
    return polygons


def get_utm_crs(lon, lat):
    """Mendapatkan string inisialisasi CRS UTM untuk longitude/latitude tertentu."""
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = "south" if lat < 0 else "north"
    return f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"


def calculate_accurate_area(poly):
    """
    Hitung area polygon (m2) dengan akurasi tinggi menggunakan proyeksi UTM lokal.
    """
    lon, lat = poly.centroid.x, poly.centroid.y
    utm_crs = get_utm_crs(lon, lat)
    
    project = pyproj.Transformer.from_crs(
        pyproj.CRS('epsg:4326'), # lon/lat asli
        pyproj.CRS(utm_crs),     # UTM lokal
        always_xy=True
    ).transform
    
    utm_poly = transform(project, poly)
    return utm_poly.area


def main():
    parser = argparse.ArgumentParser(description="Vektorisasi prediksi mask menjadi GeoJSON")
    parser.add_argument("--predictions", type=str, default="predictions/masks",
                        help="Folder berisi mask prediksi")
    parser.add_argument("--output", type=str, default="predictions/buildings.geojson",
                        help="Path output GeoJSON")
    args = parser.parse_args()
    
    mask_dir = Path(args.predictions)
    if not mask_dir.exists():
        print(f"Error: Folder {mask_dir} tidak ditemukan!")
        print("Jalankan 06_predict.py terlebih dahulu.")
        return
    
    # Baca pairs.csv untuk mendapatkan koordinat geo setiap tile
    pairs_csv = Path(config.DATASET_DIR) / "pairs.csv"
    if not pairs_csv.exists():
        print(f"Error: {pairs_csv} tidak ditemukan!")
        return
    
    df = pd.read_csv(pairs_csv)
    
    # Buat lookup: nama file (mask biasa / instance) → koordinat tile
    tile_lookup = {}
    for _, row in df.iterrows():
        img_name = Path(row['image_path']).stem  # tile_17_108433_65543
        
        # Simpan info tile-nya untuk lookup
        info = {
            "tile_x": int(row["tile_x"]),
            "tile_y": int(row["tile_y"]),
            "tile_z": int(row["tile_z"]),
        }
        
        # Support old format just in case, but prefer instance_{name}
        tile_lookup[f"instance_{img_name}.png"] = info
        tile_lookup[f"mask_{img_name}.png"] = info
    
    # Proses SEMUA mask instance
    mask_files = sorted(list(mask_dir.glob("instance_*.png")))
    if not mask_files:
        print(f"Warning: Tidak ada file 'instance_*.png' di {mask_dir}.")
        print("Pastikan Anda sudah menjalankan 06_predict.py versi terbaru.")
        return
        
    print(f"Memproses {len(mask_files)} mask instance...")
    
    all_features = []
    total_buildings = 0
    
    for i, mask_path in enumerate(mask_files):
        mask_name = mask_path.name
        
        # Cari koordinat tile
        if mask_name not in tile_lookup:
            # Parse fallback dari string (instance_tile_Z_X_Y.png)
            parts = mask_path.stem.replace("instance_tile_", "").split("_")
            if len(parts) == 3:
                z, x, y = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                print(f"  Skip {mask_name} (koordinat tidak ditemukan)")
                continue
        else:
            info = tile_lookup[mask_name]
            x, y, z = info["tile_x"], info["tile_y"], info["tile_z"]
        
        # Dapatkan batas geografis tile
        tile_bounds = mercantile.bounds(mercantile.Tile(x=x, y=y, z=z))
        
        # Ekstrak polygon bangunan (mendapatkan array dictionary)
        building_dicts = extract_buildings_from_instance_mask(mask_path, tile_bounds, config.TILE_SIZE)
        
        for b_dict in building_dicts:
            poly = b_dict["polygon"]
            
            feature = {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "source_tile": f"{z}/{x}/{y}",
                    "instance_id": b_dict["instance_id"],
                    "num_vertices": b_dict["num_vertices"],
                    "area_m2": round(calculate_accurate_area(poly), 1)
                }
            }
            all_features.append(feature)
        
        total_buildings += len(building_dicts)
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(mask_files)}] Total bangunan: {total_buildings}")
    
    # Simpan GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": all_features
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    
    print(f"\n{'='*60}")
    print(f"  VEKTORISASI SELESAI (3-CLASS PIPELINE)")
    print(f"  Total bangunan  : {total_buildings}")
    print(f"  Output GeoJSON  : {output_path}")
    print(f"  Bisa dibuka di  : QGIS, Google Earth, geojson.io")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
