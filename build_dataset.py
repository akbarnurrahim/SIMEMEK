# pip install requests geopandas shapely pandas tqdm pillow matplotlib mercantile rasterio numpy

import os
import time
import argparse
import random
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import mercantile
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from shapely import wkt
from shapely.geometry import box
import rasterio.features
from rasterio.transform import from_bounds
import warnings

# Suppress pandas/geopandas warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
# Area of Interest (bounding box)
# Contoh default: sebagian Jawa Barat
AOI_BBOX = (106.7, -6.3, 107.0, -6.0)  # (lon_min, lat_min, lon_max, lat_max)

# Zoom level tile (17 = ~1m/pixel, 18 = ~0.5m/pixel)
ZOOM_LEVEL = 17

# Ukuran output tile dan mask (pixel)
TILE_SIZE = 256  # 256x256 pixel

# Minimum confidence Open Buildings
MIN_CONFIDENCE = 0.75

# URL tiles Open Buildings
OB_TILES_URL = "https://sites.research.google/open-buildings/tiles.geojson"

# Output directory
OUTPUT_DIR = "dataset"


# ============================================================
# FUNGSI-FUNGSI PIPELINE
# ============================================================

def get_tiles_for_bbox(bbox, zoom):
    """Mendapatkan daftar tile yang masuk dalam bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = list(mercantile.tiles(lon_min, lat_min, lon_max, lat_max, zooms=[zoom]))
    
    tile_list = []
    for t in tiles:
        bounds = mercantile.bounds(t)
        tile_bbox = (bounds.west, bounds.south, bounds.east, bounds.north)
        tile_list.append((t.x, t.y, t.z, tile_bbox))
    return tile_list


def download_osm_tile(x, y, z, output_path, tile_size=TILE_SIZE):
    """Mengunduh tile dari OSM dan meresizenya jika perlu."""
    if os.path.exists(output_path):
        return True
        
    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    headers = {"User-Agent": "BuildingDatasetBuilder/1.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Simpan file sementara
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        # Buka dan pastikan ukuran sesuai
        with Image.open(output_path) as img:
            if img.size != (tile_size, tile_size):
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                img.save(output_path)
                
        time.sleep(0.1)  # Menghormati rate limit OSM
        return True
    except Exception as e:
        print(f"\nGagal download tile {z}/{x}/{y}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def get_ob_buildings_for_bbox(bbox, min_conf=MIN_CONFIDENCE):
    """Mengunduh data bangunan menggunakan Overpass API (OSM) sebagai pengganti Open Buildings."""
    lon_min, lat_min, lon_max, lat_max = bbox
    area_poly = box(lon_min, lat_min, lon_max, lat_max)
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
      relation["building"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out geom;
    """
    
    try:
        print(f"📡 Mengambil metadata bangunan dari OSM (Overpass API)...")
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        polygons = []
        from shapely.geometry import Polygon
        for element in data.get('elements', []):
            if element['type'] == 'way':
                if 'geometry' in element:
                    coords = [(pt['lon'], pt['lat']) for pt in element['geometry']]
                    if len(coords) >= 3:
                        # Ensure it's closed
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        polygons.append(Polygon(coords))
                        
        if polygons:
            gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
            return gdf[gdf.intersects(area_poly)]
            
    except Exception as e:
        print(f"⚠️ Peringatan: Tidak dapat mengunduh data bangunan ({e}).")
        
    return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")


def rasterize_buildings(gdf, tile_bbox, tile_size=TILE_SIZE, x=None, y=None, z=None):
    """Mengubah polygon bangunan menjadi mask numpy array."""
    if gdf.empty:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)
        
    # Konversi ke EPSG:3857 (Web Mercator) untuk rasterisasi akurat
    gdf_3857 = gdf.to_crs(epsg=3857)
    
    # Ambil bounds tile dalam Web Mercator
    # Jika kita punya xyz, lebih akurat pakai mercantile.xy_bounds
    if x is not None and y is not None and z is not None:
        bounds = mercantile.xy_bounds(x, y, z)
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    else:
        # Fallback konversi manual
        lon_min, lat_min, lon_max, lat_max = tile_bbox
        bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326").to_crs(epsg=3857)
        west, south, east, north = bbox_gdf.total_bounds

    transform = from_bounds(west, south, east, north, tile_size, tile_size)
    
    geometries = [(geom, 255) for geom in gdf_3857.geometry if not geom.is_empty]
    
    if not geometries:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)
        
    mask = rasterio.features.rasterize(
        geometries,
        out_shape=(tile_size, tile_size),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask


def save_mask(mask_array, output_path):
    """Menyimpan array mask ke file PNG."""
    img = Image.fromarray(mask_array, mode='L')
    img.save(output_path)
    return output_path


def build_dataset(bbox, zoom, output_dir, min_conf=MIN_CONFIDENCE, tile_size=TILE_SIZE):
    """Pipeline utama pembuat dataset."""
    print("=" * 60)
    print("  Building Segmentation Dataset Builder")
    print("=" * 60)
    print(f"  AOI Bbox  : {bbox}")
    print(f"  Zoom Level: {zoom}")
    print(f"  Output Dir: {output_dir}/")
    print("=" * 60)

    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print("\n📡 Mengambil building data dari Open Buildings...")
    buildings_gdf = get_ob_buildings_for_bbox(bbox, min_conf)
    print(f"✅ {len(buildings_gdf)} bangunan ditemukan dalam AOI")
    
    print("\n🗺️  Menghitung tile yang dibutuhkan...")
    tiles = get_tiles_for_bbox(bbox, zoom)
    print(f"✅ {len(tiles)} tile pada zoom level {zoom}")
    
    pairs_data = []
    failed_tiles = []
    
    pairs_csv_path = os.path.join(output_dir, 'pairs.csv')
    if os.path.exists(pairs_csv_path):
        existing_df = pd.read_csv(pairs_csv_path)
        pairs_data = existing_df.to_dict('records')
        existing_images = set(existing_df['image_path'])
    else:
        existing_images = set()

    print(f"\nDownloading and processing {len(tiles)} tiles...")
    pbar = tqdm(tiles, desc="Downloading tiles")
    
    success_count = 0
    for x, y, z, tile_bbox in pbar:
        img_name = f"tile_{z}_{x}_{y}.png"
        mask_name = f"mask_{z}_{x}_{y}.png"
        
        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, mask_name)
        
        # Skip jika sudah dikerjakan
        if f"images/{img_name}" in existing_images and os.path.exists(img_path) and os.path.exists(mask_path):
            success_count += 1
            continue
            
        # Download Image
        if not download_osm_tile(x, y, z, img_path, tile_size):
            failed_tiles.append(f"{z}/{x}/{y}")
            continue
            
        # Buat & Simpan Mask
        tile_poly = box(*tile_bbox)
        # Ambil subset bangunan untuk kecepatan rasterize
        tile_buildings = buildings_gdf[buildings_gdf.intersects(tile_poly)]
        mask_array = rasterize_buildings(tile_buildings, tile_bbox, tile_size, x, y, z)
        save_mask(mask_array, mask_path)
        
        pairs_data.append({
            'image_path': f"images/{img_name}",
            'mask_path': f"masks/{mask_name}",
            'tile_x': x,
            'tile_y': y,
            'tile_z': z,
            'lon_min': tile_bbox[0],
            'lat_min': tile_bbox[1],
            'lon_max': tile_bbox[2],
            'lat_max': tile_bbox[3]
        })
        success_count += 1
        
        # Simpan CSV secara reguler setiap 50 tile
        if success_count % 50 == 0:
            pd.DataFrame(pairs_data).to_csv(pairs_csv_path, index=False)

    print(f"  ✓ {success_count} tile berhasil")
    if failed_tiles:
        print(f"  ✗ {len(failed_tiles)} tile gagal (dicatat di failed_tiles.txt)")
        with open(os.path.join(output_dir, 'failed_tiles.txt'), 'w') as f:
            f.write('\n'.join(failed_tiles))
            
    print("\n💾 Menyimpan pairs.csv...")
    if pairs_data:
        pd.DataFrame(pairs_data).to_csv(pairs_csv_path, index=False)
        
    print(f"✅ Dataset selesai! {success_count} pasang image-mask tersimpan di {output_dir}/")


# ============================================================
# FUNGSI VALIDASI & PREVIEW
# ============================================================

def validate_dataset(output_dir):
    print("\n" + "="*40)
    print("Validasi Dataset")
    print("="*40)
    
    csv_path = os.path.join(output_dir, 'pairs.csv')
    if not os.path.exists(csv_path):
        print("❌ pairs.csv tidak ditemukan!")
        return
        
    df = pd.read_csv(csv_path)
    total_pairs = len(df)
    
    missing_files = 0
    wrong_sizes = 0
    has_buildings_count = 0
    total_coverage = 0.0
    
    for _, row in tqdm(df.iterrows(), total=total_pairs, desc="Memvalidasi"):
        img_path = os.path.join(output_dir, row['image_path'])
        mask_path = os.path.join(output_dir, row['mask_path'])
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            missing_files += 1
            continue
            
        with Image.open(img_path) as img, Image.open(mask_path) as mask:
            if img.size != (TILE_SIZE, TILE_SIZE) or mask.size != (TILE_SIZE, TILE_SIZE):
                wrong_sizes += 1
            
            mask_arr = np.array(mask)
            white_pixels = np.sum(mask_arr == 255)
            if white_pixels > 0:
                has_buildings_count += 1
                coverage = white_pixels / (TILE_SIZE * TILE_SIZE)
                total_coverage += coverage
                
    print("\n--- Summary ---")
    print(f"Total pairs       : {total_pairs}")
    print(f"File missing      : {missing_files}")
    print(f"Ukuran salah      : {wrong_sizes}")
    
    if total_pairs - missing_files > 0:
        pct_has_buildings = (has_buildings_count / (total_pairs - missing_files)) * 100
        avg_cov = (total_coverage / has_buildings_count * 100) if has_buildings_count > 0 else 0
        print(f"Ada bangunan      : {has_buildings_count} tiles ({pct_has_buildings:.1f}%)")
        print(f"Rata-rata cov.    : {avg_cov:.2f}% (hanya pada tile berbangunan)")
    else:
        print("Tidak ada data valid untuk dihitung statistiknya.")


def visualize_samples(output_dir, n=5):
    csv_path = os.path.join(output_dir, 'pairs.csv')
    if not os.path.exists(csv_path):
        print("❌ pairs.csv tidak ditemukan!")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("Dataset kosong.")
        return
        
    samples = df.sample(min(n, len(df)))
    
    fig, axes = plt.subplots(len(samples), 2, figsize=(10, 5 * len(samples)))
    
    # Jika hanya 1 sample, axes bentuknya 1D
    if len(samples) == 1:
        axes = [axes]
        
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(output_dir, row['image_path'])
        mask_path = os.path.join(output_dir, row['mask_path'])
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        axes[i][0].imshow(img)
        axes[i][0].set_title(f"Image {row['tile_z']}/{row['tile_x']}/{row['tile_y']}")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(mask, cmap='gray')
        axes[i][1].set_title("Mask")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    out_img = os.path.join(output_dir, 'sample_preview.png')
    plt.savefig(out_img)
    print(f"✅ Preview tersimpan di: {out_img}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float,
                        default=list(AOI_BBOX),
                        help="LON_MIN LAT_MIN LON_MAX LAT_MAX",
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    parser.add_argument("--zoom", type=int, default=ZOOM_LEVEL, help="Tile zoom level (15-18)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--min_conf", type=float, default=MIN_CONFIDENCE, help="Min confidence")
    parser.add_argument("--tile_size", type=int, default=TILE_SIZE, help="Tile size (256/512)")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--preview", action="store_true", help="Generate preview samples")
    args = parser.parse_args()

    if args.validate:
        validate_dataset(args.output)
    elif args.preview:
        visualize_samples(args.output)
    else:
        build_dataset(tuple(args.bbox), args.zoom, args.output, args.min_conf, args.tile_size)
