import os
import sys
import argparse
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import mercantile
from PIL import Image
from pathlib import Path
from shapely.geometry import box
import json
import rasterio.features
from tqdm import tqdm

# Import config
import config

def get_tiles_for_bbox(bbox, zoom):
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = list(mercantile.tiles(lon_min, lat_min, lon_max, lat_max, zooms=zoom))
    return tiles

def load_buildings_fgb(fgb_path, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    print(f" Membaca IDN.fgb untuk bbox: {bbox}")
    try:
        if not os.path.exists(fgb_path):
            print(f" Peringatan: File {fgb_path} tidak ditemukan!")
            print(" Pastikan Anda sudah mengunduh dan memindahkan file IDN.fgb ke folder ini.")
            return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")
            
        gdf = gpd.read_file(fgb_path, bbox=(lon_min, lat_min, lon_max, lat_max))
        print(f" Ditemukan {len(gdf)} bangunan dalam bbox.")
        return gdf
    except Exception as e:
        print(f" Error saat membaca FGB: {e}")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326")

def download_tile_image(x, y, z, output_path, tile_size):
    url = config.TILE_URL.format(x=x, y=y, z=z)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.arcgis.com",
    }
    
    if os.path.exists(output_path):
        return True
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Tambahkan delay kecil agar tidak dicap sebagai DDoS / Bot oleh Google Maps
            time.sleep(0.3)
            
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code in [403, 429]:
                print(f" Akses ditolak (403/429) oleh Google Maps. Menunggu sebelum retry...")
                time.sleep(2 * (attempt + 1))
                continue
                
            resp.raise_for_status()
            
            import uuid
            tmp_path = f"{output_path}.{uuid.uuid4().hex}.tmp"
            
            with open(tmp_path, 'wb') as f:
                f.write(resp.content)
                
            # Verify and resize
            with Image.open(tmp_path) as img:
                if img.size != (tile_size, tile_size):
                    img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                    img.save(tmp_path)
                    
            # Rename atomically to avoid concurrent read/write errors
            if os.path.exists(output_path):
                os.remove(tmp_path)
                return True
            else:
                try:
                    os.rename(tmp_path, output_path)
                except FileExistsError:
                    os.remove(tmp_path)
                except PermissionError:
                    os.remove(tmp_path)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f" Gagal download citra tile {z}/{x}/{y}: {e}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except: pass
                return False
    return False

def rasterize_buildings(tile, gdf, output_path, tile_size):
    bounds = mercantile.bounds(tile)
    tile_poly = box(bounds.west, bounds.south, bounds.east, bounds.north)
    
    # Filter bangunan yang beririsan dengan tile
    intersecting = gdf[gdf.intersects(tile_poly)].copy()
    
    if intersecting.empty:
        return 0, 0.0 # Pixel count, Coverage
        
    # Transform geometry to local pixel coordinates
    # PENTING: Harus konversi ke Web Mercator dulu karena tile satelit
    # menggunakan proyeksi Mercator (latitude TIDAK linear di derajat!)
    xy_bnds = mercantile.xy_bounds(tile)  # Bounds dalam meter Mercator
    
    def to_pixel_coords(geom):
        def transform_point(lon, lat):
            # Konversi lon/lat ke meter Mercator
            mx, my = mercantile.xy(lon, lat)
            # Interpolasi linear di ruang Mercator (yang memang linear untuk tile)
            px = round(((mx - xy_bnds.left) / (xy_bnds.right - xy_bnds.left)) * tile_size)
            py = round(((xy_bnds.top - my) / (xy_bnds.top - xy_bnds.bottom)) * tile_size)
            return px, py
            
        from shapely.ops import transform
        return transform(lambda x, y, z=None: transform_point(x, y), geom)
        
    local_geoms = intersecting['geometry'].apply(to_pixel_coords)
    
    # === 3-CLASS MASK GENERATION ===
    # Kelas: 0=background, 1=interior bangunan, 2=border bangunan
    border_px = config.BORDER_WIDTH_PX
    min_area = config.MIN_BORDER_AREA
    
    # Step 1: Rasterize semua polygon ASLI → "full building" mask
    full_shapes = [(geom, 1) for geom in local_geoms if not geom.is_empty]
    if not full_shapes:
        return 0, 0.0
    
    full_mask = rasterio.features.rasterize(
        full_shapes,
        out_shape=(tile_size, tile_size),
        fill=0,
        dtype=np.uint8
    )
    
    # Step 2: Untuk setiap bangunan, buat versi shrunk (interior)
    interior_shapes = []
    for geom in local_geoms:
        if geom.is_empty:
            continue
        if geom.area < min_area:
            # Bangunan kecil: jangan buat border, langsung full interior
            interior_shapes.append((geom, 1))
        else:
            shrunk = geom.buffer(-border_px)
            if not shrunk.is_empty and shrunk.is_valid and shrunk.area > 0:
                interior_shapes.append((shrunk, 1))
            # Jika shrunk habis (bangunan terlalu kecil setelah di-shrink),
            # bangunan ini akan jadi full border (dari full_mask - interior)
    
    interior_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    if interior_shapes:
        interior_mask = rasterio.features.rasterize(
            interior_shapes,
            out_shape=(tile_size, tile_size),
            fill=0,
            dtype=np.uint8
        )
    
    # Step 3: Gabungkan → mask 3 kelas
    # border = area yang ada di full_mask tapi TIDAK ada di interior_mask
    mask_3class = np.zeros((tile_size, tile_size), dtype=np.uint8)
    mask_3class[full_mask == 1] = 2       # Semua area bangunan → border dulu
    mask_3class[interior_mask == 1] = 1   # Timpa interior di atasnya → interior
    # Hasilnya: 0=background, 1=interior, 2=border (cincin di tepi bangunan)
    
    pixel_count = np.sum(full_mask)
    coverage = (pixel_count / (tile_size * tile_size)) * 100
    
    # Save mask langsung sebagai uint8 (0, 1, 2)
    Image.fromarray(mask_3class).save(output_path)
    
    return pixel_count, coverage

def main():
    parser = argparse.ArgumentParser(description="Prepare Dataset for SegFormer")
    parser.add_argument("--bbox", type=float, nargs=4, help="Bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--province", type=str, help="Gunakan preset bbox provinsi dari config.py")
    parser.add_argument("--zoom", type=int, default=config.ZOOM_LEVEL, help="Zoom level")
    parser.add_argument("--output", type=str, default=config.DATASET_DIR, help="Output directory")
    parser.add_argument("--max-tiles", type=int, default=None, help="Batasi jumlah tile yang didownload")
    parser.add_argument("--min_conf", type=float, default=config.MIN_CONFIDENCE, help="Minimum confidence (jika ada)")
    parser.add_argument("--tile_size", type=int, default=config.TILE_SIZE, help="Ukuran pixel tile")
    parser.add_argument("--validate", action="store_true", help="Jalankan mode validasi (menghitung statistik mask)")
    parser.add_argument("--force-regenerate", action="store_true", help="Timpa semua tile existing (abaikan pairs.csv lama)")
    parser.add_argument("--user", type=str, default="", help="Username untuk file progress spesifik")
    
    args = parser.parse_args()
    
    if args.validate:
        print(" Menjalankan Verifikasi Dataset...")
        pairs_file = os.path.join(args.output, "pairs.csv")
        if not os.path.exists(pairs_file):
            print(" Dataset belum ada atau pairs.csv tidak ditemukan.")
            sys.exit(1)
        df = pd.read_csv(pairs_file)
        valid = df[df['building_pixel_count'] > 0]
        print(f" Total Tile: {len(df)}")
        print(f" Tile dengan Bangunan: {len(valid)}")
        sys.exit(0)
    
    # Tentukan bbox
    bbox = config.AOI_BBOX
    if args.bbox:
        bbox = tuple(args.bbox)
    elif args.province and args.province in config.PROVINCE_BBOX:
        bbox = config.PROVINCE_BBOX[args.province]
        
    print("=" * 60)
    print("  SegFormer Dataset Builder (FGB + Google Maps)")
    print("=" * 60)
    print(f"  AOI Bbox  : {bbox}")
    print(f"  Zoom Level: {args.zoom}")
    print(f"  Output Dir: {args.output}")
    print("=" * 60)
    
    # Buat struktur folder
    out_dir = Path(args.output)
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    geojson_dir = out_dir / "geojson"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate tiles dulu agar kita tahu area total yang dibutuhkan
    tiles = get_tiles_for_bbox(bbox, args.zoom)
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    
    # Hitung bbox gabungan dari SEMUA tiles (lebih besar dari bbox user)
    # Ini memastikan tile di pinggiran tetap mendapat data bangunan yang lengkap
    all_bounds = [mercantile.bounds(t) for t in tiles]
    expanded_bbox = (
        min(b.west for b in all_bounds),
        min(b.south for b in all_bounds),
        max(b.east for b in all_bounds),
        max(b.north for b in all_bounds),
    )
    
    # Load FGB dengan bbox yang sudah diperluas
    gdf = load_buildings_fgb(config.FGB_PATH, expanded_bbox)
    # TIDAK perlu buffer di sini - pemisahan bangunan sudah ditangani
    # di fungsi rasterize_buildings() menggunakan shrink piksel yang presisi
    if not gdf.empty:
        gdf = gdf[~gdf.is_empty]
        
    
    print(f" Total Tile yang akan diproses: {len(tiles)}")
    
    # Inisialisasi progress.json spesifik user (atau global jika tidak ada)
    progress_filename = f"progress_{args.user}.json" if args.user else "progress.json"
    progress_file = out_dir / progress_filename
    progress_data = {"total": len(tiles), "done": 0, "failed": 0}
    with open(progress_file, "w") as f:
        json.dump(progress_data, f)
    
    # 3. Proses per tile
    pairs = []
    failed_tiles = []
    processed_count = 0
    
    # Jika sebelumnya ada pairs.csv, kita bisa append atau skip
    csv_path = out_dir / "pairs.csv"
    existing_tiles = set()
    
    if csv_path.exists():
        if args.force_regenerate:
            print("! --force-regenerate aktif: Mengabaikan data existing, memproses ulang semua tile.")
        else:
            existing_df = pd.read_csv(csv_path)
            
            # CEK SAFEGURAD: pastikan mask yang sudah ada BUKAN format binary lama (0/255)
            if len(existing_df) > 0:
                sample_mask_path = Path(existing_df.iloc[0]['mask_path'])
                if sample_mask_path.exists():
                    sample_mask = np.array(Image.open(sample_mask_path))
                    unique_vals = set(np.unique(sample_mask))
                    if not unique_vals.issubset({0, 1, 2}):
                        print("\n" + "!"*70)
                        print(" WARNING: Terdeteksi mask format lama (binary) di dataset existing!")
                        print(f"   Ditemukan nilai pixel: {unique_vals}")
                        print("   Anda sedang menjalankan pipeline 3-class. DILARANG MENCAMPUR FORMAT.")
                        print("   Solusi:")
                        print("   1. Jalankan script ini lagi dengan flag --force-regenerate, ATAU")
                        print("   2. Hapus folder dataset lama secara manual.")
                        print("!"*70 + "\n")
                        sys.exit(1)
            
            existing_tiles = set(zip(existing_df.tile_x, existing_df.tile_y, existing_df.tile_z))
            print(f"  Ditemukan {len(existing_tiles)} tile existing, akan di-skip.")
    
    pbar = tqdm(tiles, desc="Proses Tile", unit="tile")
    for tile in pbar:
        processed_count += 1
        
        # Update progress.json (ditaruh di awal agar yang skip juga terhitung)
        progress_data["done"] = processed_count
        progress_data["failed"] = len(failed_tiles)
        with open(progress_file, "w") as f:
            json.dump(progress_data, f)
            
        if (tile.x, tile.y, tile.z) in existing_tiles:
            continue
            
        img_path = img_dir / f"tile_{tile.z}_{tile.x}_{tile.y}.png"
        mask_path = mask_dir / f"mask_{tile.z}_{tile.x}_{tile.y}.png"
        geojson_path = geojson_dir / f"tile_{tile.z}_{tile.x}_{tile.y}.geojson"
        
        # Download citra
        if not download_tile_image(tile.x, tile.y, tile.z, str(img_path), args.tile_size):
            failed_tiles.append(tile)
            continue
            
        # Rasterize Mask & Save GeoJSON
        bounds = mercantile.bounds(tile)
        tile_poly = box(bounds.west, bounds.south, bounds.east, bounds.north)
        intersecting = gdf[gdf.intersects(tile_poly)].copy()
        
        if not intersecting.empty:
            # Simpan versi geojson untuk inspeksi visual
            intersecting.to_file(geojson_path, driver="GeoJSON")
            pixel_count, coverage = rasterize_buildings(tile, gdf, str(mask_path), args.tile_size)
        else:
            pixel_count, coverage = 0, 0.0
            mask = np.zeros((args.tile_size, args.tile_size), dtype=np.uint8)
            Image.fromarray(mask).save(str(mask_path))
            
        bounds = mercantile.bounds(tile)
        pairs.append({
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "tile_x": tile.x,
            "tile_y": tile.y,
            "tile_z": tile.z,
            "lon_min": bounds.west,
            "lat_min": bounds.south,
            "lon_max": bounds.east,
            "lat_max": bounds.north,
            "building_pixel_count": pixel_count,
            "building_coverage_pct": coverage
        })
            
        # Hormati rate limit
        time.sleep(0.1)
        
    # Simpan hasil
    # Simpan hasil dengan retry logic (mengatasi PermissionError concurrent)
    if pairs:
        df_new = pd.DataFrame(pairs)
        
        for attempt in range(5):
            try:
                if csv_path.exists():
                    df_new.to_csv(csv_path, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(csv_path, index=False)
                break
            except PermissionError:
                time.sleep(1 + (attempt * 0.5))
                
    if failed_tiles:
        for attempt in range(5):
            try:
                with open(out_dir / "failed_tiles.txt", "a") as f:
                    for t in failed_tiles:
                        f.write(f"{t.z},{t.x},{t.y}\n")
                break
            except PermissionError:
                time.sleep(1 + (attempt * 0.5))
                
    print("\n Proses Selesai!")
    print(f"   Tile berhasil: {len(pairs)}")
    print(f"   Tile gagal   : {len(failed_tiles)}")

if __name__ == "__main__":
    main()
