"""
Script migrasi: Re-generate mask dari FGB source dengan skema 3-class baru.
Tidak perlu download ulang citra satelit — hanya mask yang di-overwrite.

Kelas mask baru:
  0 = background (hitam)
  1 = building interior (abu-abu gelap, nilai 1)
  2 = building border (abu-abu terang, nilai 2)
"""
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import mercantile
from PIL import Image
from pathlib import Path
from shapely.geometry import box
from tqdm import tqdm
import rasterio.features
import config


def rasterize_buildings_3class(tile, gdf, output_path, tile_size):
    """Generate mask 3-class: 0=background, 1=interior, 2=border."""
    bounds = mercantile.bounds(tile)
    tile_poly = box(bounds.west, bounds.south, bounds.east, bounds.north)
    
    intersecting = gdf[gdf.intersects(tile_poly)].copy()
    
    if intersecting.empty:
        mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        Image.fromarray(mask).save(output_path)
        return 0, 0.0
    
    xy_bnds = mercantile.xy_bounds(tile)
    
    def to_pixel_coords(geom):
        def transform_point(lon, lat):
            mx, my = mercantile.xy(lon, lat)
            px = round(((mx - xy_bnds.left) / (xy_bnds.right - xy_bnds.left)) * tile_size)
            py = round(((xy_bnds.top - my) / (xy_bnds.top - xy_bnds.bottom)) * tile_size)
            return px, py
        from shapely.ops import transform
        return transform(lambda x, y, z=None: transform_point(x, y), geom)
    
    local_geoms = intersecting['geometry'].apply(to_pixel_coords)
    
    border_px = config.BORDER_WIDTH_PX
    min_area = config.MIN_BORDER_AREA
    
    # Step 1: Full building mask
    full_shapes = [(geom, 1) for geom in local_geoms if not geom.is_empty]
    if not full_shapes:
        mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        Image.fromarray(mask).save(output_path)
        return 0, 0.0
    
    full_mask = rasterio.features.rasterize(
        full_shapes, out_shape=(tile_size, tile_size), fill=0, dtype=np.uint8
    )
    
    # Step 2: Interior (shrunk) mask
    interior_shapes = []
    for geom in local_geoms:
        if geom.is_empty:
            continue
        if geom.area < min_area:
            interior_shapes.append((geom, 1))
        else:
            shrunk = geom.buffer(-border_px)
            if not shrunk.is_empty and shrunk.is_valid and shrunk.area > 0:
                interior_shapes.append((shrunk, 1))
    
    interior_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    if interior_shapes:
        interior_mask = rasterio.features.rasterize(
            interior_shapes, out_shape=(tile_size, tile_size), fill=0, dtype=np.uint8
        )
    
    # Step 3: Combine → 3-class
    mask_3class = np.zeros((tile_size, tile_size), dtype=np.uint8)
    mask_3class[full_mask == 1] = 2       # All building area → border first
    mask_3class[interior_mask == 1] = 1   # Overwrite interior on top
    
    pixel_count = np.sum(full_mask)
    coverage = (pixel_count / (tile_size * tile_size)) * 100
    
    Image.fromarray(mask_3class).save(output_path)
    return pixel_count, coverage


def main():
    parser = argparse.ArgumentParser(description="Re-generate mask dengan skema 3-class")
    parser.add_argument("--max-tiles", type=int, default=None, help="Batasi jumlah tile")
    parser.add_argument("--tile-size", type=int, default=config.TILE_SIZE)
    args = parser.parse_args()
    
    out_dir = Path(config.DATASET_DIR)
    csv_path = out_dir / "pairs.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} tidak ditemukan! Jalankan 01_prepare_dataset.py dulu.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Ditemukan {len(df)} tile di pairs.csv")
    
    if args.max_tiles:
        df = df.head(args.max_tiles)
        print(f"Dibatasi ke {len(df)} tile")
    
    # Hitung expanded bbox dari semua tiles
    all_bboxes = []
    for _, row in df.iterrows():
        all_bboxes.append((row['lon_min'], row['lat_min'], row['lon_max'], row['lat_max']))
    
    expanded_bbox = (
        min(b[0] for b in all_bboxes),
        min(b[1] for b in all_bboxes),
        max(b[2] for b in all_bboxes),
        max(b[3] for b in all_bboxes),
    )
    
    print(f"Loading FGB untuk bbox: {expanded_bbox}")
    gdf = gpd.read_file(config.FGB_PATH, bbox=expanded_bbox)
    gdf = gdf[~gdf.is_empty]
    print(f"Ditemukan {len(gdf)} bangunan")
    
    # Re-generate setiap mask
    stats = {'bg': 0, 'interior': 0, 'border': 0, 'total': 0}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Regenerate Mask"):
        tile = mercantile.Tile(x=int(row['tile_x']), y=int(row['tile_y']), z=int(row['tile_z']))
        mask_path = row['mask_path']
        
        rasterize_buildings_3class(tile, gdf, mask_path, args.tile_size)
        
        # Statistik
        mask = np.array(Image.open(mask_path))
        stats['bg'] += np.sum(mask == 0)
        stats['interior'] += np.sum(mask == 1)
        stats['border'] += np.sum(mask == 2)
        stats['total'] += mask.size
    
    # Juga regenerate mask di subfolder train/val/test
    for split in ['train', 'val', 'test']:
        split_csv = out_dir / f"{split}.csv"
        split_mask_dir = out_dir / split / "masks"
        if split_csv.exists() and split_mask_dir.exists():
            split_df = pd.read_csv(split_csv)
            print(f"\nUpdating {split} masks ({len(split_df)} files)...")
            for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"  {split}"):
                src_mask = Path(row['mask_path'])
                dst_mask = split_mask_dir / src_mask.name
                if src_mask.exists():
                    import shutil
                    shutil.copy(src_mask, dst_mask)
    
    # Report
    print(f"\n{'='*50}")
    print("Statistik Mask 3-Class:")
    print(f"  Background (0): {stats['bg']:>12,} piksel ({stats['bg']/stats['total']*100:.1f}%)")
    print(f"  Interior   (1): {stats['interior']:>12,} piksel ({stats['interior']/stats['total']*100:.1f}%)")
    print(f"  Border     (2): {stats['border']:>12,} piksel ({stats['border']/stats['total']*100:.1f}%)")
    print(f"  Total         : {stats['total']:>12,} piksel")
    
    # Hitung inverse-frequency weights
    counts = np.array([stats['bg'], stats['interior'], stats['border']], dtype=np.float64)
    counts = np.maximum(counts, 1)  # Avoid division by zero
    inv_freq = stats['total'] / (3 * counts)
    # Normalize sehingga background = 1.0
    inv_freq = inv_freq / inv_freq[0]
    
    print(f"\nRecommended class_weights (inverse-frequency):")
    print(f"  [bg={inv_freq[0]:.2f}, interior={inv_freq[1]:.2f}, border={inv_freq[2]:.2f}]")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
