# Building Segmentation Dataset Pipeline
## Prompt untuk Claude Code

---

## Konteks Proyek

Saya ingin membuat dataset untuk training model segmentasi bangunan (U-Net atau sejenisnya).
Dataset terdiri dari pasangan:
- **Input**: tile citra satelit dari OpenStreetMap (PNG)
- **Label**: binary mask bangunan dari Google Open Buildings v3 (PNG)

Semua tile dan mask harus perfectly aligned (area, resolusi, dan ukuran pixel sama persis).

---

## Yang Harus Dibuat

Buat satu script Python bernama `build_dataset.py` dengan pipeline lengkap berikut:

---

### 1. Konfigurasi (di bagian atas script)

```python
# Area of Interest (bounding box)
# Contoh default: sebagian Jawa Barat
AOI_BBOX = (106.7, -6.3, 107.0, -6.0)  # (lon_min, lat_min, lon_max, lat_max)

# Zoom level tile (17 = ~1m/pixel, 18 = ~0.5m/pixel)
# Rekomendasi: 17 untuk keseimbangan detail dan jumlah tile
ZOOM_LEVEL = 17

# Ukuran output tile dan mask (pixel)
TILE_SIZE = 256  # 256x256 pixel

# Minimum confidence Open Buildings
MIN_CONFIDENCE = 0.75

# URL tiles Open Buildings
OB_TILES_URL = "https://sites.research.google/open-buildings/tiles.geojson"

# Output directory
OUTPUT_DIR = "dataset"
# Struktur:
# dataset/
#   images/   → tile PNG (input model)
#   masks/    → mask PNG (label model)
#   pairs.csv → mapping nama file tile → mask
```

---

### 2. Fungsi yang Harus Ada

#### `get_tiles_for_bbox(bbox, zoom)`
- Hitung semua tile XYZ yang masuk dalam bounding box pada zoom level tertentu
- Gunakan rumus konversi lat/lon → tile XYZ (mercantile atau manual)
- Return: list of (x, y, z, tile_bbox) dimana tile_bbox adalah (lon_min, lat_min, lon_max, lat_max) tile tersebut

#### `download_osm_tile(x, y, z, output_path)`
- Download tile PNG dari OpenStreetMap Tile Server:
  `https://tile.openstreetmap.org/{z}/{x}/{y}.png`
- Tambahkan User-Agent header (wajib oleh OSM policy): `User-Agent: BuildingDatasetBuilder/1.0`
- Resize ke TILE_SIZE x TILE_SIZE jika perlu
- Simpan sebagai PNG
- Tambahkan delay kecil (0.1 detik) antar request untuk menghormati OSM rate limit
- Return: True jika sukses, False jika gagal

#### `get_ob_buildings_for_bbox(bbox)`
- Download daftar tile Open Buildings dari OB_TILES_URL
- Filter tile yang beririsan dengan bbox
- Download tile CSV gzip yang relevan
- Filter baris berdasarkan bbox dan MIN_CONFIDENCE
- Parse kolom `geometry` (WKT) ke Shapely geometry
- Return: GeoDataFrame dengan CRS EPSG:4326

#### `rasterize_buildings(gdf, tile_bbox, tile_size)`
- Rasterize polygon bangunan dari GDF ke binary mask numpy array
- tile_bbox: (lon_min, lat_min, lon_max, lat_max) area tile
- tile_size: ukuran output dalam pixel (default 256)
- Pixel bernilai 255 jika ada bangunan, 0 jika tidak ada
- Return: numpy array shape (tile_size, tile_size) dtype uint8

#### `save_mask(mask_array, output_path)`
- Simpan numpy array sebagai grayscale PNG
- Return: path file yang disimpan

#### `build_dataset(bbox, zoom, output_dir)`
- Fungsi utama yang memanggil semua fungsi di atas
- Loop semua tile dalam bbox:
  1. Download OSM tile → simpan ke `output_dir/images/tile_{z}_{x}_{y}.png`
  2. Rasterize mask untuk area tile tersebut → simpan ke `output_dir/masks/mask_{z}_{x}_{y}.png`
  3. Catat ke pairs.csv: `image_path, mask_path, tile_x, tile_y, tile_z, lon_min, lat_min, lon_max, lat_max`
- Tampilkan progress bar dengan tqdm
- Skip tile yang sudah ada (resume-able)
- Simpan pairs.csv di akhir

---

### 3. Validasi Dataset

Tambahkan fungsi `validate_dataset(output_dir)` yang:
- Cek setiap pair di pairs.csv: apakah kedua file ada?
- Cek apakah ukuran tile dan mask sama (harus 256x256)
- Hitung statistik:
  - Total pairs
  - Jumlah mask yang memiliki bangunan (mask tidak semua hitam)
  - Persentase tile yang punya bangunan
  - Rata-rata coverage bangunan per tile (% pixel putih)
- Print summary ke terminal

---

### 4. Visualisasi Sample

Tambahkan fungsi `visualize_samples(output_dir, n=5)` yang:
- Ambil n sample acak dari pairs.csv
- Untuk setiap sample, tampilkan tile dan mask berdampingan dengan matplotlib
- Simpan output ke `output_dir/sample_preview.png`

---

### 5. Entry Point

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float,
                        default=list(AOI_BBOX),
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    parser.add_argument("--zoom", type=int, default=ZOOM_LEVEL)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    if args.validate:
        validate_dataset(args.output)
    elif args.preview:
        visualize_samples(args.output)
    else:
        build_dataset(tuple(args.bbox), args.zoom, args.output)
```

---

## Dependencies

Script harus bisa diinstall dengan:
```bash
pip install requests geopandas shapely pandas tqdm pillow matplotlib mercantile rasterio numpy
```

Tambahkan di bagian atas script komentar install command ini.

---

## Aturan Penting

1. **Jangan hardcode** path atau koordinat selain di bagian KONFIGURASI
2. **Handle error** dengan graceful — kalau satu tile gagal download, skip dan lanjut
3. **Resume-able** — kalau script dihentikan di tengah, bisa dilanjut tanpa download ulang dari awal
4. **Rate limiting** — tambahkan delay antar request OSM (minimal 0.1 detik)
5. **Logging** — print progress yang informatif, termasuk berapa tile berhasil/gagal
6. **CRS consistency** — semua operasi spatial harus dalam EPSG:4326 kecuali saat rasterize gunakan Web Mercator (EPSG:3857) untuk akurasi pixel

---

## Contoh Penggunaan yang Diharapkan

```bash
# Download dataset untuk area kecil (test)
python build_dataset.py --bbox 106.7 -6.3 107.0 -6.0 --zoom 17 --output dataset

# Validasi hasil
python build_dataset.py --validate --output dataset

# Lihat preview
python build_dataset.py --preview --output dataset
```

---

## Output yang Diharapkan di Terminal

```
============================================================
  Building Segmentation Dataset Builder
============================================================
  AOI Bbox  : (106.7, -6.3, 107.0, -6.0)
  Zoom Level: 17
  Output Dir: dataset/
============================================================

📡 Mengambil building data dari Open Buildings...
✅ 1,234 bangunan ditemukan dalam AOI

🗺️  Menghitung tile yang dibutuhkan...
✅ 847 tile pada zoom level 17

Downloading tiles: 100%|████████████| 847/847 [12:34<00:00, tile/s]
  ✓ 831 tile berhasil
  ✗ 16 tile gagal (dicatat di failed_tiles.txt)

💾 Menyimpan pairs.csv...
✅ Dataset selesai! 831 pasang image-mask tersimpan di dataset/
```