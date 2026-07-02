# Web UI — Building Segmentation Dataset Builder
## Prompt untuk Claude Code

---

## Konteks

Saya punya script `build_dataset.py` yang men-download tile OpenStreetMap dan membuat binary mask dari Google Open Buildings untuk training model segmentasi bangunan.

Saya ingin Web UI yang bisa mengontrol dan memonitor script tersebut dari browser, tanpa harus buka terminal.

---

## Stack yang Digunakan

- **Backend**: FastAPI + uvicorn
- **Frontend**: HTML + Vanilla JS (satu file `index.html`, tidak pakai framework)
- **Komunikasi realtime**: Server-Sent Events (SSE) untuk streaming log ke browser
- **Map**: Leaflet.js (via CDN) untuk pilih area AOI
- **Chart**: Chart.js (via CDN) untuk statistik dataset

Install:
```bash
pip install fastapi uvicorn python-multipart aiofiles
```

---

## File yang Harus Dibuat

```
project/
├── build_dataset.py       ← script yang sudah ada (jangan diubah)
├── app.py                 ← FastAPI backend
├── templates/
│   └── index.html         ← Frontend UI
└── static/
    └── style.css          ← CSS tambahan jika perlu
```

---

## Fitur UI yang Dibutuhkan

### Panel Kiri — Konfigurasi

**1. Pilih Area (AOI)**
- Tampilkan peta Leaflet dengan basemap OpenStreetMap
- Default center: Indonesia (lat -2.5, lon 118, zoom 5)
- User bisa **gambar rectangle** di peta untuk pilih AOI
- Koordinat bbox otomatis muncul di field input (lon_min, lat_min, lon_max, lat_max)
- Bisa juga input koordinat manual
- Tampilkan estimasi jumlah tile berdasarkan zoom level yang dipilih

**2. Parameter**
- Zoom Level: slider 15-18, default 17, tampilkan label "~Xm/pixel" yang update dinamis
  - Zoom 15 = ~4m/pixel
  - Zoom 16 = ~2m/pixel  
  - Zoom 17 = ~1m/pixel
  - Zoom 18 = ~0.5m/pixel
- Min Confidence: slider 0.5-1.0, default 0.75, step 0.05
- Output Directory: text input, default "dataset"
- Tile Size: dropdown 256 / 512, default 256

**3. Tombol Aksi**
- **▶ Mulai Download** — jalankan build_dataset.py dengan parameter dari form
- **⏹ Stop** — hentikan proses yang sedang berjalan
- **✓ Validasi Dataset** — jalankan validasi
- **👁 Preview Sample** — generate dan tampilkan preview

---

### Panel Kanan Atas — Log Terminal

- Text area yang menampilkan output realtime dari script via SSE
- Auto-scroll ke bawah
- Warna berbeda untuk: info (putih), sukses (hijau), warning (kuning), error (merah)
- Tombol "Clear Log"
- Indikator status: IDLE / RUNNING / DONE / ERROR (dengan warna)

---

### Panel Kanan Bawah — Statistik & Preview

**Tab 1: Statistik**
- Total tile diproses
- Tile berhasil / gagal
- Jumlah bangunan ditemukan
- Progress bar overall
- Estimasi waktu selesai (ETA)
- Bar chart: tile dengan bangunan vs tanpa bangunan

**Tab 2: Preview Pairs**
- Grid 2x3 (atau lebih) menampilkan sample pasangan tile + mask
- Setiap pair ditampilkan berdampingan: [🖼 Tile OSM] [⬛ Mask]
- Tombol "Refresh Preview" untuk load sample baru
- Klik gambar untuk zoom

**Tab 3: File Manager**
- List file dalam output directory
- Tampilkan ukuran folder total
- Tombol download pairs.csv
- Tombol hapus dataset (dengan konfirmasi)

---

## Backend API Endpoints (app.py)

```
POST /api/start
  Body: { bbox, zoom, output_dir, min_confidence, tile_size }
  → Jalankan build_dataset.py sebagai subprocess
  → Return: { job_id, status }

POST /api/stop
  → Kill subprocess yang berjalan
  → Return: { status }

GET  /api/status
  → Return: { status, progress, total_tiles, done_tiles, failed_tiles }

GET  /api/stream
  → SSE endpoint, stream stdout dari subprocess realtime

POST /api/validate
  → Jalankan validasi dataset
  → Return: { total_pairs, has_buildings, avg_coverage, errors }

GET  /api/preview
  → Return: list of { image_path, mask_path } untuk N sample acak
  → Encode gambar sebagai base64

GET  /api/files
  → List file dalam output directory
  → Return: { files, total_size_mb }

GET  /api/download/pairs.csv
  → Download file pairs.csv

DELETE /api/dataset
  → Hapus seluruh output directory (dengan konfirmasi di frontend)
```

---

## Desain UI

**Visual direction:**
- Dark theme — background #0f1117, panel #1a1d27
- Accent warna hijau terminal #00ff88 untuk elemen aktif dan progress
- Font: monospace untuk log terminal, sans-serif untuk UI umum
- Layout: dua kolom (30% kiri konfigurasi, 70% kanan log + stats)
- Peta Leaflet mengisi seluruh area panel kiri bagian atas
- Tidak ada animasi berlebihan — clean dan fungsional

**Nuansa keseluruhan:** seperti tool internal data engineer, bukan consumer app.

---

## Aturan Penting

1. **Satu proses saja** — kalau sudah ada job running, tombol Start di-disable
2. **State persistent** — kalau browser di-refresh, status job masih terbaca dari backend
3. **Error handling** — kalau script crash, tampilkan error message yang jelas di log
4. **Subprocess management** — pastikan subprocess di-kill dengan benar saat Stop atau server shutdown
5. **CORS** — enable CORS untuk development (localhost)
6. **Semua dalam satu command** — user cukup jalankan `python app.py` lalu buka `http://localhost:8000`

---

## Contoh Cara Jalankan

```bash
# Install dependencies
pip install fastapi uvicorn python-multipart aiofiles requests geopandas shapely pandas tqdm pillow matplotlib mercantile rasterio numpy

# Jalankan server
python app.py

# Buka browser
# http://localhost:8000
```

---

## Catatan Tambahan

- Leaflet draw plugin untuk rectangle selection: gunakan CDN `leaflet-draw`
- SSE harus handle reconnect otomatis jika koneksi putus
- Preview gambar di-encode base64 dan ditampilkan sebagai `<img src="data:image/png;base64,...">` 
- Estimasi tile count: gunakan rumus mercantile untuk hitung jumlah tile dalam bbox sebelum download dimulai
- Pastikan `app.py` bisa detect apakah `build_dataset.py` ada di folder yang sama, jika tidak tampilkan error yang informatif