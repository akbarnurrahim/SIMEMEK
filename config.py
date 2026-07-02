import os

# ─── PATH ───
FGB_PATH = "IDN.fgb"
DATASET_DIR = "dataset"
MODEL_DIR = "models"

# ─── AREA OF INTEREST ───
AOI_BBOX = (107.4, -7.1, 107.8, -6.8)  # Default: Bandung

PROVINCE_BBOX = {
    "jakarta":      (106.6, -6.4, 107.0, -6.0),
    "bandung":      (107.4, -7.1, 107.8, -6.8),
    "surabaya":     (112.5, -7.4, 112.9, -7.1),
    "yogyakarta":   (110.2, -8.0, 110.6, -7.7),
    "semarang":     (110.3, -7.1, 110.6, -6.9),
    "jawa_barat":   (106.0, -7.8, 108.8, -5.9),
    "jawa_tengah":  (108.8, -8.2, 111.7, -6.5),
    "jawa_timur":   (111.0, -8.8, 114.4, -7.0),
}

# ─── TILE SETTINGS ───
ZOOM_LEVEL = 19
TILE_SIZE  = 256   # ESRI default adalah 256x256. Jangan di-stretch ke 512 agar tidak buram!
MIN_CONFIDENCE = 0.75

# ─── CITRA SATELIT ───
# ESRI World Imagery — HD, gratis, tanpa autentikasi, resolusi ~0.3m/px di zoom 17
# Alternatif lebih jelas dari Google Maps, dipakai oleh banyak riset AI pemetaan
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# ─── DATASET SPLIT ───
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
MIN_BUILDING_PIXELS = 100

# ─── SEGFORMER CONFIG ───
MODEL_NAME  = "nvidia/mit-b2"   # Lebih akurat dari b0, masih aman untuk GPU <8GB
NUM_CLASSES = 3                 # 0 = background, 1 = building interior, 2 = building border
IMAGE_SIZE  = 256  # Harus sama dengan TILE_SIZE agar pixel perfect
BATCH_SIZE  = 4
EPOCHS      = 50
LR          = 2e-5              # Diturunkan dari 6e-5 agar stabil
WEIGHT_DECAY = 0.01

# ─── 3-CLASS MASK SETTINGS ───
BORDER_WIDTH_PX = 4             # Lebar border (piksel) antar bangunan berdempetan
MIN_BORDER_AREA = 100           # Bangunan < area ini tidak diberi border (full interior)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─── AUGMENTASI ───
AUGMENT = True
