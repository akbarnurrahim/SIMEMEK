import os
import sys
import asyncio
import subprocess
import threading
import base64
import shutil
from pathlib import Path
from fastapi import FastAPI, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
from datetime import datetime

# Setup FastAPI
app = FastAPI(title="Dataset Builder UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Buat direktori untuk static dan templates jika belum ada
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# STATE & BACKGROUND PROCESS
# ============================================================
current_process = None
log_buffer = []

def read_stream_sync(stream):
    global log_buffer
    while True:
        # Baca per karakter atau chunk agar \r dari tqdm tertangkap
        chunk = stream.read(512)
        if chunk:
            log_buffer.append(chunk.decode('utf-8', errors='replace'))
        else:
            break

# ============================================================
# MODELS
# ============================================================
class StartRequest(BaseModel):
    bbox: list[float]
    zoom: int
    output_dir: str
    min_confidence: float
    tile_size: int
    username: str = "Unknown"

# ============================================================
# WEBSOCKET MANAGER
# ============================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.boxes: dict = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        await websocket.send_json({"type": "init", "boxes": self.boxes, "user_count": len(self.active_connections)})
        await self.broadcast({"type": "user_count", "user_count": len(self.active_connections)})

    def disconnect(self, websocket: WebSocket, client_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if client_id in self.boxes:
            del self.boxes[client_id]

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(request=request, name="landing.html", context={"request": request})

@app.get("/app", response_class=HTMLResponse)
async def index(request: Request):
    if not os.path.exists("01_prepare_dataset.py"):
        return HTMLResponse("<h1>Error: 01_prepare_dataset.py tidak ditemukan di direktori ini.</h1>", status_code=500)
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "update_box":
                manager.boxes[client_id] = {
                    "username": data.get("username", "Unknown"),
                    "bounds": data.get("bounds")
                }
                await manager.broadcast({
                    "type": "update_box",
                    "client_id": client_id,
                    "data": manager.boxes[client_id]
                })
            elif msg_type == "delete_box":
                if client_id in manager.boxes:
                    del manager.boxes[client_id]
                await manager.broadcast({
                    "type": "delete_box",
                    "client_id": client_id
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        await manager.broadcast({
            "type": "delete_box",
            "client_id": client_id
        })
        await manager.broadcast({
            "type": "user_count",
            "user_count": len(manager.active_connections)
        })

active_processes = {}
history_lock = asyncio.Lock()
zip_status = {"state": "idle", "count": 0}

def make_zip_task():
    global zip_status
    zip_status["state"] = "processing"
    try:
        count = 0
        img_dir = "dataset/images"
        if os.path.exists(img_dir):
            count = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
        shutil.make_archive("dataset_export", 'zip', "dataset")
        zip_status["count"] = count
        zip_status["state"] = "ready"
    except Exception as e:
        zip_status["state"] = "error"

@app.post("/api/start")
async def start_dataset_build(req: StartRequest):
    global active_processes
    user = req.username
    
    if user in active_processes and active_processes[user].poll() is None:
        return JSONResponse({"status": "error", "message": "Proses sudah berjalan!"}, status_code=400)
        
    cmd = [
        sys.executable, "01_prepare_dataset.py",
        "--bbox", str(req.bbox[0]), str(req.bbox[1]), str(req.bbox[2]), str(req.bbox[3]),
        "--zoom", str(req.zoom),
        "--output", req.output_dir,
        "--min_conf", str(req.min_confidence),
        "--tile_size", str(req.tile_size),
        "--force-regenerate",
        "--user", user
    ]
    
    active_processes[user] = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    async with history_lock:
        history_file = "history.json"
        history = []
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                try:
                    history = json.load(f)
                except:
                    pass
        
        new_item = {
            "timestamp": datetime.now().isoformat(),
            "lon_min": req.bbox[0],
            "lat_min": req.bbox[1],
            "lon_max": req.bbox[2],
            "lat_max": req.bbox[3],
            "username": user
        }
        history.append(new_item)
        with open(history_file, "w") as f:
            json.dump(history, f)
        
    await manager.broadcast({
        "type": "new_history",
        "data": new_item
    })
    
    return {"status": "started", "message": "Proses dimulai"}

@app.post("/api/stop")
async def stop_process(request: Request):
    global active_processes
    data = await request.json()
    user = data.get("username", "GUEST")
    
    if user in active_processes and active_processes[user].poll() is None:
        active_processes[user].terminate()
        return {"status": "stopped"}
    return {"status": "not_running"}

@app.get("/api/status")
async def get_status(user: str = "GUEST"):
    global active_processes
    current_process = active_processes.get(user)
    
    status_str = "IDLE"
    if current_process:
        if current_process.poll() is None:
            status_str = "RUNNING"
        else:
            status_str = "DONE" if current_process.returncode == 0 else "ERROR"
            
    total_tiles = 0
    done_tiles = 0
    failed_tiles = 0
    
    progress_file = f"dataset/progress_{user}.json"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                data = json.load(f)
                total_tiles = data.get("total", 0)
                done_tiles = data.get("done", 0)
                failed_tiles = data.get("failed", 0)
        except:
            pass
            
    total_db = 0
    img_dir = "dataset/images"
    if os.path.exists(img_dir):
        total_db = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
            
    return {
        "status": status_str,
        "total_tiles": total_tiles,
        "done_tiles": done_tiles,
        "failed_tiles": failed_tiles,
        "total_db": total_db
    }

@app.get("/api/history")
async def get_history():
    async with history_lock:
        history_file = "history.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                try:
                    return json.load(f)
                except:
                    pass
        return []

@app.post("/api/prepare_download")
async def prepare_download(background_tasks: BackgroundTasks):
    global zip_status
    if not os.path.exists("dataset"):
        return JSONResponse({"status": "error", "message": "Dataset tidak ditemukan!"}, status_code=404)
        
    if zip_status["state"] == "processing":
        return {"status": "processing"}
        
    background_tasks.add_task(make_zip_task)
    return {"status": "started"}

@app.get("/api/download_status")
async def get_download_status():
    global zip_status
    return zip_status

@app.get("/api/download_file")
async def download_file():
    if not os.path.exists("dataset_export.zip"):
        return JSONResponse({"status": "error", "message": "File ZIP belum siap!"}, status_code=404)
        
    return FileResponse(
        path="dataset_export.zip",
        media_type="application/zip",
        filename="SIMEMEK_Dataset.zip"
    )

@app.get("/api/stream")
async def stream_logs(request: Request, user: str = "GUEST"):
    """Server-Sent Events endpoint untuk stream stdout."""
    global active_processes
    
    async def event_generator():
        current_process = active_processes.get(user)
        if current_process is None:
            yield "data: Proses belum berjalan\n\n"
            return
            
        while True:
            if await request.is_disconnected():
                break
            
            line = current_process.stdout.readline()
            if not line:
                if current_process.poll() is not None:
                    break
                await asyncio.sleep(0.1)
                continue
                
            yield f"data: {line.strip()}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/validate")
async def run_validation():
    # Jalankan validasi menggunakan subprocess
    cmd = [sys.executable, "01_prepare_dataset.py", "--validate", "--output", "dataset"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    def _run_cmd():
        import subprocess
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        
    proc = await asyncio.to_thread(_run_cmd)
    return {"status": "done", "log": proc.stdout.decode('utf-8', errors='replace')}

@app.get("/api/preview")
async def get_preview():
    # Kembalikan gambar dalam bentuk base64
    pairs_csv = Path("dataset/pairs.csv")
    if not pairs_csv.exists():
        return {"samples": []}
        
    df = pd.read_csv(pairs_csv)
    if len(df) == 0:
        return {"samples": []}
        
    samples = df.sample(min(6, len(df))) # ambil maksimal 6 sampel (grid 2x3)
    
    result = []
    for _, row in samples.iterrows():
        img_path = Path("dataset") / row['image_path']
        mask_path = Path("dataset") / row['mask_path']
        
        if img_path.exists() and mask_path.exists():
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            with open(mask_path, "rb") as f:
                mask_b64 = base64.b64encode(f.read()).decode()
                
            result.append({
                "image": f"data:image/png;base64,{img_b64}",
                "mask": f"data:image/png;base64,{mask_b64}",
                "info": f"Tile: {row['tile_z']}/{row['tile_x']}/{row['tile_y']}"
            })
            
    return {"samples": result}

@app.get("/api/files")
async def list_files():
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        return {"files": [], "total_size_mb": 0}
        
    def get_size(p):
        return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
        
    total_size = get_size(dataset_dir) / (1024 * 1024)
    
    # Ambil struktur folder dasar
    files_info = []
    for path in dataset_dir.glob('*'):
        if path.is_file():
            files_info.append({"name": path.name, "type": "file", "size_kb": path.stat().st_size / 1024})
        elif path.is_dir():
            count = len(list(path.glob('*')))
            files_info.append({"name": path.name, "type": "dir", "count": count})
            
    return {"files": files_info, "total_size_mb": total_size}

@app.get("/api/download/pairs.csv")
async def download_csv():
    csv_path = "dataset/pairs.csv"
    if os.path.exists(csv_path):
        return FileResponse(path=csv_path, filename="pairs.csv", media_type="text/csv")
    return JSONResponse({"error": "File tidak ditemukan"}, status_code=404)

@app.delete("/api/dataset")
async def delete_dataset():
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")
        return {"status": "deleted"}
    return {"status": "not_found"}

def get_level_info(exp):
    level = 1
    req = 500
    total_needed_for_next = req
    current_tier_base = 0
    
    while exp >= total_needed_for_next:
        level += 1
        current_tier_base = total_needed_for_next
        req = int(req * 1.5)
        total_needed_for_next += req
        
    exp_in_level = exp - current_tier_base
    return {
        "level": level,
        "exp_total": exp,
        "exp_current": exp_in_level,
        "exp_needed": req
    }

@app.get("/api/user_info")
async def get_user_info(user: str = "GUEST"):
    users_file = "users.json"
    user_exp = 0
    if os.path.exists(users_file):
        try:
            with open(users_file, "r") as f:
                data = json.load(f)
                if user in data:
                    user_exp = data[user].get("exp", 0)
        except:
            pass
            
    return get_level_info(user_exp)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
