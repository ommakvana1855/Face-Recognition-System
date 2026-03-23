"""
Search-Based Object Detection — HLS + YOLO overlay
FastAPI backend: ffmpeg RTSP→HLS for smooth video, YOLO boxes via WebSocket overlay
"""

import asyncio
import base64
import io
import logging
import os
import sys
import tempfile
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s › %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("search_detection.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SearchDetection")

MAX_CAMERAS = 9

# HLS output directory — served as static files
HLS_DIR = Path("hls_streams")
HLS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RTSP Stream Manager
# ═══════════════════════════════════════════════════════════════════════════════
class RTSPStreamManager:
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.url: str = ""
        self.lock = threading.Lock()
        self.error: str = ""
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.label: str = f"Camera {camera_id + 1}"

    def connect(self, rtsp_url: str, label: str = "") -> bool:
        self.disconnect()
        logger.info(f"[Cam {self.camera_id}] Connecting → {rtsp_url}")
        self.url = rtsp_url
        self.label = label or f"Camera {self.camera_id + 1}"
        self.error = ""

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.error = f"Cannot open stream: {rtsp_url}"
            logger.error(self.error)
            cap.release()
            return False

        self.cap = cap
        self.fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"[Cam {self.camera_id}] Connected {self.width}×{self.height} @ {self.fps:.1f}fps")

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        fails = 0
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.error = "Stream disconnected"; self.running = False; break
            ret, frame = self.cap.read()
            if not ret:
                fails += 1
                if fails >= 30:
                    logger.warning(f"[Cam {self.camera_id}] Reconnecting…")
                    self._attempt_reconnect(); fails = 0
                time.sleep(0.05); continue
            fails = 0
            with self.lock:
                self.latest_frame = (frame.copy(), time.time())

    def _attempt_reconnect(self):
        if self.cap: self.cap.release()
        time.sleep(2)
        if self.url:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.cap = cap; self.error = ""
                logger.info(f"[Cam {self.camera_id}] Reconnected ✅")
            else:
                self.error = "Reconnect failed"

    def get_frame(self) -> Optional[tuple]:
        """Returns (frame, timestamp) or None."""
        with self.lock:
            if self.latest_frame is None:
                return None
            frame, ts = self.latest_frame
            return (frame.copy(), ts)

    def disconnect(self):
        self.running = False
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=3)
        if self.cap: self.cap.release(); self.cap = None
        self.latest_frame = None; self.url = ""
        logger.info(f"[Cam {self.camera_id}] Disconnected")

    @property
    def is_connected(self):
        return self.running and self.cap is not None and self.cap.isOpened()

    def to_dict(self):
        return dict(camera_id=self.camera_id, connected=self.is_connected,
                    url=self.url, label=self.label, error=self.error,
                    width=self.width, height=self.height, fps=self.fps)


class MultiCameraManager:
    def __init__(self, max_cameras=MAX_CAMERAS):
        self.cameras: Dict[int, RTSPStreamManager] = {
            i: RTSPStreamManager(i) for i in range(max_cameras)
        }
    def get_camera(self, cid): return self.cameras.get(cid)
    def get_all_status(self): return [c.to_dict() for c in self.cameras.values()]
    def disconnect_all(self):
        for c in self.cameras.values():
            if c.is_connected: c.disconnect()


# ═══════════════════════════════════════════════════════════════════════════════
# HLS via ffmpeg
# ═══════════════════════════════════════════════════════════════════════════════
_cam_ffmpeg: Dict[int, subprocess.Popen] = {}


def start_hls(camera_id: int, rtsp_url: str):
    stop_hls(camera_id)

    out_dir = HLS_DIR / str(camera_id)
    out_dir.mkdir(exist_ok=True)

    for f in out_dir.glob("*.ts"):   f.unlink(missing_ok=True)
    for f in out_dir.glob("*.m3u8"): f.unlink(missing_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-fflags", "nobuffer",           # ← add: disable input buffering
        "-flags", "low_delay",           # ← add: low delay mode
        "-i", rtsp_url,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", "28",
        "-g", "15",                      # ← add: keyframe every 15 frames (0.6s at 25fps)
        "-sc_threshold", "0",            # ← add: disable scene change detection
        "-an",
        "-f", "hls",
        "-hls_time", "0.5",              # ← change: was 1, now 0.5s chunks
        "-hls_list_size", "2",           # ← change: was 3, now keep only 2 chunks
        "-hls_flags", "delete_segments+append_list+omit_endlist",
        "-hls_segment_filename", str(out_dir / "seg%05d.ts"),
        str(out_dir / "stream.m3u8"),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _cam_ffmpeg[camera_id] = proc
    logger.info(f"[Cam {camera_id}] HLS ffmpeg started (pid {proc.pid})")


def stop_hls(camera_id: int):
    proc = _cam_ffmpeg.pop(camera_id, None)
    if proc:
        proc.terminate()
        try:   proc.wait(timeout=3)
        except subprocess.TimeoutExpired: proc.kill()
        logger.info(f"[Cam {camera_id}] HLS stopped")


# ═══════════════════════════════════════════════════════════════════════════════
# Face Recognition
# ═══════════════════════════════════════════════════════════════════════════════
def load_single_face_encoding(img_path: str, fr_module):
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img_rgb = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))
        locs = fr_module.face_locations(img_rgb, number_of_times_to_upsample=1, model="hog")
        if not locs: return None
        encs = fr_module.face_encodings(img_rgb, known_face_locations=locs, num_jitters=1)
        return encs[0] if encs else None
    except Exception as e:
        logger.exception(f"Error encoding {img_path}: {e}"); return None


def build_face_database(folder: str, fr_module) -> dict:
    db = {}
    for p in Path(folder).glob("*"):
        if p.suffix.lower() not in {".jpg",".jpeg",".png",".webp",".bmp"}: continue
        name = p.stem.replace("_"," ").replace("-"," ").title()
        enc = load_single_face_encoding(str(p), fr_module)
        if enc is not None: db[name] = enc
    return db


def _reload_known_folder(folder="known_persons"):
    if state.fr_module and os.path.isdir(folder):
        state.face_db = build_face_database(folder, state.fr_module)
        logger.info(f"Loaded {len(state.face_db)} known face(s)")


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loader
# ═══════════════════════════════════════════════════════════════════════════════
def load_models():
    logger.info("Loading models…")
    try:
        from ultralytics import YOLO
        state.yolo_model = YOLO("yolov8x.pt")
        state.yolo_model.to("cuda")
        logger.info("YOLOv8x loaded ✅")
    except Exception as e:
        logger.error(f"YOLO load error: {e}")
        state.load_error += f"YOLO: {e}. "

    try:
        import face_recognition as _fr
        state.fr_module = _fr
        logger.info("face_recognition loaded ✅")
    except Exception as e:
        logger.warning(f"face_recognition load error: {e}")
        state.load_error += f"FaceRec: {e}. "

    _reload_known_folder()
    state.models_loaded = True
    logger.info("Model loading complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════════════════════════════════════════
class AppState:
    yolo_model = None
    fr_module  = None
    face_db: dict = {}
    models_loaded = False
    load_error: str = ""
    multi_camera: MultiCameraManager = MultiCameraManager()


state = AppState()

from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=6)

_frame_counter = 0
FACE_REC_EVERY_N_FRAMES = 15

# Non-blocking inference state
_cam_busy: Dict[int, bool] = {}
_cam_last_annotated: Dict[int, np.ndarray] = {}
_cam_last_labels: Dict[int, list] = {}
_cam_frame_buffer: Dict[int, list] = {}   # cam_id → [(ts, frame), ...]
HLS_DELAY_SECONDS: float = 2.5            # tunable; ~segment latency of HLS player
_BUFFER_MAX_AGE:   float = 8.0

# ═══════════════════════════════════════════════════════════════════════════════
# YOLO Classes + Colors
# ═══════════════════════════════════════════════════════════════════════════════
ALL_YOLO_CLASSES = {
    0:"person",1:"bicycle",2:"car",3:"motorcycle",4:"airplane",
    5:"bus",6:"train",7:"truck",8:"boat",9:"traffic light",
    10:"fire hydrant",11:"stop sign",12:"parking meter",13:"bench",14:"bird",
    15:"cat",16:"dog",17:"horse",18:"sheep",19:"cow",
    20:"elephant",21:"bear",22:"zebra",23:"giraffe",24:"backpack",
    25:"umbrella",26:"handbag",27:"tie",28:"suitcase",29:"frisbee",
    30:"skis",31:"snowboard",32:"sports ball",33:"kite",34:"baseball bat",
    35:"baseball glove",36:"skateboard",37:"surfboard",38:"tennis racket",39:"bottle",
    40:"wine glass",41:"cup",42:"fork",43:"knife",44:"spoon",
    45:"bowl",46:"banana",47:"apple",48:"sandwich",49:"orange",
    50:"broccoli",51:"carrot",52:"hot dog",53:"pizza",54:"donut",
    55:"cake",56:"chair",57:"couch",58:"potted plant",59:"bed",
    60:"dining table",61:"toilet",62:"tv",63:"laptop",64:"mouse",
    65:"remote",66:"keyboard",67:"cell phone",68:"microwave",69:"oven",
    70:"toaster",71:"sink",72:"refrigerator",73:"book",74:"clock",
    75:"vase",76:"scissors",77:"teddy bear",78:"hair drier",79:"toothbrush"
}

def generate_class_colors():
    np.random.seed(42)
    return {name: tuple(np.random.randint(50,255,3).tolist()) for name in ALL_YOLO_CLASSES.values()}

CLASS_COLORS = generate_class_colors()


# ═══════════════════════════════════════════════════════════════════════════════
# YOLO Inference — returns TRANSPARENT PNG overlay (boxes only, no background)
# ═══════════════════════════════════════════════════════════════════════════════
def process_frame(
    frame_bgr: np.ndarray,
    conf_threshold: float = 0.40,
    class_filter: Optional[List[str]] = None,
    frame_num: int = 0,
) -> tuple:
    labels_found = []
    if state.yolo_model is None or not class_filter:
        return None, labels_found

    filter_ids = {cid for cid, cname in ALL_YOLO_CLASSES.items()
                  if cname.lower() in [c.lower() for c in class_filter]}
    if not filter_ids:
        return None, labels_found

    INFER_SCALE = 0.35
    h_orig, w_orig = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (int(w_orig * INFER_SCALE), int(h_orig * INFER_SCALE)))

    try:
        results = state.yolo_model(small, conf=conf_threshold, verbose=False)[0]
    except Exception as e:
        logger.error(f"YOLO inference error: {e}")
        return None, labels_found

    # Create transparent RGBA overlay — only boxes drawn, background clear
    overlay = np.zeros((h_orig, w_orig, 4), dtype=np.uint8)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in filter_ids: continue

        conf  = float(box.conf[0])
        label = ALL_YOLO_CLASSES[cls_id]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = int(x1 / INFER_SCALE); y1 = int(y1 / INFER_SCALE)
        x2 = int(x2 / INFER_SCALE); y2 = int(y2 / INFER_SCALE)

        display_name = label

        # Face recognition
        if label == "person" and state.fr_module and state.face_db and frame_num % FACE_REC_EVERY_N_FRAMES == 0:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                roi = np.ascontiguousarray(
                    frame_rgb[max(0,y1):min(h_orig,y2), max(0,x1):min(w_orig,x2)], dtype=np.uint8
                )
                if roi.size > 0:
                    locs = state.fr_module.face_locations(roi, number_of_times_to_upsample=1, model="hog")
                    if locs:
                        encs = state.fr_module.face_encodings(roi, known_face_locations=locs, num_jitters=1)
                        known_encs  = list(state.face_db.values())
                        known_names = list(state.face_db.keys())
                        for enc in encs:
                            dists   = state.fr_module.face_distance(known_encs, enc)
                            matches = state.fr_module.compare_faces(known_encs, enc, tolerance=0.55)
                            if len(dists):
                                best = int(np.argmin(dists))
                                if matches[best]: display_name = known_names[best]
            except Exception as e:
                logger.debug(f"Face rec error: {e}")

        # Draw on transparent overlay (BGRA)
        bgra = (*color[::-1], 255)  # BGR → BGRA opaque
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bgra, 2)

        tag = f"{display_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        # Label background — semi-transparent fill
        label_overlay = overlay.copy()
        cv2.rectangle(label_overlay, (x1, y1 - th - 8), (x1 + tw + 6, y1), (*color[::-1], 200), -1)
        cv2.addWeighted(label_overlay, 0.85, overlay, 0.15, 0, overlay)
        cv2.putText(overlay, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255,255,255,255), 1)
        labels_found.append(display_name)

    return overlay, labels_found


# ═══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield
    state.multi_camera.disconnect_all()
    for cid in list(_cam_ffmpeg.keys()):
        stop_hls(cid)


# ═══════════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Multi-Camera YOLO Detection", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve HLS segments as static files
app.mount("/hls", StaticFiles(directory="hls_streams"), name="hls")


# ═══════════════════════════════════════════════════════════════════════════════
# REST Endpoints
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "cctv_v3.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    return {
        "yolo": state.yolo_model is not None,
        "face_recognition": state.fr_module is not None,
        "known_faces": list(state.face_db.keys()),
        "models_loaded": state.models_loaded,
        "error": state.load_error or None,
        "all_classes": ALL_YOLO_CLASSES,
        "class_colors": CLASS_COLORS,
        "cameras": state.multi_camera.get_all_status(),
        "max_cameras": MAX_CAMERAS,
    }


class RTSPConnectRequest(BaseModel):
    camera_id: int
    url: str
    label: str = ""


@app.post("/api/cameras/connect")
async def camera_connect(req: RTSPConnectRequest):
    if req.camera_id < 0 or req.camera_id >= MAX_CAMERAS:
        raise HTTPException(400, f"camera_id must be 0–{MAX_CAMERAS-1}")
    url = req.url.strip()
    if not url: raise HTTPException(400, "RTSP URL required")

    cam = state.multi_camera.get_camera(req.camera_id)
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, cam.connect, url, req.label)
    if not success: raise HTTPException(503, cam.error or "Failed to connect")

    # Start HLS conversion immediately on connect
    start_hls(req.camera_id, url)

    return cam.to_dict()


@app.post("/api/cameras/{camera_id}/disconnect")
async def camera_disconnect(camera_id: int):
    cam = state.multi_camera.get_camera(camera_id)
    if not cam: raise HTTPException(404, "Camera not found")
    cam.disconnect()
    stop_hls(camera_id)
    return {"camera_id": camera_id, "connected": False}


@app.get("/api/cameras")
async def get_cameras():
    return {"cameras": state.multi_camera.get_all_status()}


@app.get("/api/cameras/{camera_id}/hls_ready")
async def hls_ready(camera_id: int):
    """Poll this until HLS stream is ready (ffmpeg has written first segment)."""
    m3u8 = HLS_DIR / str(camera_id) / "stream.m3u8"
    seg  = list((HLS_DIR / str(camera_id)).glob("*.ts"))
    ready = m3u8.exists() and len(seg) > 0
    return {"ready": ready, "url": f"/hls/{camera_id}/stream.m3u8" if ready else None}


@app.post("/api/cameras/{camera_id}/disconnect")
async def camera_disconnect_id(camera_id: int):
    cam = state.multi_camera.get_camera(camera_id)
    if not cam: raise HTTPException(404, "Camera not found")
    cam.disconnect(); stop_hls(camera_id)
    return {"camera_id": camera_id, "connected": False}


# ── Image / Video ─────────────────────────────────────────────────────────────
@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")
):
    class_filter = [c.strip() for c in classes.split(",") if c.strip()]
    contents = await file.read()
    img_bgr  = cv2.cvtColor(np.array(Image.open(io.BytesIO(contents)).convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay, labels = process_frame(img_bgr, conf, class_filter, 0)

    # For image endpoint, composite overlay onto original
    if overlay is not None:
        bgra_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        alpha = overlay[:,:,3:4] / 255.0
        bgra_frame[:,:,:3] = (bgra_frame[:,:,:3] * (1 - alpha) + overlay[:,:,:3] * alpha).astype(np.uint8)
        result = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2BGR)
    else:
        result = img_bgr

    _, buf = cv2.imencode(".jpg", result)
    b64 = base64.b64encode(buf).decode()
    return {"image": f"data:image/jpeg;base64,{b64}", "labels": labels, "unique": list(set(labels))}


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")
):
    class_filter = [c.strip() for c in classes.split(",") if c.strip()]
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read()); in_path = tmp.name
    out_path = in_path.replace(suffix, "_annotated.mp4")
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    all_labels = set(); fc = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        overlay, lbls = process_frame(frame, conf, class_filter, fc)
        if overlay is not None:
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            alpha = overlay[:,:,3:4] / 255.0
            bgra[:,:,:3] = (bgra[:,:,:3] * (1-alpha) + overlay[:,:,:3] * alpha).astype(np.uint8)
            frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        all_labels.update(lbls); out.write(frame); fc += 1
    cap.release(); out.release(); os.unlink(in_path)
    return FileResponse(out_path, media_type="video/mp4", filename="annotated_video.mp4",
                        headers={"X-Detections": ",".join(sorted(all_labels))})


# ── Face management ───────────────────────────────────────────────────────────
@app.post("/api/faces/upload")
async def upload_faces(files: list[UploadFile] = File(...)):
    if not state.fr_module: raise HTTPException(503, "face_recognition not available")
    tmp_dir = tempfile.mkdtemp()
    try:
        for uf in files:
            fp = os.path.join(tmp_dir, uf.filename)
            with open(fp, "wb") as f: f.write(await uf.read())
        extra = build_face_database(tmp_dir, state.fr_module)
        state.face_db.update(extra)
        return {"enrolled": list(extra.keys()), "total_known": list(state.face_db.keys())}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    if name not in state.face_db: raise HTTPException(404, f"'{name}' not found")
    del state.face_db[name]
    return {"deleted": name, "remaining": list(state.face_db.keys())}


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket — Webcam (still uses full JPEG, webcam doesn't need HLS)
# ═══════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    global _frame_counter
    await websocket.accept()
    try:
        loop = asyncio.get_event_loop()
        while True:
            data = await websocket.receive_json()
            frame_b64    = data.get("frame", "")
            class_filter = data.get("classes", [])
            conf         = data.get("conf", 0.40)
            if "," in frame_b64: frame_b64 = frame_b64.split(",")[1]
            frame_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(frame_b64), np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                await websocket.send_json({"error": "invalid frame"}); continue
            _frame_counter += 1
            overlay, labels = await loop.run_in_executor(
                _executor, process_frame, frame_bgr, conf, class_filter, _frame_counter
            )
            # For webcam composite overlay onto frame
            if overlay is not None:
                bgra  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
                alpha = overlay[:,:,3:4] / 255.0
                bgra[:,:,:3] = (bgra[:,:,:3] * (1-alpha) + overlay[:,:,:3] * alpha).astype(np.uint8)
                frame_bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 55])
            b64 = base64.b64encode(buf).decode()
            await websocket.send_json({"image": f"data:image/jpeg;base64,{b64}", "labels": labels, "unique": list(set(labels))})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"Webcam WS error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket — RTSP overlay (sends transparent PNG boxes only, NOT full video)
# ═══════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws/rtsp/{camera_id}")
async def websocket_rtsp(websocket: WebSocket, camera_id: int):
    """
    Sends ONLY the YOLO bounding-box overlay as transparent PNG.
    The actual video is played by hls.js in the browser.
    This means: smooth full-FPS video from HLS + async YOLO boxes on top.
    """
    global _frame_counter
    await websocket.accept()
    logger.info(f"WS overlay client connected — cam {camera_id}")

    cam = state.multi_camera.get_camera(camera_id)
    if not cam:
        await websocket.send_json({"error": f"Invalid camera_id: {camera_id}"}); await websocket.close(); return

    class_filter = []; conf = 0.40; interval_ms = 100  # overlay can be slower than video

    try:
        loop = asyncio.get_event_loop()
        try:
            init = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
            class_filter = init.get("classes",     [])
            conf         = init.get("conf",         0.40)
            interval_ms  = init.get("interval_ms",  100)
        except asyncio.TimeoutError:
            pass

        if not cam.is_connected:
            await websocket.send_json({"error": f"Camera {camera_id} not connected."})
            await websocket.close(); return

        last_send = 0.0

        while True:
            # Non-blocking config update
            try:
                update = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if "classes"        in update: class_filter      = update["classes"]
                if "conf"           in update: conf              = update["conf"]
                if "interval_ms"    in update: interval_ms       = update["interval_ms"]
                if "hls_lag_seconds" in update:
                    global HLS_DELAY_SECONDS
                    HLS_DELAY_SECONDS = max(0.5, float(update["hls_lag_seconds"]))
                    logger.info(f"[Cam {camera_id}] HLS delay updated → {HLS_DELAY_SECONDS:.2f}s")
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            now = time.time()
            if (now - last_send) * 1000 < interval_ms:
                await asyncio.sleep(0.002); continue

            result = cam.get_frame()
            if result is None:
                if not cam.is_connected:
                    await websocket.send_json({"error": f"Camera {camera_id} disconnected"}); break
                await asyncio.sleep(0.005); continue

            raw_frame, raw_ts = result

            # ── Buffer frame, then serve the one that's HLS_DELAY_SECONDS old ──
            buf = _cam_frame_buffer.setdefault(camera_id, [])
            buf.append((raw_ts, raw_frame))
            now = time.time()
            # Prune frames older than _BUFFER_MAX_AGE
            _cam_frame_buffer[camera_id] = [
                (t, f) for t, f in buf if now - t < _BUFFER_MAX_AGE
            ]
            # Pick most-recent frame that is at least HLS_DELAY_SECONDS old
            candidates = [
                (t, f) for t, f in _cam_frame_buffer[camera_id]
                if now - t >= HLS_DELAY_SECONDS
            ]
            if not candidates:
                await asyncio.sleep(0.005); continue
            frame = candidates[-1][1]

            _frame_counter += 1

            # Non-blocking inference
            if not _cam_busy.get(camera_id, False):
                _cam_busy[camera_id] = True
                snap = frame.copy(); fc = _frame_counter

                async def _infer(f, cid, cf, c, fn):
                    try:
                        ovl, lbl = await loop.run_in_executor(_executor, process_frame, f, c, cf, fn)
                        _cam_last_annotated[cid] = ovl
                        _cam_last_labels[cid]    = lbl
                    finally:
                        _cam_busy[cid] = False

                asyncio.create_task(_infer(snap, camera_id, class_filter, conf, fc))

            # Send last overlay (transparent PNG) — don't wait for current inference
            ovl_out = _cam_last_annotated.get(camera_id)
            lbl_out = _cam_last_labels.get(camera_id, [])

            if ovl_out is not None:
                # Encode as PNG to preserve transparency
                _, buf = cv2.imencode(".png", ovl_out)
                b64 = base64.b64encode(buf).decode()
                try:
                    await websocket.send_json({
                        "overlay": f"data:image/png;base64,{b64}",
                        "labels":  lbl_out,
                        "unique":  list(set(lbl_out)),
                        "camera_id": camera_id,
                    })
                except Exception:
                    break
            else:
                # No detections yet — send empty signal so frontend knows WS is alive
                try:
                    await websocket.send_json({"overlay": None, "labels": [], "unique": [], "camera_id": camera_id})
                except Exception:
                    break

            last_send = time.time()

    except WebSocketDisconnect:
        logger.info(f"Overlay WS cam {camera_id} disconnected")
    except Exception as e:
        logger.exception(f"Overlay WS cam {camera_id} error: {e}")
    finally:
        _cam_busy.pop(camera_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cctv_v3:app", host="0.0.0.0", port=8002, reload=False, ws="wsproto")