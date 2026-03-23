"""
Search-Based Object Detection with All 80 YOLO Classes
FastAPI Backend with dynamic class filtering + RTSP camera support
"""

import asyncio
import base64
import io
import logging
import os
import sys
import tempfile
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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


# ─── RTSP Stream Manager ───────────────────────────────────────────────────────
class RTSPStreamManager:
    """
    Manages an RTSP camera connection.
    Runs a background thread that continuously reads frames.
    Always keeps the latest frame available for on-demand access.
    """
    def __init__(self):
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

    def connect(self, rtsp_url: str) -> bool:
        """Connect to an RTSP stream."""
        self.disconnect()  # Stop any existing stream

        logger.info(f"Connecting to RTSP: {rtsp_url}")
        self.url = rtsp_url
        self.error = ""

        # Use FFMPEG backend for better RTSP compatibility
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        # RTSP-friendly buffer settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.error = f"Cannot open stream: {rtsp_url}"
            logger.error(self.error)
            cap.release()
            return False

        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"RTSP connected: {self.width}x{self.height} @ {self.fps:.1f}fps")

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        """Background thread: continuously drain the RTSP buffer, keep latest frame."""
        consecutive_failures = 0
        MAX_FAILURES = 30

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                logger.warning("RTSP cap closed, stopping read loop.")
                self.error = "Stream disconnected"
                self.running = False
                break

            ret, frame = self.cap.read()

            if not ret:
                consecutive_failures += 1
                logger.warning(f"RTSP read failure #{consecutive_failures}")
                if consecutive_failures >= MAX_FAILURES:
                    logger.error("Too many RTSP failures, attempting reconnect…")
                    self._attempt_reconnect()
                    consecutive_failures = 0
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            with self.lock:
                self.latest_frame = frame.copy()

    def _attempt_reconnect(self):
        """Try to reconnect to the RTSP stream."""
        if self.cap:
            self.cap.release()
        time.sleep(2)
        if self.url:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.cap = cap
                logger.info("RTSP reconnected ✅")
                self.error = ""
            else:
                self.error = "Reconnect failed"
                logger.error("RTSP reconnect failed")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe)."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def disconnect(self):
        """Stop the stream and release resources."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.latest_frame = None
        self.url = ""
        logger.info("RTSP stream disconnected")

    @property
    def is_connected(self) -> bool:
        return self.running and self.cap is not None and self.cap.isOpened()


# ─── Face Recognition helpers ─────────────────────────────────────────────────
def load_single_face_encoding(img_path: str, fr_module):
    """Load and encode a single face from an image file."""
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(img_path).convert("RGB")
        img_rgb = np.array(pil_img, dtype=np.uint8)
        img_rgb = np.ascontiguousarray(img_rgb)

        locs = fr_module.face_locations(img_rgb, number_of_times_to_upsample=1, model="hog")
        logger.debug(f"face_locations found {len(locs)} face(s) in {Path(img_path).name}")

        if not locs:
            logger.warning(f"No face detected in {Path(img_path).name}")
            return None

        encs = fr_module.face_encodings(img_rgb, known_face_locations=locs, num_jitters=1)
        if not encs:
            logger.warning(f"face_encodings returned empty for {Path(img_path).name}")
            return None

        logger.info(f"✅ Encoding built for {Path(img_path).name}")
        return encs[0]

    except Exception as e:
        logger.exception(f"Error encoding {img_path}: {e}")
        return None


def build_face_database(folder: str, fr_module) -> dict:
    """Build face database from images in a folder."""
    db = {}
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in Path(folder).glob("*"):
        if p.suffix.lower() not in supported:
            continue
        name = p.stem.replace("_", " ").replace("-", " ").title()
        enc = load_single_face_encoding(str(p), fr_module)
        if enc is not None:
            db[name] = enc
            logger.info(f"Added '{name}' to face DB")
    return db


def _reload_known_folder(folder="known_persons"):
    """Reload face database from known_persons folder."""
    if state.fr_module and os.path.isdir(folder):
        state.face_db = build_face_database(folder, state.fr_module)
        logger.info(f"Loaded {len(state.face_db)} known face(s) from '{folder}'")


# ─── Model loader ───────────────────────────────────────────────────────────
def load_models():
    """Load YOLO and face_recognition models once at startup."""
    logger.info("Loading AI models…")
    try:
        from ultralytics import YOLO
        state.yolo_model = YOLO("yolov8x.pt")
        logger.info("YOLOv8 loaded ✅")
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

    # Load known persons folder
    _reload_known_folder()
    state.models_loaded = True
    logger.info("Model loading complete.")


# ─── Global state ─────────────────────────────────────────────────────────────
class AppState:
    yolo_model = None
    fr_module = None
    face_db: dict = {}
    models_loaded = False
    load_error: str = ""
    active_filters: set = set()
    rtsp_manager: RTSPStreamManager = RTSPStreamManager()

state = AppState()


from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=3)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield
    state.rtsp_manager.disconnect()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Search-Based Object Detection",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── All 80 COCO Classes ──────────────────────────────────────────────────────
ALL_YOLO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

def generate_class_colors():
    np.random.seed(42)
    colors = {}
    for class_id, class_name in ALL_YOLO_CLASSES.items():
        colors[class_name] = tuple(np.random.randint(50, 255, 3).tolist())
    return colors

CLASS_COLORS = generate_class_colors()


# ─── Detection with filtering ──────────────────────────────────────────────────
_frame_counter = 0
FACE_REC_EVERY_N_FRAMES = 15

def process_frame(
    frame_bgr: np.ndarray,
    conf_threshold: float = 0.40,
    class_filter: Optional[List[str]] = None,
    frame_num: int = 0
):
    labels_found = []

    if state.yolo_model is None:
        return frame_bgr, labels_found

    if not class_filter:
        return frame_bgr, labels_found

    filter_ids = set()
    for class_name in class_filter:
        for cid, cname in ALL_YOLO_CLASSES.items():
            if cname.lower() == class_name.lower():
                filter_ids.add(cid)
                break

    if not filter_ids:
        return frame_bgr, labels_found

    INFER_SCALE = 0.35
    h_orig, w_orig = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (int(w_orig * INFER_SCALE), int(h_orig * INFER_SCALE)))

    try:
        results = state.yolo_model(small, conf=conf_threshold, verbose=False)[0]
    except Exception as e:
        logger.error(f"YOLO inference error: {e}")
        return frame_bgr, labels_found

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in filter_ids:
            continue
        if cls_id not in ALL_YOLO_CLASSES:
            continue

        conf = float(box.conf[0])
        label = ALL_YOLO_CLASSES[cls_id]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = int(x1 / INFER_SCALE); y1 = int(y1 / INFER_SCALE)
        x2 = int(x2 / INFER_SCALE); y2 = int(y2 / INFER_SCALE)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        display_name = label

        run_face_rec = (frame_num % FACE_REC_EVERY_N_FRAMES == 0)
        if label == "person" and state.fr_module and state.face_db and run_face_rec:
            try:
                h, w = frame_bgr.shape[:2]
                frame_rgb_orig = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                roi = np.ascontiguousarray(
                    frame_rgb_orig[max(0, y1):min(h, y2), max(0, x1):min(w, x2)], dtype=np.uint8
                )
                if roi.size > 0:
                    locs = state.fr_module.face_locations(roi, number_of_times_to_upsample=1, model="hog")
                    if locs:
                        encs = state.fr_module.face_encodings(roi, known_face_locations=locs, num_jitters=1)
                        for enc in encs:
                            known_encs  = list(state.face_db.values())
                            known_names = list(state.face_db.keys())
                            dists   = state.fr_module.face_distance(known_encs, enc)
                            matches = state.fr_module.compare_faces(known_encs, enc, tolerance=0.55)
                            if len(dists):
                                best = int(np.argmin(dists))
                                if matches[best]:
                                    display_name = known_names[best]
            except Exception as e:
                logger.debug(f"Face recognition error: {e}")

        tag = f"{display_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_bgr, tag, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
        labels_found.append(display_name)

    return frame_bgr, labels_found


# ─── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "cctv.html"
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
        "total_classes": len(ALL_YOLO_CLASSES),
        "rtsp": {
            "connected": state.rtsp_manager.is_connected,
            "url": state.rtsp_manager.url,
            "error": state.rtsp_manager.error,
            "width": state.rtsp_manager.width,
            "height": state.rtsp_manager.height,
            "fps": state.rtsp_manager.fps,
        }
    }


@app.get("/api/classes")
async def get_all_classes():
    return {
        "classes": ALL_YOLO_CLASSES,
        "colors": CLASS_COLORS,
        "total": len(ALL_YOLO_CLASSES)
    }


# ─── RTSP Endpoints ────────────────────────────────────────────────────────────

class RTSPConnectRequest(BaseModel):
    url: str


@app.post("/api/rtsp/connect")
async def rtsp_connect(req: RTSPConnectRequest):
    """Connect to an RTSP stream."""
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "RTSP URL is required")

    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, state.rtsp_manager.connect, url)

    if not success:
        raise HTTPException(503, state.rtsp_manager.error or "Failed to connect to RTSP stream")

    return {
        "connected": True,
        "url": url,
        "width": state.rtsp_manager.width,
        "height": state.rtsp_manager.height,
        "fps": state.rtsp_manager.fps,
    }


@app.post("/api/rtsp/disconnect")
async def rtsp_disconnect():
    """Disconnect from the RTSP stream."""
    state.rtsp_manager.disconnect()
    return {"connected": False}


@app.get("/api/rtsp/status")
async def rtsp_status():
    """Get RTSP connection status."""
    return {
        "connected": state.rtsp_manager.is_connected,
        "url": state.rtsp_manager.url,
        "error": state.rtsp_manager.error,
        "width": state.rtsp_manager.width,
        "height": state.rtsp_manager.height,
        "fps": state.rtsp_manager.fps,
    }


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")
):
    """Detect only specified classes in an uploaded image."""
    class_filter = [c.strip() for c in classes.split(",") if c.strip()] if classes else []

    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    annotated, labels = process_frame(img_bgr, conf_threshold=conf, class_filter=class_filter, frame_num=0)
    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode()

    return {
        "image": f"data:image/jpeg;base64,{b64}",
        "labels": labels,
        "unique": list(set(labels)),
        "filtered_classes": class_filter
    }


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")
):
    """Process an uploaded video, detecting only specified classes."""
    class_filter = [c.strip() for c in classes.split(",") if c.strip()] if classes else []

    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(await file.read())
        in_path = tmp_in.name

    out_path = in_path.replace(suffix, "_annotated.mp4")

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    all_labels = set()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ann, lbls = process_frame(frame, conf_threshold=conf, class_filter=class_filter, frame_num=frame_count)
        all_labels.update(lbls)
        out.write(ann)
        frame_count += 1

    cap.release()
    out.release()
    os.unlink(in_path)

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="annotated_video.mp4",
        headers={"X-Detections": ",".join(sorted(all_labels))},
        background=None,
    )


@app.post("/api/faces/upload")
async def upload_faces(files: list[UploadFile] = File(...)):
    if not state.fr_module:
        raise HTTPException(503, "face_recognition module not available")

    tmp_dir = tempfile.mkdtemp()
    saved = []
    try:
        for uf in files:
            data = await uf.read()
            fp = os.path.join(tmp_dir, uf.filename)
            with open(fp, "wb") as f:
                f.write(data)
            saved.append(fp)

        extra = build_face_database(tmp_dir, state.fr_module)
        state.face_db.update(extra)

        return {
            "enrolled": list(extra.keys()),
            "total_known": list(state.face_db.keys()),
            "failed": len(saved) - len(extra),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    if name not in state.face_db:
        raise HTTPException(404, f"'{name}' not found in face database")
    del state.face_db[name]
    return {"deleted": name, "remaining": list(state.face_db.keys())}


# ─── WebSocket: browser webcam frames ─────────────────────────────────────────
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    Client sends: {"frame": <base64_jpeg>, "classes": ["person", "car"]}
    Server responds: {"image": <base64_jpeg>, "labels": [...]}
    """
    global _frame_counter
    await websocket.accept()
    logger.info("WebSocket client connected (browser cam)")

    try:
        loop = asyncio.get_event_loop()
        while True:
            data = await websocket.receive_json()

            frame_b64 = data.get("frame", "")
            class_filter = data.get("classes", [])
            conf = data.get("conf", 0.40)

            if "," in frame_b64:
                frame_b64 = frame_b64.split(",")[1]

            img_data = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                await websocket.send_json({"error": "invalid frame"})
                continue

            _frame_counter += 1

            annotated, labels = await loop.run_in_executor(
                _executor,
                process_frame,
                frame_bgr,
                conf,
                class_filter,
                _frame_counter
            )

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode()

            await websocket.send_json({
                "image": f"data:image/jpeg;base64,{b64}",
                "labels": labels,
                "unique": list(set(labels))
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")


# ─── WebSocket: RTSP stream detection ─────────────────────────────────────────
@app.websocket("/ws/rtsp")
async def websocket_rtsp(websocket: WebSocket):
    """
    Server pulls frames from the connected RTSP stream, runs detection,
    and pushes annotated frames to the client.

    Client sends config: {"classes": ["person", "car"], "conf": 0.4, "interval_ms": 200}
    Server streams:      {"image": <base64_jpeg>, "labels": [...], "fps": ...}
    """
    global _frame_counter
    await websocket.accept()
    logger.info("WebSocket RTSP client connected")

    class_filter = []
    conf = 0.40
    interval_ms = 200  # ms between frames sent to client

    try:
        loop = asyncio.get_event_loop()

        # Receive initial config (optional, with timeout)
        try:
            init_data = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
            class_filter = init_data.get("classes", [])
            conf = init_data.get("conf", 0.40)
            interval_ms = init_data.get("interval_ms", 200)
        except asyncio.TimeoutError:
            pass  # Use defaults

        logger.info(f"RTSP WS config: classes={class_filter}, conf={conf}, interval={interval_ms}ms")

        # Check RTSP is connected
        if not state.rtsp_manager.is_connected:
            await websocket.send_json({"error": "RTSP stream not connected. Call /api/rtsp/connect first."})
            await websocket.close()
            return

        last_send = 0.0
        frame_times = []

        while True:
            # Handle incoming config updates (non-blocking)
            try:
                update = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if "classes" in update:
                    class_filter = update["classes"]
                if "conf" in update:
                    conf = update["conf"]
                if "interval_ms" in update:
                    interval_ms = update["interval_ms"]
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            now = time.time()
            if (now - last_send) * 1000 < interval_ms:
                await asyncio.sleep(0.005)
                continue

            # Get latest RTSP frame
            frame = state.rtsp_manager.get_frame()
            if frame is None:
                if not state.rtsp_manager.is_connected:
                    await websocket.send_json({"error": "RTSP stream disconnected"})
                    break
                await asyncio.sleep(0.02)
                continue

            _frame_counter += 1
            t0 = time.time()

            annotated, labels = await loop.run_in_executor(
                _executor,
                process_frame,
                frame,
                conf,
                class_filter,
                _frame_counter
            )

            # Track processing fps
            elapsed = time.time() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            b64 = base64.b64encode(buf).decode()

            try:
                await websocket.send_json({
                    "image": f"data:image/jpeg;base64,{b64}",
                    "labels": labels,
                    "unique": list(set(labels)),
                    "processing_fps": round(avg_fps, 1),
                    "stream_fps": round(state.rtsp_manager.fps, 1),
                })
            except Exception:
                break

            last_send = time.time()

    except WebSocketDisconnect:
        logger.info("RTSP WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"RTSP WebSocket error: {e}")
    finally:
        logger.info("RTSP WebSocket handler cleaned up")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cctv:app", host="0.0.0.0", port=8001, reload=False)