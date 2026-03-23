"""
Ultra-Low-Latency Object Detection
FastAPI Backend — TensorRT + MJPEG streaming + zero-copy pipeline
outcome : 50ms end-to-end browser display latency on Jetson AGX
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
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
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
logger = logging.getLogger("UltraDetect")

MAX_CAMERAS      = 9
JPEG_QUALITY     = 75
INFER_SCALE      = 0.40
FACE_REC_EVERY_N = 20
_BASE       = Path(__file__).parent
ENGINE_PATH = str(_BASE / "yolo26x.engine")
PT_PATH     = str(_BASE / "yolo26x.pt")

# ─── TurboJPEG (optional — falls back to cv2 if not installed) ────────────────
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    _turbo = TurboJPEG()
    logger.info("TurboJPEG available ✅")
    def encode_jpeg(frame_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
        return _turbo.encode(frame_bgr, quality=quality, pixel_format=TJPF_BGR)
except (ImportError, RuntimeError):
    _turbo = None
    logger.warning("TurboJPEG not available — using cv2. To enable: sudo apt install libturbojpeg")
    def encode_jpeg(frame_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return bytes(buf)

# ─── All 80 COCO Classes ──────────────────────────────────────────────────────
ALL_YOLO_CLASSES = {
    0:"person",1:"bicycle",2:"car",3:"motorcycle",4:"airplane",5:"bus",6:"train",7:"truck",
    8:"boat",9:"traffic light",10:"fire hydrant",11:"stop sign",12:"parking meter",13:"bench",
    14:"bird",15:"cat",16:"dog",17:"horse",18:"sheep",19:"cow",20:"elephant",21:"bear",
    22:"zebra",23:"giraffe",24:"backpack",25:"umbrella",26:"handbag",27:"tie",28:"suitcase",
    29:"frisbee",30:"skis",31:"snowboard",32:"sports ball",33:"kite",34:"baseball bat",
    35:"baseball glove",36:"skateboard",37:"surfboard",38:"tennis racket",39:"bottle",
    40:"wine glass",41:"cup",42:"fork",43:"knife",44:"spoon",45:"bowl",46:"banana",
    47:"apple",48:"sandwich",49:"orange",50:"broccoli",51:"carrot",52:"hot dog",53:"pizza",
    54:"donut",55:"cake",56:"chair",57:"couch",58:"potted plant",59:"bed",60:"dining table",
    61:"toilet",62:"tv",63:"laptop",64:"mouse",65:"remote",66:"keyboard",67:"cell phone",
    68:"microwave",69:"oven",70:"toaster",71:"sink",72:"refrigerator",73:"book",74:"clock",
    75:"vase",76:"scissors",77:"teddy bear",78:"hair drier",79:"toothbrush"
}

def generate_class_colors():
    np.random.seed(42)
    return {name: tuple(np.random.randint(50, 255, 3).tolist()) for name in ALL_YOLO_CLASSES.values()}

CLASS_COLORS = generate_class_colors()


# ─── RTSP Stream Manager (zero-copy optimized) ────────────────────────────────
class RTSPStreamManager:
    """
    Zero-copy optimized RTSP reader.
    - CAP_PROP_BUFFERSIZE = 1  -> always newest frame, no queue buildup
    - Frame stored WITHOUT .copy() in read loop; copy only on get_frame()
    - threading.Event signals new frame availability for low-latency polling
    """

    def __init__(self, camera_id: int):
        self.camera_id  = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_id  = 0
        self.running    = False
        self.thread: Optional[threading.Thread] = None
        self.url        = ""
        self.lock       = threading.Lock()
        self.new_frame  = threading.Event()
        self.error      = ""
        self.fps: float = 0.0
        self.width      = 0
        self.height     = 0
        self.label      = f"Camera {camera_id + 1}"

    def connect(self, rtsp_url: str, label: str = "") -> bool:
        self.disconnect()
        logger.info(f"[Cam {self.camera_id}] Connecting: {rtsp_url}")
        self.url   = rtsp_url
        self.label = label or f"Camera {self.camera_id + 1}"
        self.error = ""

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.error = f"Cannot open stream: {rtsp_url}"
            logger.error(self.error)
            cap.release()
            return False

        self.cap    = cap
        self.fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"[Cam {self.camera_id}] Connected: {self.width}x{self.height} @ {self.fps:.1f}fps")

        self.running = True
        self.thread  = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        consecutive_failures = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.error   = "Stream disconnected"
                self.running = False
                break
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    logger.error(f"[Cam {self.camera_id}] Too many failures, reconnecting")
                    self._attempt_reconnect()
                    consecutive_failures = 0
                time.sleep(0.02)
                continue
            consecutive_failures = 0
            with self.lock:
                self._frame    = frame      # no extra copy — frame is already new memory
                self._frame_id += 1
            self.new_frame.set()

    def _attempt_reconnect(self):
        if self.cap:
            self.cap.release()
        time.sleep(2)
        if self.url:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.cap   = cap
                self.error = ""
                logger.info(f"[Cam {self.camera_id}] Reconnected")
            else:
                self.error = "Reconnect failed"

    def get_frame(self):
        """Returns (frame_copy, frame_id). Single copy happens here only."""
        with self.lock:
            if self._frame is None:
                return None, 0
            return self._frame.copy(), self._frame_id

    def wait_for_new_frame(self, last_id: int, timeout: float = 0.05) -> bool:
        """Block until a frame newer than last_id is available, or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self._frame_id > last_id:
                    return True
            self.new_frame.wait(timeout=min(0.005, deadline - time.monotonic()))
            self.new_frame.clear()
        with self.lock:
            return self._frame_id > last_id

    def disconnect(self):
        self.running = False
        self.new_frame.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None
        self._frame    = None
        self._frame_id = 0
        self.url       = ""
        logger.info(f"[Cam {self.camera_id}] Disconnected")

    @property
    def is_connected(self) -> bool:
        return self.running and self.cap is not None and self.cap.isOpened()

    def to_dict(self):
        return {
            "camera_id": self.camera_id,
            "connected": self.is_connected,
            "url":       self.url,
            "label":     self.label,
            "error":     self.error,
            "width":     self.width,
            "height":    self.height,
            "fps":       self.fps,
        }


class MultiCameraManager:
    def __init__(self, max_cameras: int = MAX_CAMERAS):
        self.cameras: Dict[int, RTSPStreamManager] = {
            i: RTSPStreamManager(i) for i in range(max_cameras)
        }

    def get_camera(self, camera_id: int) -> Optional[RTSPStreamManager]:
        return self.cameras.get(camera_id)

    def get_all_status(self):
        return [cam.to_dict() for cam in self.cameras.values()]

    def disconnect_all(self):
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()


# ─── Face Recognition helpers ─────────────────────────────────────────────────
def load_single_face_encoding(img_path: str, fr_module):
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img_rgb = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))
        locs    = fr_module.face_locations(img_rgb, number_of_times_to_upsample=1, model="hog")
        if not locs:
            return None
        encs = fr_module.face_encodings(img_rgb, known_face_locations=locs, num_jitters=1)
        return encs[0] if encs else None
    except Exception as e:
        logger.exception(f"Error encoding {img_path}: {e}")
        return None

def build_face_database(folder: str, fr_module) -> dict:
    db        = {}
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in Path(folder).glob("*"):
        if p.suffix.lower() not in supported:
            continue
        name = p.stem.replace("_", " ").replace("-", " ").title()
        enc  = load_single_face_encoding(str(p), fr_module)
        if enc is not None:
            db[name] = enc
    return db

def _reload_known_folder(folder="known_persons"):
    if state.fr_module and os.path.isdir(folder):
        state.face_db = build_face_database(folder, state.fr_module)
        logger.info(f"Loaded {len(state.face_db)} known face(s)")


# ─── TensorRT export + load ────────────────────────────────────────────────────
# def _export_tensorrt(pt_path: str, engine_path: str) -> bool:
#     """
#     Export yolov8x.pt to TensorRT .engine optimized for this exact Jetson AGX.
#     Runs once (~10 min). Subsequent startups load .engine directly in ~5s.
#     """
#     logger.info("Exporting to TensorRT — this takes ~10 min on first run...")
#     try:
#         from ultralytics import YOLO
#         model = YOLO(pt_path)
#         model.export(
#             format="engine",
#             device=0,
#             half=True,        # FP16
#             workspace=4,      # GB of GPU workspace for TRT optimizer
#             simplify=True,
#         )
#         exported = Path(pt_path).with_suffix(".engine")
#         if exported.exists():
#             if str(exported) != engine_path:
#                 exported.rename(engine_path)
#             logger.info(f"TensorRT engine saved: {engine_path}")
#             return True
#         else:
#             logger.error("TRT export finished but .engine not found")
#             return False
#     except Exception as e:
#         logger.error(f"TensorRT export failed: {e}")
#         return False

def load_models():
    logger.info("Loading AI models...")
    try:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            logger.warning("CUDA not available — falling back to CPU")
            state.yolo_model = YOLO(PT_PATH)
            state.device     = "cpu"
        else:
            state.device = f"cuda:{torch.cuda.current_device()}"
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

            if Path(ENGINE_PATH).exists():
                logger.info(f"Loading existing TensorRT engine: {ENGINE_PATH}")
                state.yolo_model = YOLO(ENGINE_PATH, task="detect")
                logger.info("TensorRT engine loaded")
            else:
                if not Path(PT_PATH).exists():
                    logger.info(f"Downloading {PT_PATH}...")
                    YOLO(PT_PATH)  # triggers download only, don't assign yet

                logger.warning("No .engine found — loading PT model directly (slower)")
                state.yolo_model = YOLO(PT_PATH)
                if state.device != "cpu":
                    state.yolo_model.to(state.device)
                    # state.yolo_model.model.to(state.device)

                # ok = _export_tensorrt(PT_PATH, ENGINE_PATH)
                # if ok and Path(ENGINE_PATH).exists():
                #     state.yolo_model = YOLO(ENGINE_PATH)
                #     logger.info("TensorRT engine loaded after export")
                # else:
                #     logger.warning("TRT export failed — falling back to FP16 PyTorch")
                #     state.yolo_model = YOLO(PT_PATH)
                #     state.yolo_model.model.to("cuda").half()

        # Warm-up: 3 passes to fully JIT-compile CUDA kernels
        if state.yolo_model is not None:
            logger.info("Running warm-up passes...")
            dummy = np.zeros((int(640 * INFER_SCALE), int(640 * INFER_SCALE), 3), dtype=np.uint8)
            for _ in range(3):
                state.yolo_model(dummy, verbose=False)
            logger.info("Warm-up complete")
        else:
            logger.error("Skipping warm-up — YOLO model failed to load")

    except Exception as e:
        logger.error(f"YOLO load error: {e}")
        state.load_error += f"YOLO: {e}. "

    try:
        import face_recognition as _fr
        state.fr_module = _fr
        logger.info("face_recognition loaded")
    except Exception as e:
        logger.warning(f"face_recognition not available: {e}")

    _reload_known_folder()
    state.models_loaded = True
    logger.info("All models ready.")


# ─── Global state ─────────────────────────────────────────────────────────────
class AppState:
    yolo_model    = None
    fr_module     = None
    face_db: dict = {}
    models_loaded = False
    load_error: str = ""
    device: str   = "cpu"
    multi_camera: MultiCameraManager = MultiCameraManager()

state     = AppState()
_executor = ThreadPoolExecutor(max_workers=8)
_frame_counter = 0


# ─── Per-camera MJPEG + metadata state ───────────────────────────────────────
class CameraStreamState:
    """
    Holds the latest annotated JPEG and metadata for one camera.
    Decouples inference rate from MJPEG consumer rate.
    """
    def __init__(self):
        self.jpeg_bytes: Optional[bytes] = None
        self.labels: list     = []
        self.fps: float       = 0.0
        self.stream_fps: float = 0.0
        self._lock            = threading.Lock()

    def update(self, jpeg: bytes, labels: list, fps: float, stream_fps: float):
        with self._lock:
            self.jpeg_bytes = jpeg
            self.labels     = labels
            self.fps        = fps
            self.stream_fps = stream_fps

    def get(self):
        with self._lock:
            return self.jpeg_bytes, self.labels, self.fps, self.stream_fps

cam_stream_states: Dict[int, CameraStreamState] = {
    i: CameraStreamState() for i in range(MAX_CAMERAS)
}
cam_infer_tasks: Dict[int, asyncio.Task] = {}


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield
    state.multi_camera.disconnect_all()
    for task in cam_infer_tasks.values():
        task.cancel()


app = FastAPI(title="Ultra-Low-Latency Detection", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Core annotate ────────────────────────────────────────────────────────────
def _annotate_results(frame_bgr, results, filter_ids, frame_num):
    labels_found = []
    h_orig, w_orig = frame_bgr.shape[:2]
    run_face_rec   = (frame_num % FACE_REC_EVERY_N == 0)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in filter_ids or cls_id not in ALL_YOLO_CLASSES:
            continue
        conf  = float(box.conf[0])
        label = ALL_YOLO_CLASSES[cls_id]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = int(x1 / INFER_SCALE); y1 = int(y1 / INFER_SCALE)
        x2 = int(x2 / INFER_SCALE); y2 = int(y2 / INFER_SCALE)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        display_name = label

        if label == "person" and state.fr_module and state.face_db and run_face_rec:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                roi = np.ascontiguousarray(
                    frame_rgb[max(0,y1):min(h_orig,y2), max(0,x1):min(w_orig,x2)], dtype=np.uint8
                )
                if roi.size > 0:
                    locs = state.fr_module.face_locations(roi, model="hog")
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
            except Exception:
                pass

        tag = f"{display_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(frame_bgr, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(frame_bgr, tag, (x1+3, y1-4), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0,0,0), 1)
        labels_found.append(display_name)

    return frame_bgr, labels_found


def process_frame(frame_bgr, conf_threshold=0.40, class_filter=None, frame_num=0):
    labels_found = []
    if state.yolo_model is None or not class_filter:
        return frame_bgr, labels_found

    filter_ids = {
        cid for cls_name in class_filter
        for cid, cname in ALL_YOLO_CLASSES.items()
        if cname.lower() == cls_name.lower()
    }
    if not filter_ids:
        return frame_bgr, labels_found

    h, w   = frame_bgr.shape[:2]
    small  = cv2.resize(frame_bgr, (int(w * INFER_SCALE), int(h * INFER_SCALE)))
    try:
        results = state.yolo_model(small, conf=conf_threshold, verbose=False)[0]
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return frame_bgr, labels_found
    return _annotate_results(frame_bgr, results, filter_ids, frame_num)


# ─── Per-camera inference loop ────────────────────────────────────────────────
async def _camera_infer_loop(camera_id: int, class_filter: list, conf: float):
    """
    Dedicated asyncio Task per camera.
    - Uses wait_for_new_frame() so it wakes immediately when RTSP delivers a frame
    - Inference runs in ThreadPoolExecutor (non-blocking for event loop)
    - Publishes result to CameraStreamState for MJPEG and metadata consumers
    """
    global _frame_counter
    cam  = state.multi_camera.get_camera(camera_id)
    css  = cam_stream_states[camera_id]
    loop = asyncio.get_event_loop()

    last_frame_id = 0
    frame_times   = []
    logger.info(f"[Cam {camera_id}] Inference loop started")

    try:
        while True:
            got_new = await loop.run_in_executor(
                None, cam.wait_for_new_frame, last_frame_id, 0.05
            )
            if not cam.is_connected:
                break
            if not got_new:
                continue

            frame, frame_id = cam.get_frame()
            if frame is None or frame_id == last_frame_id:
                continue
            last_frame_id = frame_id

            _frame_counter += 1
            t0 = time.perf_counter()

            annotated, labels = await loop.run_in_executor(
                _executor, process_frame, frame, conf, class_filter, _frame_counter
            )

            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            jpeg_bytes = await loop.run_in_executor(
                _executor, encode_jpeg, annotated, JPEG_QUALITY
            )
            css.update(jpeg_bytes, labels, round(avg_fps, 1), round(cam.fps, 1))

    except asyncio.CancelledError:
        logger.info(f"[Cam {camera_id}] Inference loop cancelled")
    except Exception as e:
        logger.exception(f"[Cam {camera_id}] Inference loop error: {e}")


# ─── MJPEG stream endpoint ─────────────────────────────────────────────────────
@app.get("/stream/{camera_id}")
async def mjpeg_stream(camera_id: int):
    """
    Pure MJPEG HTTP stream.
    Browser renders with: <img src="/stream/0">
    Lowest possible latency — raw multipart bytes, native browser decoder, zero JS overhead.
    """
    cam = state.multi_camera.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    if not cam.is_connected:
        raise HTTPException(503, "Camera not connected")

    css = cam_stream_states[camera_id]

    async def generate():
        boundary = b"--mjpegboundary\r\n"
        last_jpeg = None
        while cam.is_connected:
            jpeg, _, _, _ = css.get()
            if jpeg is None or jpeg is last_jpeg:
                await asyncio.sleep(0.005)
                continue
            last_jpeg = jpeg
            header = (
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            )
            yield boundary + header + jpeg + b"\r\n"
            await asyncio.sleep(0.016)   # ~60fps push cap — browser limits anyway

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=mjpegboundary",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0",
            "Connection":    "keep-alive",
        }
    )


# ─── Metadata WebSocket (labels + fps only, no image) ─────────────────────────
@app.websocket("/ws/meta/{camera_id}")
async def websocket_meta(websocket: WebSocket, camera_id: int):
    """
    Sends only labels + FPS at 10Hz. No image data.
    Client sends config updates: {"classes": [...], "conf": 0.4}
    """
    await websocket.accept()
    cam = state.multi_camera.get_camera(camera_id)
    if not cam:
        await websocket.send_json({"error": f"Invalid camera_id: {camera_id}"})
        await websocket.close()
        return

    class_filter = []
    conf         = 0.40

    try:
        try:
            init = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
            class_filter = init.get("classes", [])
            conf         = init.get("conf", 0.40)
        except asyncio.TimeoutError:
            pass

        _start_infer_loop(camera_id, class_filter, conf)
        css = cam_stream_states[camera_id]

        while True:
            try:
                upd = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                if "classes" in upd or "conf" in upd:
                    class_filter = upd.get("classes", class_filter)
                    conf         = upd.get("conf", conf)
                    _start_infer_loop(camera_id, class_filter, conf)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            _, labels, fps, stream_fps = css.get()
            try:
                await websocket.send_json({
                    "labels":         labels,
                    "unique":         list(set(labels)),
                    "processing_fps": fps,
                    "stream_fps":     stream_fps,
                    "camera_id":      camera_id,
                })
            except Exception:
                break

            await asyncio.sleep(0.1)   # 10Hz metadata

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"Meta WS cam {camera_id}: {e}")


def _start_infer_loop(camera_id: int, class_filter: list, conf: float):
    existing = cam_infer_tasks.get(camera_id)
    if existing and not existing.done():
        existing.cancel()
    task = asyncio.ensure_future(_camera_infer_loop(camera_id, class_filter, conf))
    cam_infer_tasks[camera_id] = task

def _stop_infer_loop(camera_id: int):
    existing = cam_infer_tasks.get(camera_id)
    if existing and not existing.done():
        existing.cancel()
    cam_infer_tasks.pop(camera_id, None)


# ─── Legacy WebSocket (browser webcam) ────────────────────────────────────────
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    global _frame_counter
    await websocket.accept()
    try:
        loop = asyncio.get_event_loop()
        while True:
            data         = await websocket.receive_json()
            frame_b64    = data.get("frame", "")
            class_filter = data.get("classes", [])
            conf         = data.get("conf", 0.40)
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",")[1]
            img_data  = base64.b64decode(frame_b64)
            np_arr    = np.frombuffer(img_data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                await websocket.send_json({"error": "invalid frame"})
                continue
            _frame_counter += 1
            annotated, labels = await loop.run_in_executor(
                _executor, process_frame, frame_bgr, conf, class_filter, _frame_counter
            )
            jpeg = await loop.run_in_executor(_executor, encode_jpeg, annotated, JPEG_QUALITY)
            b64  = base64.b64encode(jpeg).decode()
            await websocket.send_json({
                "image":  f"data:image/jpeg;base64,{b64}",
                "labels": labels,
                "unique": list(set(labels)),
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"Webcam WS error: {e}")


# ─── REST endpoints ────────────────────────────────────────────────────────────
class RTSPConnectRequest(BaseModel):
    camera_id: int
    url: str
    label: str = ""

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "cctv_v4.html"
    return html_path.read_text(encoding="utf-8")

@app.get("/api/status")
async def get_status():
    return {
        "yolo":             state.yolo_model is not None,
        "face_recognition": state.fr_module is not None,
        "known_faces":      list(state.face_db.keys()),
        "models_loaded":    state.models_loaded,
        "error":            state.load_error or None,
        "device":           state.device,
        "tensorrt":         Path(ENGINE_PATH).exists(),
        "turbojpeg":        _turbo is not None,
        "all_classes":      ALL_YOLO_CLASSES,
        "class_colors":     CLASS_COLORS,
        "total_classes":    len(ALL_YOLO_CLASSES),
        "cameras":          state.multi_camera.get_all_status(),
        "max_cameras":      MAX_CAMERAS,
    }

@app.get("/api/classes")
async def get_all_classes():
    return {"classes": ALL_YOLO_CLASSES, "colors": CLASS_COLORS, "total": len(ALL_YOLO_CLASSES)}

@app.post("/api/cameras/connect")
async def camera_connect(req: RTSPConnectRequest):
    if req.camera_id < 0 or req.camera_id >= MAX_CAMERAS:
        raise HTTPException(400, f"camera_id must be 0-{MAX_CAMERAS-1}")
    if not req.url.strip():
        raise HTTPException(400, "RTSP URL is required")
    cam  = state.multi_camera.get_camera(req.camera_id)
    loop = asyncio.get_event_loop()
    ok   = await loop.run_in_executor(None, cam.connect, req.url.strip(), req.label)
    if not ok:
        raise HTTPException(503, cam.error or "Failed to connect")
    return cam.to_dict()

@app.post("/api/cameras/{camera_id}/disconnect")
async def camera_disconnect(camera_id: int):
    _stop_infer_loop(camera_id)
    cam = state.multi_camera.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    cam.disconnect()
    return {"camera_id": camera_id, "connected": False}

@app.get("/api/cameras")
async def get_cameras():
    return {"cameras": state.multi_camera.get_all_status()}

@app.get("/api/cameras/{camera_id}")
async def get_camera_status(camera_id: int):
    cam = state.multi_camera.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    return cam.to_dict()


# ─── Image detection ──────────────────────────────────────────────────────────
@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")
):
    class_filter = [c.strip() for c in classes.split(",") if c.strip()] if classes else []
    contents = await file.read()
    img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
    img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    annotated, labels = process_frame(img_bgr, conf_threshold=conf, class_filter=class_filter)
    jpeg = encode_jpeg(annotated, JPEG_QUALITY)
    b64  = base64.b64encode(jpeg).decode()
    return {"image": f"data:image/jpeg;base64,{b64}", "labels": labels, "unique": list(set(labels))}


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def _reencode_h264(in_path, out_path):
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-movflags", "+faststart", "-pix_fmt", "yuv420p", out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        return result.returncode == 0
    except Exception:
        return False

@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")
):
    class_filter = [c.strip() for c in classes.split(",") if c.strip()] if classes else []
    suffix    = Path(file.filename).suffix or ".mp4"
    tmp_dir   = tempfile.mkdtemp()
    try:
        in_path      = os.path.join(tmp_dir, f"input{suffix}")
        raw_out_path = os.path.join(tmp_dir, "annotated_raw.mp4")
        h264_path    = os.path.join(tmp_dir, "annotated.mp4")
        with open(in_path, "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(raw_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        all_labels  = set()
        frame_count = 0
        write_queue = []
        loop        = asyncio.get_event_loop()

        def write_batch(frames):
            for f in frames:
                out.write(f)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ann, lbls = await loop.run_in_executor(
                _executor, process_frame, frame, conf, class_filter, frame_count
            )
            all_labels.update(lbls)
            write_queue.append(ann)
            frame_count += 1
            if len(write_queue) >= 30:
                batch = write_queue[:]
                write_queue.clear()
                await loop.run_in_executor(_executor, write_batch, batch)

        if write_queue:
            await loop.run_in_executor(_executor, write_batch, write_queue)
        cap.release()
        out.release()

        if _ffmpeg_available():
            ok = await loop.run_in_executor(_executor, _reencode_h264, raw_out_path, h264_path)
            serve_path = h264_path if ok else raw_out_path
        else:
            serve_path = raw_out_path

        return FileResponse(
            serve_path, media_type="video/mp4", filename="annotated_video.mp4",
            headers={"X-Detections": ",".join(sorted(all_labels))}
        )
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, str(e))


# ─── Face management ──────────────────────────────────────────────────────────
@app.post("/api/faces/upload")
async def upload_faces(files: list[UploadFile] = File(...)):
    if not state.fr_module:
        raise HTTPException(503, "face_recognition not available")
    tmp_dir = tempfile.mkdtemp()
    saved   = []
    try:
        for uf in files:
            data = await uf.read()
            fp   = os.path.join(tmp_dir, uf.filename)
            with open(fp, "wb") as f:
                f.write(data)
            saved.append(fp)
        extra = build_face_database(tmp_dir, state.fr_module)
        state.face_db.update(extra)
        return {"enrolled": list(extra.keys()), "total_known": list(state.face_db.keys()), "failed": len(saved)-len(extra)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    if name not in state.face_db:
        raise HTTPException(404, f"'{name}' not found")
    del state.face_db[name]
    return {"deleted": name, "remaining": list(state.face_db.keys())}


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "cctv_v4:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        ws="wsproto",
        loop="uvloop",       # ~2x faster than asyncio default on Linux/Jetson
        http="httptools",    # faster HTTP parser
    )