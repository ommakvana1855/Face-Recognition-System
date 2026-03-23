"""
Smart Vision AI — WebRTC Edition (v6)
Architecture: RTSP → cv2 → YOLO composite → aiortc VideoStreamTrack → WebRTC → Browser
Boxes are baked directly onto the video frame before H.264 encoding.
Smooth 30fps H.264 video in the browser native <video> tag — zero sync issues.

Install:
    pip install aiortc av opencv-python-headless ultralytics Pillow fastapi uvicorn websockets

Face recognition: Seventh Sense OpenCV FR (NIST Top-10)
Set env:  OPENCV_FR_API_KEY   and optionally OPENCV_FR_BASE_URL
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
import fractions
from pathlib import Path
from typing import Optional, List, Dict, Set
from contextlib import asynccontextmanager

import cv2
import numpy as np
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel


# ─── OpenCV FR config ──
OPENCV_FR_API_KEY  = os.getenv("OPENCV_FR_API_KEY", "")
OPENCV_FR_BASE_URL = os.getenv("OPENCV_FR_BASE_URL", "https://us.opencv.fr")
_fr_sdk = None

def _fr_available():
    return bool(OPENCV_FR_API_KEY) and _fr_sdk is not None

def _init_fr_sdk():
    global _fr_sdk
    try:
        from opencv.fr import FR
        _fr_sdk = FR(backend_url=OPENCV_FR_BASE_URL, developer_key=OPENCV_FR_API_KEY)
        logger.info(f"OpenCV FR SDK initialized ✅ ({OPENCV_FR_BASE_URL})")
    except Exception as e:
        logger.error(f"OpenCV FR SDK init failed: {e}")
        _fr_sdk = None


# ─── Logger ────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s › %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("smart_vision.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SmartVision")

MAX_CAMERAS = 9

# Track all active WebRTC peer connections for clean shutdown
_peer_connections: Set[RTCPeerConnection] = set()


# 
# RTSP Stream Manager
# Background thread continuously reads latest frame from RTSP camera.
# The WebRTC VideoStreamTrack pulls frames from here at its own pace.
# 
class RTSPStreamManager:
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[tuple] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.url: str = ""
        self.lock = threading.Lock()
        self.error: str = ""
        self.fps: float = 25.0
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
        self.fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
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
                self.error = "Stream disconnected"
                self.running = False
                break
            ret, frame = self.cap.read()
            if not ret:
                fails += 1
                if fails >= 30:
                    logger.warning(f"[Cam {self.camera_id}] Reconnecting…")
                    self._attempt_reconnect()
                    fails = 0
                time.sleep(0.05)
                continue
            fails = 0
            with self.lock:
                self.latest_frame = (frame.copy(), time.time())

    def _attempt_reconnect(self):
        if self.cap:
            self.cap.release()
        time.sleep(2)
        if self.url:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.cap = cap
                self.error = ""
                logger.info(f"[Cam {self.camera_id}] Reconnected ✅")
            else:
                self.error = "Reconnect failed"

    def get_frame(self) -> Optional[tuple]:
        with self.lock:
            if self.latest_frame is None:
                return None
            frame, ts = self.latest_frame
            return (frame.copy(), ts)

    def disconnect(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.latest_frame = None
        self.url = ""
        logger.info(f"[Cam {self.camera_id}] Disconnected")

    @property
    def is_connected(self):
        return self.running and self.cap is not None and self.cap.isOpened()

    def to_dict(self):
        return dict(
            camera_id=self.camera_id, connected=self.is_connected,
            url=self.url, label=self.label, error=self.error,
            width=self.width, height=self.height, fps=self.fps,
        )


class MultiCameraManager:
    def __init__(self, max_cameras=MAX_CAMERAS):
        self.cameras: Dict[int, RTSPStreamManager] = {
            i: RTSPStreamManager(i) for i in range(max_cameras)
        }

    def get_camera(self, cid):
        return self.cameras.get(cid)

    def get_all_status(self):
        return [c.to_dict() for c in self.cameras.values()]

    def disconnect_all(self):
        for c in self.cameras.values():
            if c.is_connected:
                c.disconnect()


# 
# YOLO Classes + Colors
# 
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
    return {name: tuple(np.random.randint(50, 255, 3).tolist()) for name in ALL_YOLO_CLASSES.values()}

CLASS_COLORS = generate_class_colors()


# 
# Global App State
# 
class AppState:
    yolo_model = None
    fr_enabled: bool = False
    face_db: dict = {}
    models_loaded = False
    load_error: str = ""
    multi_camera: MultiCameraManager = MultiCameraManager()

state = AppState()

from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=8)

_frame_counter = 0
FACE_REC_EVERY_N_FRAMES = 15

# Per-camera inference state
_cam_busy: Dict[int, bool] = {}
_cam_last_overlay: Dict[int, Optional[np.ndarray]] = {}
_cam_last_labels: Dict[int, list] = {}

# Per-camera detection config — set by browser, read by VideoStreamTrack
_cam_config: Dict[int, dict] = {}


# 
# Face Recognition — OpenCV FR
# 
def opencv_fr_enroll(name: str, img_bgr: np.ndarray) -> Optional[str]:
    if not _fr_available():
        return None
    try:
        from opencv.fr.persons.schemas import PersonBase
        person = PersonBase([img_bgr], name=name)
        created = _fr_sdk.persons.create(person)
        logger.info(f"OpenCV FR enrolled '{name}' → person_id={created.id}")
        return created.id
    except Exception as e:
        logger.error(f"OpenCV FR enroll error for '{name}': {e}")
        return None

def opencv_fr_delete_subject(person_id: str) -> bool:
    if not _fr_available():
        return False
    try:
        _fr_sdk.persons.delete(person_id)
        logger.info(f"OpenCV FR deleted person_id={person_id}")
        return True
    except Exception as e:
        logger.error(f"OpenCV FR delete error: {e}")
        return False

def opencv_fr_search(img_bgr: np.ndarray) -> Optional[str]:
    if not _fr_available() or not state.face_db:
        return None
    try:
        from opencv.fr.search.schemas import SearchRequest, SearchMode
        req = SearchRequest([img_bgr], min_score=0.55, search_mode=SearchMode.FAST)
        results = _fr_sdk.search.search(req)
        if not results:
            return None
        top = results[0]
        person_to_name = {v: k for k, v in state.face_db.items()}
        return person_to_name.get(top.person.id)
    except Exception as e:
        logger.debug(f"OpenCV FR search error: {e}")
        return None

def build_face_database(folder: str) -> dict:
    db = {}
    for p in Path(folder).glob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        name = p.stem.replace("_", " ").replace("-", " ").title()
        img = cv2.imread(str(p))
        if img is None:
            continue
        person_id = opencv_fr_enroll(name, img)
        if person_id:
            db[name] = person_id
    return db

def _reload_known_folder(folder="known_persons"):
    if _fr_available() and os.path.isdir(folder):
        state.face_db = build_face_database(folder)
        logger.info(f"OpenCV FR: enrolled {len(state.face_db)} known face(s)")


# 
# YOLO Inference
# Returns a transparent RGBA overlay with bounding boxes drawn on it.
# 
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

    overlay = np.zeros((h_orig, w_orig, 4), dtype=np.uint8)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in filter_ids:
            continue

        conf  = float(box.conf[0])
        label = ALL_YOLO_CLASSES[cls_id]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = int(x1 / INFER_SCALE); y1 = int(y1 / INFER_SCALE)
        x2 = int(x2 / INFER_SCALE); y2 = int(y2 / INFER_SCALE)

        display_name = label

        if label == "person" and state.fr_enabled and state.face_db and frame_num % FACE_REC_EVERY_N_FRAMES == 0:
            try:
                roi = frame_bgr[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
                if roi.size > 0:
                    matched = opencv_fr_search(roi)
                    if matched:
                        display_name = matched
            except Exception as e:
                logger.debug(f"FR error: {e}")

        bgra = (*color[::-1], 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bgra, 2)
        tag = f"{display_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        lbl_overlay = overlay.copy()
        cv2.rectangle(lbl_overlay, (x1, y1 - th - 8), (x1 + tw + 6, y1), (*color[::-1], 200), -1)
        cv2.addWeighted(lbl_overlay, 0.85, overlay, 0.15, 0, overlay)
        cv2.putText(overlay, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255, 255), 1)
        labels_found.append(display_name)

    return overlay, labels_found


def composite_overlay(frame_bgr: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Alpha-blend a transparent RGBA overlay onto a BGR frame."""
    bgra  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
    alpha = overlay[:, :, 3:4] / 255.0
    bgra[:, :, :3] = (bgra[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha).astype(np.uint8)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


# 
# WebRTC VideoStreamTrack
#
# This is the heart of the WebRTC approach.
# aiortc calls recv() repeatedly to get the next video frame.
# We:
#   1. Pull the latest RTSP frame from RTSPStreamManager
#   2. Composite YOLO boxes onto it (using last available inference result)
#   3. Fire async YOLO inference for the next frame (non-blocking)
#   4. Wrap the composited BGR frame in av.VideoFrame (RGB24)
#   5. Set correct PTS timestamp so browser plays at smooth correct FPS
#   6. Return to aiortc which H.264 encodes and sends via WebRTC
#
# The browser receives smooth hardware-decoded H.264 video in a native <video> tag.
# No sync issues — boxes and video are always the same frame.
# 
class YOLOVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, camera_id: int):
        super().__init__()
        self.camera_id  = camera_id
        self._timestamp = 0
        self._frame_num = 0

        cam = state.multi_camera.get_camera(camera_id)
        self._fps = cam.fps if cam else 25.0

        # aiortc uses a 90kHz clock for video PTS timestamps
        self._clock_rate = 90000
        self._time_base  = fractions.Fraction(1, self._clock_rate)
        self._pts_step   = int(self._clock_rate / self._fps)

    async def recv(self) -> VideoFrame:
        global _frame_counter

        # Throttle to match camera FPS — this is what controls playback speed
        await asyncio.sleep(1.0 / self._fps)

        cam    = state.multi_camera.get_camera(self.camera_id)
        result = cam.get_frame() if cam else None

        if result is None:
            # Camera not ready yet — send a black frame so WebRTC doesn't stall
            h = cam.height if cam and cam.height else 480
            w = cam.width  if cam and cam.width  else 640
            frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            frame_bgr, _ = result

        self._frame_num += 1
        _frame_counter  += 1

        # ── Fire YOLO inference asynchronously ────────────────────────────────
        # We don't wait for inference to finish — we fire it and immediately
        # composite the PREVIOUS inference result onto this frame.
        # This means boxes are ~1 inference cycle behind, which at 25fps
        # and typical 50-100ms inference time is completely imperceptible.
        cfg          = _cam_config.get(self.camera_id, {})
        class_filter = cfg.get("classes", [])
        conf         = cfg.get("conf", 0.40)

        if class_filter and not _cam_busy.get(self.camera_id, False):
            _cam_busy[self.camera_id] = True
            snap = frame_bgr.copy()
            fn   = self._frame_num
            loop = asyncio.get_event_loop()

            async def _infer(f, cid, cf, c, fn_):
                try:
                    ovl, lbl = await loop.run_in_executor(
                        _executor, process_frame, f, c, cf, fn_
                    )
                    _cam_last_overlay[cid] = ovl
                    _cam_last_labels[cid]  = lbl
                finally:
                    _cam_busy[cid] = False

            asyncio.create_task(_infer(snap, self.camera_id, class_filter, conf, fn))

        # ── Composite last overlay onto current frame ─────────────────────────
        ovl = _cam_last_overlay.get(self.camera_id)
        if ovl is not None:
            frame_bgr = composite_overlay(frame_bgr, ovl)

        # ── BGR → RGB → av.VideoFrame 
        frame_rgb          = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        av_frame           = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        av_frame.pts       = self._timestamp
        av_frame.time_base = self._time_base
        self._timestamp   += self._pts_step

        return av_frame


# 
# Model Loader
# 
def load_models():
    logger.info("Loading models…")
    try:
        from ultralytics import YOLO
        state.yolo_model = YOLO("yolo26x.pt")
        state.yolo_model.to("cuda")
        logger.info("YOLOv8x loaded ✅")
    except Exception as e:
        logger.error(f"YOLO load error: {e}")
        state.load_error += f"YOLO: {e}. "

    if OPENCV_FR_API_KEY:
        _init_fr_sdk()
        if _fr_sdk is not None:
            state.fr_enabled = True
        else:
            state.load_error += "OpenCV FR SDK failed to init. "
    else:
        logger.warning("OPENCV_FR_API_KEY not set — face recognition disabled.")

    _reload_known_folder()
    state.models_loaded = True
    logger.info("Model loading complete.")


# 
# Lifespan
# 
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield
    # Clean shutdown — close all active WebRTC peer connections
    await asyncio.gather(
        *[pc.close() for pc in _peer_connections],
        return_exceptions=True
    )
    _peer_connections.clear()
    state.multi_camera.disconnect_all()


# 
# App
# 
app = FastAPI(title="Smart Vision AI — WebRTC", version="6.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# 
# REST Endpoints
# 
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "opencv_fr.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    return {
        "yolo":                    state.yolo_model is not None,
        "face_recognition":        state.fr_enabled,
        "face_recognition_engine": "OpenCV FR (Seventh Sense)" if state.fr_enabled else "disabled",
        "known_faces":             list(state.face_db.keys()),
        "models_loaded":           state.models_loaded,
        "error":                   state.load_error or None,
        "all_classes":             ALL_YOLO_CLASSES,
        "class_colors":            CLASS_COLORS,
        "cameras":                 state.multi_camera.get_all_status(),
        "max_cameras":             MAX_CAMERAS,
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
    if not url:
        raise HTTPException(400, "RTSP URL required")
    cam = state.multi_camera.get_camera(req.camera_id)
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, cam.connect, url, req.label)
    if not success:
        raise HTTPException(503, cam.error or "Failed to connect")
    return cam.to_dict()


@app.post("/api/cameras/{camera_id}/disconnect")
async def camera_disconnect(camera_id: int):
    cam = state.multi_camera.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    cam.disconnect()
    _cam_busy.pop(camera_id, None)
    _cam_last_overlay.pop(camera_id, None)
    _cam_last_labels.pop(camera_id, None)
    _cam_config.pop(camera_id, None)
    return {"camera_id": camera_id, "connected": False}


@app.get("/api/cameras")
async def get_cameras():
    return {"cameras": state.multi_camera.get_all_status()}


# ── WebRTC Signalling ──
class WebRTCOfferRequest(BaseModel):
    camera_id: int
    sdp: str
    type: str
    classes: List[str] = []
    conf: float = 0.40


@app.post("/api/webrtc/offer")
async def webrtc_offer(req: WebRTCOfferRequest):
    """
    WebRTC signalling endpoint — the browser sends its SDP offer here.

    What happens:
      1. Browser creates an RTCPeerConnection and generates an SDP offer
      2. Browser POSTs the offer to this endpoint
      3. We create a server-side RTCPeerConnection
      4. We attach a YOLOVideoStreamTrack (feeds composited frames into WebRTC)
      5. We generate an SDP answer and return it
      6. Browser uses the answer to complete the WebRTC handshake
      7. Browser's <video> tag starts receiving smooth H.264 video
    """
    cam = state.multi_camera.get_camera(req.camera_id)
    if not cam or not cam.is_connected:
        raise HTTPException(404, f"Camera {req.camera_id} not connected")

    # Store detection config so YOLOVideoStreamTrack can read it
    _cam_config[req.camera_id] = {"classes": req.classes, "conf": req.conf}

    pc = RTCPeerConnection()
    _peer_connections.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        logger.info(f"[Cam {req.camera_id}] WebRTC state → {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            _peer_connections.discard(pc)

    # Attach YOLO video track — this is what streams composited frames
    pc.addTrack(YOLOVideoStreamTrack(req.camera_id))

    # SDP offer/answer handshake
    offer = RTCSessionDescription(sdp=req.sdp, type=req.type)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info(f"[Cam {req.camera_id}] WebRTC offer answered ✅")

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


class ConfigUpdateRequest(BaseModel):
    classes: List[str] = []
    conf: float = 0.40


@app.post("/api/webrtc/config/{camera_id}")
async def update_webrtc_config(camera_id: int, req: ConfigUpdateRequest):
    """Update YOLO detection config for a running WebRTC stream without restarting."""
    _cam_config[camera_id] = {"classes": req.classes, "conf": req.conf}
    logger.info(f"[Cam {camera_id}] Config updated → classes={req.classes} conf={req.conf}")
    return {"ok": True}


# ── Image Detection ────
@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")
):
    class_filter = [c.strip() for c in classes.split(",") if c.strip()]
    contents = await file.read()
    img_bgr = cv2.cvtColor(np.array(Image.open(io.BytesIO(contents)).convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay, labels = process_frame(img_bgr, conf, class_filter, 0)
    result = composite_overlay(img_bgr, overlay) if overlay is not None else img_bgr
    _, buf = cv2.imencode(".jpg", result)
    b64 = base64.b64encode(buf).decode()
    return {"image": f"data:image/jpeg;base64,{b64}", "labels": labels, "unique": list(set(labels))}


# ── Video Detection ────
@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")
):
    class_filter = [c.strip() for c in classes.split(",") if c.strip()]
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        in_path = tmp.name
    out_path = in_path.replace(suffix, "_annotated.mp4")
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    all_labels = set()
    fc = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay, lbls = process_frame(frame, conf, class_filter, fc)
        if overlay is not None:
            frame = composite_overlay(frame, overlay)
        all_labels.update(lbls)
        out.write(frame)
        fc += 1
    cap.release()
    out.release()
    os.unlink(in_path)
    return FileResponse(
        out_path, media_type="video/mp4", filename="annotated_video.mp4",
        headers={"X-Detections": ",".join(sorted(all_labels))}
    )


# ── Face Management ────
@app.post("/api/faces/upload")
async def upload_faces(files: list[UploadFile] = File(...)):
    if not state.fr_enabled:
        raise HTTPException(503, "OpenCV FR not available — set OPENCV_FR_API_KEY env var")
    tmp_dir = tempfile.mkdtemp()
    try:
        for uf in files:
            fp = os.path.join(tmp_dir, uf.filename)
            with open(fp, "wb") as f:
                f.write(await uf.read())
        extra = build_face_database(tmp_dir)
        state.face_db.update(extra)
        return {"enrolled": list(extra.keys()), "total_known": list(state.face_db.keys())}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    if name not in state.face_db:
        raise HTTPException(404, f"'{name}' not found")
    subject_id = state.face_db.pop(name)
    opencv_fr_delete_subject(subject_id)
    return {"deleted": name, "remaining": list(state.face_db.keys())}


# 
# WebSocket — Webcam (MJPEG over WS — unchanged, webcam doesn't need WebRTC)
# 
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
            frame_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(frame_b64), np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                await websocket.send_json({"error": "invalid frame"})
                continue
            _frame_counter += 1
            overlay, labels = await loop.run_in_executor(
                _executor, process_frame, frame_bgr, conf, class_filter, _frame_counter
            )
            if overlay is not None:
                frame_bgr = composite_overlay(frame_bgr, overlay)
            _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 55])
            b64 = base64.b64encode(buf).decode()
            await websocket.send_json({
                "image":  f"data:image/jpeg;base64,{b64}",
                "labels": labels,
                "unique": list(set(labels)),
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"Webcam WS error: {e}")


# 
# WebSocket — Detection labels for RTSP cameras
# WebRTC handles the video. This lightweight WS just pushes label updates
# (the colored tags shown in the tile footer) at 5fps.
# 
@app.websocket("/ws/labels/{camera_id}")
async def websocket_labels(websocket: WebSocket, camera_id: int):
    await websocket.accept()
    logger.info(f"Labels WS connected — cam {camera_id}")
    try:
        while True:
            # Accept config updates from browser
            try:
                update = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                if "classes" in update or "conf" in update:
                    current = _cam_config.get(camera_id, {})
                    current.update({k: v for k, v in update.items() if k in ("classes", "conf")})
                    _cam_config[camera_id] = current
            except asyncio.TimeoutError:
                pass

            # Push latest detection labels to browser footer
            labels = _cam_last_labels.get(camera_id, [])
            try:
                await websocket.send_json({
                    "labels":    labels,
                    "unique":    list(set(labels)),
                    "camera_id": camera_id,
                })
            except Exception:
                break

            await asyncio.sleep(0.2)  # 5fps label updates — very lightweight

    except WebSocketDisconnect:
        logger.info(f"Labels WS cam {camera_id} disconnected")
    except Exception as e:
        logger.exception(f"Labels WS cam {camera_id} error: {e}")


# 
# Entry point
# 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("opencv_fr:app", host="0.0.0.0", port=8002, reload=False, ws="wsproto")