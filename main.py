"""
Smart Vision AI — FastAPI Backend
- YOLOv8 object detection
- Face recognition
- WebSocket live stream processing
- REST endpoints for image/video upload
"""

import asyncio
import base64
import io
import logging
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)-8s %(name)s › %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("smart_vision.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SmartVisionAI")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Vision AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── COCO target classes ───────────────────────────────────────────────────────
TARGET_CLASSES = {
    39: "bottle",
    0:  "person",
    67: "cell phone",
    63: "laptop",
    58: "potted plant",
}

CLASS_COLORS = {
    "bottle":       (0, 200, 255),
    "person":       (255, 50, 200),
    "cell phone":   (50, 255, 150),
    "laptop":       (255, 200, 0),
    "potted plant": (0, 255, 80),
}

# ─── Global state ─────────────────────────────────────────────────────────────
class AppState:
    yolo_model = None
    fr_module = None
    face_db: dict = {}
    models_loaded = False
    load_error: str = ""

state = AppState()
from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=3)

def load_models():
    """Load YOLO and face_recognition models once at startup."""
    logger.info("Loading AI models…")
    try:
        from ultralytics import YOLO
        state.yolo_model = YOLO("yolov8x.pt")
        # state.yolo_model = YOLO("Face-Recognition-System/yolo26x.pt")
        # state.yolo_model = YOLO("yolov8x.engine", device="dla:0")
        # state.yolo_model.export(format="engine", device=0, int8=True)
        # state.yolo_model = YOLO("yolov8x.engine")
        # state.yolo_model.to("cuda")
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


def _reload_known_folder(folder="known_persons"):
    if state.fr_module and os.path.isdir(folder):
        state.face_db = build_face_database(folder, state.fr_module)
        logger.info(f"Loaded {len(state.face_db)} known face(s) from '{folder}'")


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)


# ─── Face helpers ──────────────────────────────────────────────────────────────
def load_single_face_encoding(img_path: str, fr_module):
    try:
        # Use PIL exactly like the Streamlit version does in process_frame
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


# ─── Detection ─────────────────────────────────────────────────────────────────
_frame_counter = 0
FACE_REC_EVERY_N_FRAMES = 15

def process_frame(frame_bgr: np.ndarray, conf_threshold: float = 0.40, frame_num: int = 0):
    labels_found = []
    if state.yolo_model is None:
        return frame_bgr, labels_found

    # ── Downscale for faster inference ────────────────────────────────────────
    INFER_SCALE = 0.35
    h_orig, w_orig = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (int(w_orig * INFER_SCALE), int(h_orig * INFER_SCALE)))

    frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    # frame_rgb_orig = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        results = state.yolo_model(small, conf=conf_threshold, verbose=False)[0]
    except Exception as e:
        logger.error(f"YOLO inference error: {e}")
        return frame_bgr, labels_found

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in TARGET_CLASSES:
            continue

        conf = float(box.conf[0])
        label = TARGET_CLASSES[cls_id]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        # Scale bounding box coords back to original frame size
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = int(x1 / INFER_SCALE); y1 = int(y1 / INFER_SCALE)
        x2 = int(x2 / INFER_SCALE); y2 = int(y2 / INFER_SCALE)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        display_name = label

        # ── Face recognition: only every N frames ─────────────────────────────
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
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    return {
        "yolo": state.yolo_model is not None,
        "face_recognition": state.fr_module is not None,
        "known_faces": list(state.face_db.keys()),
        "models_loaded": state.models_loaded,
        "error": state.load_error or None,
    }


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = 0.40,
):
    """Detect objects in an uploaded image. Returns annotated image as base64."""
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    annotated, labels = process_frame(img_bgr, conf_threshold=conf)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode()

    return {
        "image": f"data:image/jpeg;base64,{b64}",
        "labels": labels,
        "unique": list(set(labels)),
    }


@app.post("/api/faces/upload")
async def upload_faces(files: list[UploadFile] = File(...)):
    """Upload known face images to enrol persons."""
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
    """Remove a known person from the face database."""
    if name not in state.face_db:
        raise HTTPException(404, f"'{name}' not found in face database")
    del state.face_db[name]
    return {"deleted": name, "remaining": list(state.face_db.keys())}


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...), conf: float = 0.40):
    """
    Process an uploaded video. Returns annotated video as downloadable file.
    For large files, consider using a background task / queue in production.
    """
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ann, lbls = process_frame(frame, conf_threshold=conf)
        all_labels.update(lbls)
        out.write(ann)

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


# ─── WebSocket: live frame processing ─────────────────────────────────────────
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    Client sends raw JPEG frames as binary messages.
    Server responds with JSON: { image: <base64 jpeg>, labels: [...] }

    Optimisations applied:
      • Frame-drop queue  — drops stale frames so processing never falls behind
      • Downscaled inference — YOLO runs on 50 % sized frame, boxes rescaled back
      • Face-rec throttle — face recognition runs every 5th frame only
      • Lower JPEG quality — 65 instead of 75 for faster encode/transfer
    """
    global _frame_counter
    await websocket.accept()
    logger.info("WebSocket client connected")

    # Queue holds at most 2 frames; older frames are dropped when full
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)

    async def _receiver():
        """Continuously read incoming frames; drop oldest when queue is full."""
        try:
            while True:
                data = await websocket.receive_bytes()
                if queue.full():
                    try:
                        queue.get_nowait()   # discard stale frame
                        # logger.debug("Dropped stale frame (queue full)")
                    except asyncio.QueueEmpty:
                        pass
                await queue.put(data)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"Receiver task error: {e}")

    receiver_task = asyncio.create_task(_receiver())

    try:
        loop = asyncio.get_event_loop()
        while True:
            # Wait for the next frame to process
            try:
                data = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.info("WebSocket idle timeout — closing")
                break

            np_arr = np.frombuffer(data, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                await websocket.send_json({"error": "invalid frame"})
                continue

            _frame_counter += 1

            # Run heavy inference in thread-pool so the event loop stays free
            annotated, labels = await loop.run_in_executor(_executor, process_frame, frame_bgr, 0.40, _frame_counter)

            # Encode at quality 65 — visually fine, noticeably smaller payload
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
            b64 = base64.b64encode(buf).decode()

            try:
                await websocket.send_json({
                    "image": f"data:image/jpeg;base64,{b64}",
                    "labels": labels,
                })
            except Exception:
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        receiver_task.cancel()
        logger.info("WebSocket handler cleaned up")


# ─── Dev entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)