"""
Search-Based Object Detection with YOLOE Text-Prompt Detection
FastAPI Backend with dynamic text-prompt class filtering
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

# ─── Global state ─────────────────────────────────────────────────────────────
class AppState:
    yoloe_model = None           # Base YOLOE model (reused across requests)
    fr_module = None             # face_recognition module
    face_db: dict = {}           # Known face encodings
    models_loaded = False
    load_error: str = ""
    _current_classes: List[str] = []   # Classes the model is currently set to
    _text_pe_cache: dict = {}          # Cache: frozenset(classes) → text_pe tensor

state = AppState()

from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=3)

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


# ─── Model loader ────────────────────────────────────────────────────────────
def load_models():
    """Load YOLOE and face_recognition models once at startup."""
    logger.info("Loading AI models…")
    try:
        from ultralytics import YOLOE
        # Load base YOLOE model — no classes baked in yet
        state.yoloe_model = YOLOE("yoloe-26x-seg.pt")
        logger.info("YOLOE loaded ✅")
    except Exception as e:
        logger.error(f"YOLOE load error: {e}")
        state.load_error += f"YOLOE: {e}. "

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


def _get_text_pe(names: List[str]):
    """
    Return cached text positional embeddings for `names`, computing them only
    once per unique set of class names.
    """
    key = frozenset(names)
    if key not in state._text_pe_cache:
        logger.info(f"Computing text embeddings for: {names}")
        state._text_pe_cache[key] = state.yoloe_model.get_text_pe(names)
    return state._text_pe_cache[key]


def _configure_model_for_classes(names: List[str]):
    """
    Reconfigure the YOLOE model for a new set of text-prompt classes.
    Uses caching to avoid redundant embedding computation.
    """
    if names == state._current_classes:
        return  # Already configured — no-op

    text_pe = _get_text_pe(names)
    state.yoloe_model.set_classes(names, text_pe)
    state._current_classes = list(names)
    logger.info(f"YOLOE reconfigured for classes: {names}")


# ─── Color generation ─────────────────────────────────────────────────────────
_color_cache: dict = {}

def get_class_color(class_name: str) -> tuple:
    """Return a consistent color for a class name (generated on demand)."""
    if class_name not in _color_cache:
        # Deterministic color from class name hash
        rng = np.random.RandomState(abs(hash(class_name)) % (2**31))
        _color_cache[class_name] = tuple(rng.randint(50, 255, 3).tolist())
    return _color_cache[class_name]


# ─── Lifespan context manager ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="YOLOE Text-Prompt Object Detection",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Detection ────────────────────────────────────────────────────────────────
_frame_counter = 0
FACE_REC_EVERY_N_FRAMES = 15

def process_frame(
    frame_bgr: np.ndarray,
    conf_threshold: float = 0.40,
    class_filter: Optional[List[str]] = None,
    frame_num: int = 0
):
    """
    Process a frame with YOLOE text-prompt detection.

    class_filter  – list of free-text class names, e.g. ["person", "tiger", "phone"].
                    ANY natural-language label is valid; YOLOE handles it via CLIP embeddings.
    If class_filter is None or empty, the original frame is returned untouched.
    Face recognition is applied ONLY when 'person' (or similar) is among the detected labels.
    """
    labels_found = []

    if state.yoloe_model is None:
        return frame_bgr, labels_found

    if not class_filter:
        return frame_bgr, labels_found

    # ── Reconfigure model for requested classes (cached) ──────────────────
    try:
        _configure_model_for_classes(class_filter)
    except Exception as e:
        logger.error(f"set_classes error: {e}")
        return frame_bgr, labels_found

    # ── Downscale for faster inference ────────────────────────────────────
    INFER_SCALE = 0.35
    h_orig, w_orig = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (int(w_orig * INFER_SCALE), int(h_orig * INFER_SCALE)))

    try:
        results = state.yoloe_model(small, conf=conf_threshold, verbose=False)[0]
    except Exception as e:
        logger.error(f"YOLOE inference error: {e}")
        return frame_bgr, labels_found

    # ── Draw detections ───────────────────────────────────────────────────
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])

        # YOLOE names() returns the class list set via set_classes()
        try:
            label = results.names[cls_id]
        except (KeyError, IndexError):
            label = class_filter[cls_id] if cls_id < len(class_filter) else "unknown"

        color = get_class_color(label)

        # Scale bounding box back to original size
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = int(x1 / INFER_SCALE); y1 = int(y1 / INFER_SCALE)
        x2 = int(x2 / INFER_SCALE); y2 = int(y2 / INFER_SCALE)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        display_name = label

        # ── Face recognition: for any "person"-like label ─────────────────
        run_face_rec = (frame_num % FACE_REC_EVERY_N_FRAMES == 0)
        person_labels = {"person", "human", "man", "woman", "boy", "girl", "face"}
        if label.lower() in person_labels and state.fr_module and state.face_db and run_face_rec:
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
    html_path = Path(__file__).parent / "templates" / "index2_V2.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    """Return system status."""
    return {
        "yoloe": state.yoloe_model is not None,
        "face_recognition": state.fr_module is not None,
        "known_faces": list(state.face_db.keys()),
        "models_loaded": state.models_loaded,
        "error": state.load_error or None,
        "current_classes": state._current_classes,
        "note": "YOLOE accepts ANY free-text class name via text prompting (no fixed class list)."
    }


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")   # Comma-separated free-text class names
):
    """
    Detect objects matching free-text class names in an uploaded image.
    Example classes: "person, dog, red car, coffee cup"
    """
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
        "prompted_classes": class_filter
    }


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    classes: str = Form("")
):
    """Process an uploaded video with YOLOE text-prompt detection."""
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
    )


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


# ─── WebSocket: live frame processing ─────────────────────────────────────────
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    Client sends: {"frame": <base64_jpeg>, "classes": ["phone", "tiger", "red car"]}
    Server responds: {"image": <base64_jpeg>, "labels": [...], "unique": [...]}

    classes accepts ANY free-text prompts — no fixed vocabulary required.
    """
    global _frame_counter
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        loop = asyncio.get_event_loop()
        while True:
            data = await websocket.receive_json()

            frame_b64   = data.get("frame", "")
            class_filter = data.get("classes", [])
            conf        = data.get("conf", 0.40)

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
    finally:
        logger.info("WebSocket handler cleaned up")

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("allclass_V2:app", host="0.0.0.0", port=8001, reload=False)
    # uvicorn.run("allclass_V2:app", reload=False)
