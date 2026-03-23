"""
SAM3-Powered Multi-Camera CCTV Detection System
FastAPI Backend — Text prompts + Bounding box prompts + Face Recognition + Multi-RTSP
Uses: sam3 (facebookresearch), face_recognition, OpenCV, FastAPI, WebSockets
"""

import asyncio
import base64
import io
import logging
import os
import sys
import tempfile
import einops
import shutil
import threading
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form
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
        logging.FileHandler("sam3_cctv.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SAM3-CCTV")

MAX_CAMERAS = 9

# ─── Face Recognition Config ──────────────────────────────────────────────────
FR_DETECTION_MODEL  = "cnn"
FR_UPSAMPLE         = 2
FR_ENROLL_JITTERS   = 100
FR_LIVE_JITTERS     = 5
FR_DISTANCE_UNKNOWN = 0.45
FR_TOP_K_VOTES      = 3

# ─── SAM3 Config ──────────────────────────────────────────────────────────────
SAM3_CONF_THRESHOLD = 0.30   # detection confidence
SAM3_MASK_THRESHOLD = 0.50   # mask binarisation threshold
FACE_REC_EVERY_N    = 15     # run face rec every N frames (performance)
INFER_SCALE         = 0.50   # resize before SAM3 inference, rescale boxes back


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
                self.latest_frame = frame.copy()

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

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

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
# Face Recognition Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _preprocess_for_face_rec(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)


def load_single_face_encoding(img_path: str, fr_module, jitters=FR_ENROLL_JITTERS):
    try:
        img_rgb = np.ascontiguousarray(
            np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        )
        img_rgb = _preprocess_for_face_rec(img_rgb)
        locs = fr_module.face_locations(img_rgb, number_of_times_to_upsample=FR_UPSAMPLE, model=FR_DETECTION_MODEL)
        if not locs: return None
        encs = fr_module.face_encodings(img_rgb, known_face_locations=locs, num_jitters=jitters)
        return encs[0] if encs else None
    except Exception as e:
        logger.exception(f"Error encoding {img_path}: {e}"); return None


def build_face_database(folder: str, fr_module) -> dict:
    raw: dict = {}
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in Path(folder).glob("*"):
        if p.suffix.lower() not in supported: continue
        stem = re.sub(r"_\d+$", "", p.stem)
        name = stem.replace("_", " ").replace("-", " ").title()
        enc = load_single_face_encoding(str(p), fr_module, jitters=FR_ENROLL_JITTERS)
        if enc is not None:
            raw.setdefault(name, []).append(enc)
            logger.info(f"Enrolled photo for '{name}' ({p.name})")
    db = {}
    for name, encs in raw.items():
        db[name] = np.mean(encs, axis=0)
        logger.info(f"'{name}': averaged {len(encs)} encoding(s)")
    return db


# ═══════════════════════════════════════════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════════════════════════════════════════
class AppState:
    # SAM3 image model (for text + box prompts on images)
    sam3_image_model   = None
    sam3_processor     = None          # Sam3Processor wrapping the image model

    # SAM3 video/streaming predictor (for RTSP concept tracking)
    sam3_video_predictor = None

    # Face recognition
    fr_module  = None
    face_db: dict = {}

    models_loaded = False
    load_error: str = ""
    multi_camera: MultiCameraManager = MultiCameraManager()

    # Per-camera SAM3 video sessions  { camera_id: session_id }
    cam_sessions: Dict[int, str] = {}


state = AppState()
_executor = ThreadPoolExecutor(max_workers=8)
_frame_counter = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loader
# ═══════════════════════════════════════════════════════════════════════════════
def load_models():
    logger.info("Loading SAM3 models…")

    # ── SAM3 Image model ──────────────────────────────────────────────────────
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Build absolute BPE path dynamically (portable)
        base_dir = Path(__file__).resolve().parent
        bpe_path = base_dir / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        state.sam3_image_model = build_sam3_image_model(
            device="cuda",
            load_from_HF=True,
            bpe_path=str(bpe_path)
        )

        state.sam3_processor = Sam3Processor(state.sam3_image_model)
        logger.info("SAM3 image model loaded ✅")

    except Exception as e:
        logger.error(f"SAM3 image model load error: {e}")
        state.load_error += f"SAM3-Image: {e}. "

    # ── SAM3 Video predictor ──────────────────────────────────────────────────
    try:
        from sam3.model_builder import build_sam3_video_predictor
        state.sam3_video_predictor = build_sam3_video_predictor()
        logger.info("SAM3 video predictor loaded ✅")
    except Exception as e:
        logger.error(f"SAM3 video predictor load error: {e}")
        state.load_error += f"SAM3-Video: {e}. "

    # ── Face recognition ──────────────────────────────────────────────────────
    try:
        import face_recognition as _fr
        state.fr_module = _fr
        logger.info("face_recognition loaded ✅")
    except Exception as e:
        logger.warning(f"face_recognition load error: {e}")
        state.load_error += f"FaceRec: {e}. "

    _reload_known_folder()
    state.models_loaded = True
    logger.info("All models loaded.")


def _reload_known_folder(folder="known_persons"):
    if state.fr_module and os.path.isdir(folder):
        state.face_db = build_face_database(folder, state.fr_module)
        logger.info(f"Loaded {len(state.face_db)} known face(s)")


# ═══════════════════════════════════════════════════════════════════════════════
# SAM3 Inference Helpers
# ═══════════════════════════════════════════════════════════════════════════════

# ── Colour palette for labels ─────────────────────────────────────────────────
_label_colors: Dict[str, tuple] = {}
_rng = np.random.RandomState(42)

def _label_color(label: str) -> tuple:
    if label not in _label_colors:
        _label_colors[label] = tuple(_rng.randint(60, 240, 3).tolist())
    return _label_colors[label]


def _overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, color: tuple, alpha=0.40):
    """Blend a binary mask onto the frame with translucent fill + solid contour."""
    overlay = frame_bgr.copy()
    overlay[mask > 0] = (
        overlay[mask > 0] * (1 - alpha) +
        np.array(color[::-1], dtype=np.float32) * alpha
    ).astype(np.uint8)
    frame_bgr[:] = overlay

    # Draw contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_bgr, contours, -1, color[::-1], 2)


def _draw_label_box(frame_bgr, x1, y1, label_text, color):
    """Draw a filled label banner above a detection."""
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 8, y1), color[::-1], -1)
    cv2.putText(frame_bgr, label_text, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)


def process_frame_text_prompt(
    frame_bgr: np.ndarray,
    text_prompts: List[str],
    conf_threshold: float = SAM3_CONF_THRESHOLD,
    frame_num: int = 0,
) -> tuple:
    """
    Run SAM3 image model with a list of text prompts.
    Returns (annotated_frame, list_of_label_strings)
    """
    if state.sam3_processor is None or not text_prompts:
        return frame_bgr, []

    labels_found = []
    h_orig, w_orig = frame_bgr.shape[:2]

    # Scale down for inference speed
    small = cv2.resize(frame_bgr, (int(w_orig * INFER_SCALE), int(h_orig * INFER_SCALE)))
    pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    try:
        inference_state = state.sam3_processor.set_image(pil_img)
        combined_prompt = ", ".join(text_prompts)
        output = state.sam3_processor.set_text_prompt(
            state=inference_state,
            prompt=combined_prompt,
        )
    except Exception as e:
        logger.error(f"SAM3 text inference error: {e}")
        return frame_bgr, []

    masks  = output.get("masks",  [])
    boxes  = output.get("boxes",  [])
    scores = output.get("scores", [])
    prompt_labels = output.get("labels", [combined_prompt] * len(masks))

    scale_x = w_orig / (w_orig * INFER_SCALE)
    scale_y = h_orig / (h_orig * INFER_SCALE)

    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        if float(score) < conf_threshold:
            continue

        # Get label string
        raw_label = prompt_labels[i] if i < len(prompt_labels) else combined_prompt
        # Match to the user-supplied prompt (closest)
        matched_label = _match_prompt_label(raw_label, text_prompts)
        color = _label_color(matched_label)

        # Scale mask back to original resolution
        mask_np = np.array(mask, dtype=np.uint8)
        if mask_np.ndim == 3: mask_np = mask_np[0]
        mask_full = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        _overlay_mask(frame_bgr, mask_full, color)

        # Scale box
        bx1 = int(float(box[0]) * scale_x)
        by1 = int(float(box[1]) * scale_y)
        bx2 = int(float(box[2]) * scale_x)
        by2 = int(float(box[3]) * scale_y)
        cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), color[::-1], 2)

        # Face recognition on person detections
        display_name = matched_label
        if "person" in matched_label.lower() and state.fr_module and state.face_db:
            if frame_num % FACE_REC_EVERY_N == 0:
                display_name = _run_face_rec(frame_bgr, bx1, by1, bx2, by2) or matched_label

        conf_pct = int(float(score) * 100)
        tag = f"{display_name} {conf_pct}%"
        _draw_label_box(frame_bgr, bx1, by1, tag, color)
        labels_found.append(display_name)

    return frame_bgr, labels_found


def process_frame_box_prompt(
    frame_bgr: np.ndarray,
    boxes_with_labels: List[Dict],   # [{"x1","y1","x2","y2","label"}, ...]
    conf_threshold: float = SAM3_CONF_THRESHOLD,
    frame_num: int = 0,
) -> tuple:
    """
    Run SAM3 image model with manual bounding-box exemplar prompts.
    For each drawn box, SAM3 segments all similar objects in the scene.
    Returns (annotated_frame, list_of_label_strings)
    """
    if state.sam3_processor is None or not boxes_with_labels:
        return frame_bgr, []

    labels_found = []
    h_orig, w_orig = frame_bgr.shape[:2]

    small = cv2.resize(frame_bgr, (int(w_orig * INFER_SCALE), int(h_orig * INFER_SCALE)))
    pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    scale_x = w_orig / (w_orig * INFER_SCALE)
    scale_y = h_orig / (h_orig * INFER_SCALE)

    try:
        inference_state = state.sam3_processor.set_image(pil_img)

        for box_entry in boxes_with_labels:
            lbl = box_entry.get("label", "object")
            # Scale box DOWN to inference resolution
            bx1s = int(box_entry["x1"] / scale_x)
            by1s = int(box_entry["y1"] / scale_y)
            bx2s = int(box_entry["x2"] / scale_x)
            by2s = int(box_entry["y2"] / scale_y)

            output = state.sam3_processor.set_box_prompt(
                state=inference_state,
                box=[bx1s, by1s, bx2s, by2s],
            )

            masks  = output.get("masks",  [])
            boxes  = output.get("boxes",  [])
            scores = output.get("scores", [])
            color  = _label_color(lbl)

            for mask, box, score in zip(masks, boxes, scores):
                if float(score) < conf_threshold:
                    continue
                mask_np = np.array(mask, dtype=np.uint8)
                if mask_np.ndim == 3: mask_np = mask_np[0]
                mask_full = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                _overlay_mask(frame_bgr, mask_full, color)

                out_bx1 = int(float(box[0]) * scale_x)
                out_by1 = int(float(box[1]) * scale_y)
                out_bx2 = int(float(box[2]) * scale_x)
                out_by2 = int(float(box[3]) * scale_y)
                cv2.rectangle(frame_bgr, (out_bx1, out_by1), (out_bx2, out_by2), color[::-1], 2)

                display_name = lbl
                if "person" in lbl.lower() and state.fr_module and state.face_db:
                    if frame_num % FACE_REC_EVERY_N == 0:
                        display_name = _run_face_rec(frame_bgr, out_bx1, out_by1, out_bx2, out_by2) or lbl

                conf_pct = int(float(score) * 100)
                _draw_label_box(frame_bgr, out_bx1, out_by1, f"{display_name} {conf_pct}%", color)
                labels_found.append(display_name)

    except Exception as e:
        logger.error(f"SAM3 box inference error: {e}")

    return frame_bgr, labels_found


def _match_prompt_label(raw: str, prompts: List[str]) -> str:
    """Best-effort match output label to one of the user's prompts."""
    raw_l = raw.lower()
    for p in prompts:
        if p.lower() in raw_l or raw_l in p.lower():
            return p
    return prompts[0] if prompts else raw


def _run_face_rec(frame_bgr, x1, y1, x2, y2) -> Optional[str]:
    """Run face recognition on a person crop. Returns name or None."""
    try:
        h, w = frame_bgr.shape[:2]
        pad = 20
        rx1 = max(0, x1-pad); ry1 = max(0, y1-pad)
        rx2 = min(w, x2+pad); ry2 = min(h, y2+pad)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        roi = np.ascontiguousarray(frame_rgb[ry1:ry2, rx1:rx2], dtype=np.uint8)
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            return None
        roi = _preprocess_for_face_rec(roi)
        locs = state.fr_module.face_locations(roi, number_of_times_to_upsample=FR_UPSAMPLE, model=FR_DETECTION_MODEL)
        if not locs: return None
        encs = state.fr_module.face_encodings(roi, known_face_locations=locs, num_jitters=FR_LIVE_JITTERS)
        known_encs  = list(state.face_db.values())
        known_names = list(state.face_db.keys())
        for enc in encs:
            dists = state.fr_module.face_distance(known_encs, enc)
            best_idx  = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            if best_dist > FR_DISTANCE_UNKNOWN: return "Unknown"
            sorted_idx = np.argsort(dists)[:FR_TOP_K_VOTES]
            votes = {}
            for idx in sorted_idx:
                if dists[idx] <= FR_DISTANCE_UNKNOWN:
                    w_ = 1.0 / (dists[idx] + 1e-6)
                    votes[known_names[idx]] = votes.get(known_names[idx], 0) + w_
            if votes: return max(votes, key=votes.__getitem__)
    except Exception as e:
        logger.debug(f"Face rec error: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SAM3 Video Tracker Helpers (for RTSP streaming)
# ═══════════════════════════════════════════════════════════════════════════════
def start_video_session(camera_id: int, text_prompts: List[str], init_frame: np.ndarray) -> Optional[str]:
    """
    Initialise a SAM3 video tracking session on the first RTSP frame.
    Returns session_id or None on failure.
    """
    if state.sam3_video_predictor is None: return None
    try:
        import uuid
        session_id = str(uuid.uuid4())
        pil_frame  = Image.fromarray(cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB))

        response = state.sam3_video_predictor.handle_request(request=dict(
            type="start_session",
            session_id=session_id,
            frame=pil_frame,
        ))

        # Add text concept prompts
        for prompt in text_prompts:
            state.sam3_video_predictor.handle_request(request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt,
            ))

        state.cam_sessions[camera_id] = session_id
        logger.info(f"[Cam {camera_id}] SAM3 video session started: {session_id}")
        return session_id
    except Exception as e:
        logger.error(f"SAM3 session start error: {e}")
        return None


def process_rtsp_frame(
    frame_bgr: np.ndarray,
    camera_id: int,
    text_prompts: List[str],
    box_prompts: List[Dict],
    conf_threshold: float,
    frame_num: int,
) -> tuple:
    """
    Main RTSP frame processing. Tries SAM3 video tracker first,
    falls back to per-frame image model if tracker not available.
    """
    global _frame_counter

    # ── If video predictor is available, use tracking session ─────────────────
    if state.sam3_video_predictor is not None:
        session_id = state.cam_sessions.get(camera_id)

        # Initialise session on first call or if prompts changed
        if session_id is None and (text_prompts or box_prompts):
            session_id = start_video_session(camera_id, text_prompts, frame_bgr)

        if session_id:
            return _track_frame(frame_bgr, camera_id, session_id, text_prompts, box_prompts, conf_threshold, frame_num)

    # ── Fallback: per-frame image model ───────────────────────────────────────
    if text_prompts:
        return process_frame_text_prompt(frame_bgr, text_prompts, conf_threshold, frame_num)
    elif box_prompts:
        return process_frame_box_prompt(frame_bgr, box_prompts, conf_threshold, frame_num)

    return frame_bgr, []


def _track_frame(frame_bgr, camera_id, session_id, text_prompts, box_prompts, conf_threshold, frame_num):
    """Propagate SAM3 tracker for one RTSP frame."""
    labels_found = []
    h_orig, w_orig = frame_bgr.shape[:2]

    try:
        pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        response  = state.sam3_video_predictor.handle_request(request=dict(
            type="propagate",
            session_id=session_id,
            frame=pil_frame,
        ))

        out_masks  = response.get("masks",  [])
        out_boxes  = response.get("boxes",  [])
        out_scores = response.get("scores", [])
        out_ids    = response.get("object_ids", list(range(len(out_masks))))
        out_labels = response.get("labels", text_prompts * len(out_masks) if text_prompts else ["object"] * len(out_masks))

        for mask, box, score, obj_id, lbl in zip(out_masks, out_boxes, out_scores, out_ids, out_labels):
            if float(score) < conf_threshold: continue
            matched_label = _match_prompt_label(str(lbl), text_prompts) if text_prompts else str(lbl)
            color = _label_color(matched_label)

            mask_np = np.array(mask, dtype=np.uint8)
            if mask_np.ndim == 3: mask_np = mask_np[0]
            mask_full = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            _overlay_mask(frame_bgr, mask_full, color)

            bx1 = int(float(box[0])); by1 = int(float(box[1]))
            bx2 = int(float(box[2])); by2 = int(float(box[3]))
            cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), color[::-1], 2)

            display_name = matched_label
            if "person" in matched_label.lower() and state.fr_module and state.face_db:
                if frame_num % FACE_REC_EVERY_N == 0:
                    display_name = _run_face_rec(frame_bgr, bx1, by1, bx2, by2) or matched_label

            conf_pct = int(float(score) * 100)
            _draw_label_box(frame_bgr, bx1, by1, f"{display_name} #{obj_id} {conf_pct}%", color)
            labels_found.append(display_name)

    except Exception as e:
        logger.error(f"SAM3 track frame error (cam {camera_id}): {e}")
        # Session may be stale; clear it so it reinitialises next frame
        state.cam_sessions.pop(camera_id, None)

    return frame_bgr, labels_found


# ═══════════════════════════════════════════════════════════════════════════════
# Lifespan + App
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield
    state.multi_camera.disconnect_all()


app = FastAPI(title="SAM3 Multi-Camera CCTV", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════════════════════
# REST Endpoints
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "cctv_v2.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/status")
async def get_status():
    return {
        "sam3_image":   state.sam3_image_model   is not None,
        "sam3_video":   state.sam3_video_predictor is not None,
        "face_recognition": state.fr_module is not None,
        "known_faces":  list(state.face_db.keys()),
        "models_loaded": state.models_loaded,
        "error": state.load_error or None,
        "cameras": state.multi_camera.get_all_status(),
        "max_cameras": MAX_CAMERAS,
        "label_colors": {k: list(v) for k, v in _label_colors.items()},
    }


# ── Camera endpoints ──────────────────────────────────────────────────────────
class RTSPConnectRequest(BaseModel):
    camera_id: int
    url: str
    label: str = ""


@app.post("/api/cameras/connect")
async def camera_connect(req: RTSPConnectRequest):
    if req.camera_id < 0 or req.camera_id >= MAX_CAMERAS:
        raise HTTPException(400, f"camera_id must be 0–{MAX_CAMERAS-1}")
    if not req.url.strip():
        raise HTTPException(400, "RTSP URL required")
    cam = state.multi_camera.get_camera(req.camera_id)
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, cam.connect, req.url.strip(), req.label)
    if not ok: raise HTTPException(503, cam.error or "Failed to connect")
    return cam.to_dict()


@app.post("/api/cameras/{camera_id}/disconnect")
async def camera_disconnect(camera_id: int):
    cam = state.multi_camera.get_camera(camera_id)
    if not cam: raise HTTPException(404, "Camera not found")
    cam.disconnect()
    state.cam_sessions.pop(camera_id, None)
    return {"camera_id": camera_id, "connected": False}


@app.get("/api/cameras")
async def get_cameras():
    return {"cameras": state.multi_camera.get_all_status()}


@app.get("/api/cameras/{camera_id}")
async def get_camera_status(camera_id: int):
    cam = state.multi_camera.get_camera(camera_id)
    if not cam: raise HTTPException(404, "Camera not found")
    return cam.to_dict()


@app.post("/api/cameras/{camera_id}/reset_session")
async def reset_camera_session(camera_id: int):
    """Force a new SAM3 tracking session for this camera (e.g. after prompt change)."""
    state.cam_sessions.pop(camera_id, None)
    return {"camera_id": camera_id, "session_reset": True}


# ── Image / Video detection ───────────────────────────────────────────────────
@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.30),
    prompts: str = Form(""),        # comma-separated text prompts
    mode: str = Form("text"),       # "text" | "box"
    boxes: str = Form("[]"),        # JSON array for box mode
):
    import json
    text_prompts = [p.strip() for p in prompts.split(",") if p.strip()] if prompts else []
    box_prompts  = json.loads(boxes) if boxes else []

    contents = await file.read()
    img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
    img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    if mode == "box" and box_prompts:
        annotated, labels = process_frame_box_prompt(img_bgr, box_prompts, conf, 0)
    else:
        annotated, labels = process_frame_text_prompt(img_bgr, text_prompts, conf, 0)

    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode()
    return {"image": f"data:image/jpeg;base64,{b64}", "labels": labels,
            "unique": list(set(labels)), "prompts": text_prompts}


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.30),
    prompts: str = Form(""),
):
    text_prompts = [p.strip() for p in prompts.split(",") if p.strip()] if prompts else []
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read()); in_path = tmp.name
    out_path = in_path.replace(suffix, "_sam3.mp4")
    cap = cv2.VideoCapture(in_path)
    fps_ = cap.get(cv2.CAP_PROP_FPS) or 25
    w_   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out  = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_, (w_, h_))
    all_labels = set(); fc = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        ann, lbls = process_frame_text_prompt(frame, text_prompts, conf, fc)
        all_labels.update(lbls); out.write(ann); fc += 1
    cap.release(); out.release(); os.unlink(in_path)
    return FileResponse(out_path, media_type="video/mp4", filename="sam3_annotated.mp4",
                        headers={"X-Detections": ",".join(sorted(all_labels))})


# ── Face management ───────────────────────────────────────────────────────────
@app.post("/api/faces/upload")
async def upload_faces(files: list[UploadFile] = File(...)):
    if not state.fr_module: raise HTTPException(503, "face_recognition not available")
    tmp_dir = tempfile.mkdtemp(); saved = []
    try:
        for uf in files:
            fp = os.path.join(tmp_dir, uf.filename)
            with open(fp, "wb") as f: f.write(await uf.read())
            saved.append(fp)
        extra = build_face_database(tmp_dir, state.fr_module)
        state.face_db.update(extra)
        return {"enrolled": list(extra.keys()), "total_known": list(state.face_db.keys()),
                "failed": len(saved) - len(extra)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    if name not in state.face_db: raise HTTPException(404, f"'{name}' not found")
    del state.face_db[name]
    return {"deleted": name, "remaining": list(state.face_db.keys())}


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket — Browser Webcam
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
            text_prompts = data.get("prompts", [])
            box_prompts  = data.get("boxes",   [])
            conf         = data.get("conf", SAM3_CONF_THRESHOLD)
            mode         = data.get("mode", "text")   # "text" | "box"

            if "," in frame_b64: frame_b64 = frame_b64.split(",")[1]
            img_data  = base64.b64decode(frame_b64)
            frame_bgr = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                await websocket.send_json({"error": "invalid frame"}); continue

            _frame_counter += 1
            if mode == "box" and box_prompts:
                annotated, labels = await loop.run_in_executor(
                    _executor, process_frame_box_prompt, frame_bgr, box_prompts, conf, _frame_counter)
            else:
                annotated, labels = await loop.run_in_executor(
                    _executor, process_frame_text_prompt, frame_bgr, text_prompts, conf, _frame_counter)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            b64 = base64.b64encode(buf).decode()
            await websocket.send_json({
                "image": f"data:image/jpeg;base64,{b64}",
                "labels": labels, "unique": list(set(labels))
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"Webcam WS error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket — RTSP Camera Streaming
# ═══════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws/rtsp/{camera_id}")
async def websocket_rtsp(websocket: WebSocket, camera_id: int):
    global _frame_counter
    await websocket.accept()
    logger.info(f"WS RTSP client connected — cam {camera_id}")

    cam = state.multi_camera.get_camera(camera_id)
    if not cam:
        await websocket.send_json({"error": f"Invalid camera_id: {camera_id}"}); await websocket.close(); return

    text_prompts: List[str] = []
    box_prompts:  List[Dict] = []
    conf        = SAM3_CONF_THRESHOLD
    interval_ms = 200
    mode        = "text"

    try:
        loop = asyncio.get_event_loop()

        # Read initial config (2 s timeout)
        try:
            init = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
            text_prompts = init.get("prompts",   [])
            box_prompts  = init.get("boxes",     [])
            conf         = init.get("conf",      SAM3_CONF_THRESHOLD)
            interval_ms  = init.get("interval_ms", 200)
            mode         = init.get("mode",      "text")
        except asyncio.TimeoutError:
            pass

        if not cam.is_connected:
            await websocket.send_json({"error": f"Camera {camera_id} not connected."})
            await websocket.close(); return

        last_send = 0.0
        frame_times = []

        while True:
            # Non-blocking config update
            try:
                update = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if "prompts"      in update:
                    text_prompts = update["prompts"]
                    state.cam_sessions.pop(camera_id, None)  # reset tracker on prompt change
                if "boxes"        in update:
                    box_prompts  = update["boxes"]
                    state.cam_sessions.pop(camera_id, None)
                if "conf"         in update: conf        = update["conf"]
                if "interval_ms"  in update: interval_ms = update["interval_ms"]
                if "mode"         in update: mode        = update["mode"]
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            now = time.time()
            if (now - last_send) * 1000 < interval_ms:
                await asyncio.sleep(0.005); continue

            frame = cam.get_frame()
            if frame is None:
                if not cam.is_connected:
                    await websocket.send_json({"error": f"Camera {camera_id} disconnected"}); break
                await asyncio.sleep(0.02); continue

            _frame_counter += 1
            t0 = time.time()
            annotated, labels = await loop.run_in_executor(
                _executor, process_rtsp_frame,
                frame, camera_id, text_prompts, box_prompts, conf, _frame_counter
            )
            elapsed = time.time() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 30: frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            b64 = base64.b64encode(buf).decode()
            try:
                await websocket.send_json({
                    "image": f"data:image/jpeg;base64,{b64}",
                    "labels": labels, "unique": list(set(labels)),
                    "processing_fps": round(avg_fps, 1),
                    "stream_fps":     round(cam.fps, 1),
                    "camera_id":      camera_id,
                })
            except Exception:
                break
            last_send = time.time()

    except WebSocketDisconnect:
        logger.info(f"RTSP WS cam {camera_id} disconnected")
    except Exception as e:
        logger.exception(f"RTSP WS cam {camera_id} error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cctv_v2:app", host="0.0.0.0", port=8001, reload=False, ws="wsproto")