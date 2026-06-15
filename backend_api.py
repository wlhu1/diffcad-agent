# -*- coding: utf-8 -*-
"""
DIffCAD Agent Backend API
Flask-based REST API wrapping the stomata detection logic from app.py.

Supports:
  - Single / batch / video / camera frame detection
  - CSV export of phenotype metrics
  - Agent chat (mock - ready for Coze / LLM API integration)

Model loading: auto-detects weights/ directory, lazy-loads YOLO-OBB models
based on plant_type + sample_type, caches loaded models.  Falls back to mock
ONLY when a specific weight file is genuinely missing, and reports it clearly.
"""

import os
import io
import csv
import json
import sys
import time
import gc
import base64
import threading
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix Windows console encoding for UTF-8 output
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Aggressive memory optimization for cloud deployment (512MB constraints)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# optional heavy imports - graceful fallback
# ---------------------------------------------------------------------------
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    HAS_TORCH = True
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'
except ImportError:
    torch = None
    HAS_TORCH = False
    DEVICE = 'cpu'

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    YOLO = None
    HAS_YOLO = False

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'uploads'
RESULT_DIR = BASE_DIR / 'results'
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

thread_pool = ThreadPoolExecutor(max_workers=4)
sessions: dict = {}

# ---------------------------------------------------------------------------
# detection parameters (mirrors app.py defaults)
# ---------------------------------------------------------------------------
DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.7
DEFAULT_SCALE = 100.0   # um
DEFAULT_STEP = 5         # DiffBIR restoration steps (reserved)

# ---------------------------------------------------------------------------
# Model Manager - lazy loading, caching, auto-selection
# ---------------------------------------------------------------------------
WEIGHT_DIR = BASE_DIR / 'weights'

# Detection weights - mirrors app.py weight selection logic exactly
#   plant_type:  dicotyledons | monocotyledons
#   sample_type: destructive | nondestructive
# Prefers ONNX for lightweight inference; falls back to PyTorch .pt files.
DETECTION_WEIGHTS = {
    ('dicotyledons', 'destructive'):     'dicotyledons_destructive.onnx',
    ('dicotyledons', 'nondestructive'):  'dicotyledons_nondestructive.onnx',
    ('monocotyledons', 'destructive'):   'monocotyledons_destructive.onnx',
    ('monocotyledons', 'nondestructive'): 'monocotyledons_nondestructive.onnx',
}
DETECTION_WEIGHTS_PT = {
    ('dicotyledons', 'destructive'):     'dicotyledons_destructive.pt',
    ('dicotyledons', 'nondestructive'):  'dicotyledons_nondestructive.pt',
    ('monocotyledons', 'destructive'):   'monocotyledons_destructive.pt',
    ('monocotyledons', 'nondestructive'): 'monocotyledons_nondestructive.pt',
}

# Restoration / enhancement weights (reserved interface, not loaded by default)
RESTORATION_WEIGHTS = [
    'face_swinir_v1.ckpt',
    'scunet_color_real_psnr.pth',
    'v2.pth',
    'v2-1_512-ema-pruned.ckpt',
]


class ModelManager:
    """
    Thread-safe lazy-loading model cache.

    Usage:
        mgr = ModelManager()
        model, weight_path, is_mock = mgr.get('dicotyledons', 'nondestructive')
    """

    def __init__(self):
        self._cache: dict = {}       # key → YOLO instance
        self._lock = threading.Lock()
        self._availability = None    # cached scan result
        self._load_errors: dict = {}  # key → error message for models that failed to load

    # ---- weight discovery ----

    def scan_weights(self) -> dict:
        """Scan weights/ directory. Returns detailed availability info."""
        if self._availability is not None:
            return self._availability

        info = {
            'weight_dir_exists': WEIGHT_DIR.is_dir(),
            'detection': {},
            'restoration': {},
            'has_torch': HAS_TORCH,
            'has_yolo': HAS_YOLO,
        }

        if not WEIGHT_DIR.is_dir():
            info['error'] = f'Weight directory not found: {WEIGHT_DIR}'
            self._availability = info
            return info

        # scan detection weights (ONNX preferred, .pt as fallback)
        for (plant, sample), fname in DETECTION_WEIGHTS.items():
            path = WEIGHT_DIR / fname
            pt_fname = DETECTION_WEIGHTS_PT.get((plant, sample), '')
            pt_path = WEIGHT_DIR / pt_fname if pt_fname else None
            key = f'{plant}_{sample}'
            load_err = self._load_errors.get(key)
            has_onnx = path.exists()
            has_pt = pt_path and pt_path.exists()
            info['detection'][key] = {
                'file_name': fname,
                'exists': has_onnx or has_pt,
                'is_onnx': has_onnx,
                'is_pt': has_pt,
                'size_mb': round(path.stat().st_size / (1024 * 1024), 2) if has_onnx else (round(pt_path.stat().st_size / (1024 * 1024), 2) if has_pt else None),
                'plant_type': plant,
                'sample_type': sample,
                'cached': key in self._cache,
                'load_failed': bool(load_err),
                'load_error': load_err or None,
            }

        # scan restoration weights
        for fname in RESTORATION_WEIGHTS:
            path = WEIGHT_DIR / fname
            info['restoration'][fname] = {
                'exists': path.exists(),
                'size_mb': round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None,
            }

        self._availability = info
        return info

    @property
    def can_load_any(self) -> bool:
        """True if at least one detection weight is available."""
        info = self.scan_weights()
        return any(w['exists'] for w in info['detection'].values())

    def missing_weights(self) -> list:
        """List of (plant_type, sample_type) tuples where weights are missing."""
        info = self.scan_weights()
        return [
            (w['plant_type'], w['sample_type'])
            for w in info['detection'].values()
            if not w['exists']
        ]

    def available_combos(self) -> list:
        """List of (plant_type, sample_type) tuples where weights ARE available."""
        info = self.scan_weights()
        return [
            (w['plant_type'], w['sample_type'])
            for w in info['detection'].values()
            if w['exists']
        ]

    # ---- model loading ----

    def get(self, plant_type: str = 'dicotyledons',
            sample_type: str = 'nondestructive') -> tuple:
        """
        Lazy-load and cache a YOLO-OBB model.

        Returns (model_or_None, weight_path_or_None, is_mock: bool, error: str|None)
        """
        if not HAS_YOLO or not HAS_TORCH:
            return None, None, True, 'PyTorch or ultralytics not installed'

        key = f'{plant_type}_{sample_type}'

        # fast path: already cached
        with self._lock:
            if key in self._cache:
                model, wpath = self._cache[key]
                return model, wpath, False, None

        # resolve weight path — for YOLO direct loading, use .pt files
        # (ONNX path is handled by _detect_via_subprocess when USE_SUBPROCESS=1)
        weight_name = DETECTION_WEIGHTS_PT.get((plant_type, sample_type))
        if weight_name is None:
            return None, None, True, (
                f'Unknown plant_type="{plant_type}" or sample_type="{sample_type}". '
                f'Valid: {list(DETECTION_WEIGHTS_PT.keys())}'
            )

        weight_path = WEIGHT_DIR / weight_name

        if not weight_path.exists():
            return None, None, True, (
                f'Weight file not found: {weight_path.name}. '
                f'Please place it in {WEIGHT_DIR}.'
            )

        # load under lock to avoid duplicate loads
        with self._lock:
            # double-check cache
            if key in self._cache:
                model, wpath = self._cache[key]
                return model, wpath, False, None

            try:
                # Disable gradient computation globally for inference-only workloads
                if HAS_TORCH:
                    torch.set_grad_enabled(False)
                gc.collect()

                # Monkey-patch missing custom loss classes from older YOLO training
                if HAS_TORCH:
                    import torch.nn as nn
                    import ultralytics.utils.loss as loss_mod
                    if not hasattr(loss_mod, 'SoftmaxEQLV2Loss'):
                        class SoftmaxEQLV2Loss(nn.Module):
                            """Stub for models trained with custom SoftmaxEQLV2Loss."""
                            def __init__(self, *args, **kwargs):
                                super().__init__()
                            def forward(self, *args, **kwargs):
                                return torch.tensor(0.0)
                        loss_mod.SoftmaxEQLV2Loss = SoftmaxEQLV2Loss
                        print('[model] Patched: SoftmaxEQLV2Loss stub registered in ultralytics.utils.loss')

                print(f'[model] Loading {weight_path.name} ...')
                model = YOLO(str(weight_path), task='obb')
                # warm-up forward pass (mirrors app.py line 95)
                model(np.zeros((48, 48, 3), dtype=np.uint8), device=DEVICE)
                self._cache[key] = (model, str(weight_path))
                gc.collect()  # free temporary memory after model load
                print(f'[model] Loaded: {weight_path.name}  (cache size: {len(self._cache)})')
                return model, str(weight_path), False, None
            except Exception as exc:
                msg = f'Failed to load {weight_path.name}: {str(exc)}'
                print(f'[model] FAILED: {msg}')
                self._load_errors[key] = msg
                return None, None, True, msg

    def preload_all(self):
        """Preload all available detection weights. Call after startup if desired."""
        for plant_type, sample_type in self.available_combos():
            self.get(plant_type, sample_type)

    def get_restoration_info(self) -> dict:
        """Return info about available restoration weights (reserved interface)."""
        info = self.scan_weights()
        return {
            'available': [k for k, v in info['restoration'].items() if v['exists']],
            'missing': [k for k, v in info['restoration'].items() if not v['exists']],
            'note': 'Restoration models are reserved for future use. Not loaded by default.',
        }


# ---- singleton ----
model_manager = ModelManager()

# ---------------------------------------------------------------------------
# mock detection  (only used when a specific weight is genuinely missing)
# ---------------------------------------------------------------------------
def mock_detect(image: np.ndarray, conf: float = 0.5) -> dict:
    """Generate synthetic OBB-style results matching real detection output shape."""
    seed = hash(image.tobytes()[:1024]) % (2**31)
    rng = np.random.default_rng(abs(seed))

    h, w = image.shape[:2]
    n_stomata = rng.integers(20, 55)
    n_aperture = rng.integers(8, max(9, n_stomata))

    boxes = []
    for _ in range(n_stomata):
        cx = rng.uniform(0.15 * w, 0.85 * w)
        cy = rng.uniform(0.15 * h, 0.85 * h)
        bw = rng.uniform(0.03 * w, 0.08 * w)
        bh = rng.uniform(0.015 * h, 0.04 * h)
        angle = rng.uniform(0, np.pi)
        boxes.append([cx, cy, bw, bh, angle, 0])

    for _ in range(n_aperture):
        cx = rng.uniform(0.15 * w, 0.85 * w)
        cy = rng.uniform(0.15 * h, 0.85 * h)
        bw = rng.uniform(0.015 * w, 0.05 * w)
        bh = rng.uniform(0.005 * h, 0.02 * h)
        angle = rng.uniform(0, np.pi)
        boxes.append([cx, cy, bw, bh, angle, 1])

    boxes = np.array(boxes) if boxes else np.empty((0, 6))
    return {'boxes': boxes, 'orig_shape': (h, w), 'is_mock': True}


# ---------------------------------------------------------------------------
# phenotype calculation  (identical math to app.py lines 340-428)
# ---------------------------------------------------------------------------
def calc_phenotype(boxes: np.ndarray, orig_shape: tuple, scale_um: float) -> dict:
    """Calculate phenotype metrics from OBB detection results."""
    if len(boxes) == 0:
        return {
            'stoma_count': 0, 'aperture_count': 0,
            'stoma_avg_width_um': None, 'stoma_avg_height_um': None,
            'stoma_aspect_ratio': None,
            'aperture_avg_width_um': None, 'aperture_avg_height_um': None,
            'aperture_aspect_ratio': None,
            'stoma_density_mm2': None, 'conductance': None,
            'image_area_mm2': None,
        }

    cls_ids = boxes[:, 5].astype(int)
    stoma_idx = cls_ids == 0
    aperture_idx = cls_ids == 1

    stoma_w = boxes[stoma_idx, 2] if stoma_idx.any() else np.array([])
    stoma_h = boxes[stoma_idx, 3] if stoma_idx.any() else np.array([])
    aperture_w = boxes[aperture_idx, 2] if aperture_idx.any() else np.array([])
    aperture_h = boxes[aperture_idx, 3] if aperture_idx.any() else np.array([])

    ref_px = 224.0
    scale_factor = scale_um / ref_px

    if len(stoma_w) > 0:
        stoma_avg_w = float(np.mean(stoma_w)) * scale_factor
        stoma_avg_h = float(np.mean(stoma_h)) * scale_factor
        stoma_ar = float(np.mean(stoma_h / (stoma_w + 1e-8)))
    else:
        stoma_avg_w = stoma_avg_h = stoma_ar = None

    if len(aperture_w) > 0:
        aperture_avg_w = float(np.mean(aperture_w)) * scale_factor
        aperture_avg_h = float(np.mean(aperture_h)) * scale_factor
        aperture_ar = float(np.mean(aperture_h / (aperture_w + 1e-8)))
    else:
        aperture_avg_w = aperture_avg_h = aperture_ar = None

    n_stoma = int(stoma_idx.sum())
    n_aperture = int(aperture_idx.sum())
    img_h, img_w = orig_shape

    image_area_mm2 = (img_h / ref_px) * (img_w / ref_px) * (scale_um / 1000) ** 2
    density = (n_stoma / image_area_mm2) if (image_area_mm2 > 0 and n_stoma > 0) else None

    if aperture_avg_h is not None and density is not None and aperture_avg_w is not None:
        alpha_mean = (aperture_avg_h / 2) ** 2 * np.pi
        cond = (24.9e-6 * density * alpha_mean /
                (1.6 * 22.4e-3 * (aperture_avg_w + np.sqrt(alpha_mean * np.pi / 4))))
    else:
        cond = None

    return {
        'stoma_count': n_stoma, 'aperture_count': n_aperture,
        'stoma_avg_width_um': round(stoma_avg_w, 4) if stoma_avg_w is not None else None,
        'stoma_avg_height_um': round(stoma_avg_h, 4) if stoma_avg_h is not None else None,
        'stoma_aspect_ratio': round(stoma_ar, 4) if stoma_ar is not None else None,
        'aperture_avg_width_um': round(aperture_avg_w, 4) if aperture_avg_w is not None else None,
        'aperture_avg_height_um': round(aperture_avg_h, 4) if aperture_avg_h is not None else None,
        'aperture_aspect_ratio': round(aperture_ar, 4) if aperture_ar is not None else None,
        'stoma_density_mm2': round(density, 4) if density is not None else None,
        'conductance': round(cond, 4) if cond is not None else None,
        'image_area_mm2': round(image_area_mm2, 4),
    }


# ---------------------------------------------------------------------------
# core detection pipeline  (mirrors app.py process_single_image)
# ---------------------------------------------------------------------------
def _detect_via_subprocess(image, conf, iou, scale_um, plant_type, sample_type):
    """Run detection in a subprocess.  PyTorch/ONNX memory is freed when it exits."""
    import subprocess
    import tempfile

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir=str(BASE_DIR)) as tmp:
        tmp_path = Path(tmp.name)
        cv2.imwrite(str(tmp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Prefer ONNX worker if model and onnxruntime are available; fall back to PyTorch
    worker = BASE_DIR / 'onnx_worker.py'
    onnx_name = DETECTION_WEIGHTS.get((plant_type, sample_type), '')
    onnx_model = BASE_DIR / 'weights' / onnx_name if onnx_name else None
    if not (worker.exists() and onnx_model and onnx_model.exists()):
        worker = BASE_DIR / 'detection_worker.py'
    try:
        proc = subprocess.run(
            [sys.executable, str(worker), str(tmp_path),
             plant_type, sample_type, str(conf), str(iou), str(scale_um)],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or f'exit code {proc.returncode}')

        result = json.loads(proc.stdout.strip())
        if not result.get('success') and result.get('error'):
            raise RuntimeError(result['error'])

        # Map worker output keys to the expected format
        return {
            'boxes': result.get('boxes', []),
            'metrics': result.get('metrics', {}),
            'overlay_b64': result.get('overlay_b64', ''),
            'input_b64': result.get('input_b64', ''),
            'is_mock': result.get('is_mock', True),
            'mock_reason': result.get('mock_reason'),
            'weight_path': result.get('weight_path', ''),
            'plant_type': result.get('plant_type', plant_type),
            'sample_type': result.get('sample_type', sample_type),
        }
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def detect_image(image: np.ndarray, conf: float, iou: float,
                 scale_um: float = DEFAULT_SCALE,
                 plant_type: str = 'dicotyledons',
                 sample_type: str = 'nondestructive') -> dict:
    """
    Run stomata detection on a single image.

    In subprocess mode (USE_SUBPROCESS=1), spawns a worker to load PyTorch+YOLO,
    run inference, and exit — freeing all ML memory between requests.
    This allows YOLO-OBB to run on 512 MB cloud free tiers.
    """
    if os.environ.get('USE_SUBPROCESS', '').lower() == '1':
        return _detect_via_subprocess(image, conf, iou, scale_um, plant_type, sample_type)

    model, weight_path, is_mock, error_msg = model_manager.get(plant_type, sample_type)

    if model is not None:
        # ---- REAL detection ----
        results = model(image, conf=conf, iou=iou)[0]
        if results.obb is not None and len(results.obb.xywhr) > 0:
            boxes_np = results.obb.xywhr.cpu().numpy()
            cls_np = results.obb.cls.cpu().numpy().reshape(-1, 1)
            boxes = np.hstack([boxes_np, cls_np])
        else:
            boxes = np.empty((0, 6))
        is_mock = False
        mock_reason = None
    else:
        # ---- Mock fallback (weight file missing) ----
        result = mock_detect(image, conf)
        boxes = result['boxes']
        is_mock = True
        mock_reason = error_msg

    orig_shape = image.shape[:2]
    metrics = calc_phenotype(boxes, orig_shape, scale_um)
    overlay = render_overlay(image, boxes, is_mock, mock_reason)

    return {
        'boxes': boxes.tolist(),
        'metrics': metrics,
        'overlay_b64': image_to_base64(overlay),
        'input_b64': image_to_base64(image),
        'is_mock': is_mock,
        'mock_reason': mock_reason,
        'weight_path': weight_path,
        'plant_type': plant_type,
        'sample_type': sample_type,
    }


def render_overlay(image: np.ndarray, boxes: np.ndarray,
                   is_mock: bool, mock_reason: str = None) -> np.ndarray:
    """Draw oriented bounding boxes. Stoma=cyan, aperture=green."""
    overlay = image.copy()
    if len(boxes) == 0:
        if is_mock:
            label = f'MOCK - {mock_reason or "no weights"}'[:60]
            cv2.putText(overlay, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        return overlay

    for box in boxes:
        cx, cy, w, h, angle, cls_id = box
        color = (255, 255, 0) if int(cls_id) == 0 else (0, 255, 105)
        rect = ((cx, cy), (w, h), np.degrees(angle))
        pts = cv2.boxPoints(rect)
        pts = np.int32(pts)
        cv2.drawContours(overlay, [pts], 0, color, 2)

    if is_mock:
        label = f'MOCK - {mock_reason or "no weights"}'[:60]
        cv2.putText(overlay, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    return overlay


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def image_to_base64(img: np.ndarray) -> str:
    """Encode an OpenCV BGR image to JPEG base64 for frontend display.

    IMPORTANT: cv2.imencode expects BGR input and handles BGR→YUV→JPEG
    internally.  Do NOT convert to RGB before encoding — that would swap the
    R and B channels in the output JPEG, causing a visible colour shift
    between the Input and Detection Result canvases.
    """
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode('utf-8')


def read_image_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Cannot decode image')
    return img


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {'results': [], 'created_at': time.time()}
    return sessions[session_id]


def _parse_detect_params(req_form, req_json=None) -> dict:
    """Extract common detection parameters from request."""
    if req_json:
        return {
            'conf': float(req_json.get('conf', DEFAULT_CONF)),
            'iou': float(req_json.get('iou', DEFAULT_IOU)),
            'scale': float(req_json.get('scale', DEFAULT_SCALE)),
            'plant_type': req_json.get('plant_type', 'dicotyledons'),
            'sample_type': req_json.get('sample_type', 'nondestructive'),
            'session_id': req_json.get('session_id', 'default'),
        }
    return {
        'conf': float(req_form.get('conf', DEFAULT_CONF)),
        'iou': float(req_form.get('iou', DEFAULT_IOU)),
        'scale': float(req_form.get('scale', DEFAULT_SCALE)),
        'plant_type': req_form.get('plant_type', 'dicotyledons'),
        'sample_type': req_form.get('sample_type', 'nondestructive'),
        'session_id': req_form.get('session_id', 'default'),
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route('/api/status', methods=['GET'])
def api_status():
    """Return backend status, model availability, and weight inventory."""
    scan = model_manager.scan_weights()
    available = model_manager.available_combos()
    missing = model_manager.missing_weights()
    resto = model_manager.get_restoration_info()

    load_failed_combos = [
        f'{p}_{s}' for p, s in available
        if scan.get('detection', {}).get(f'{p}_{s}', {}).get('load_failed')
    ]

    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'has_torch': HAS_TORCH,
        'has_yolo': HAS_YOLO,
        'weight_dir_exists': scan['weight_dir_exists'],
        'detection_weights_available': [
            f'{p}_{s}' for p, s in available
        ],
        'detection_weights_missing': [
            f'{p}_{s}' for p, s in missing
        ],
        'detection_weights_load_failed': load_failed_combos,
        'detection_weight_details': scan.get('detection', {}),
        'restoration_weights': resto,
        'cached_models': list(model_manager._cache.keys()),
        'version': '2.1.0',
    })


@app.route('/api/restoration/status', methods=['GET'])
def api_restoration_status():
    """Restoration/enhancement weight status (DiffBIR pipeline)."""
    resto = model_manager.get_restoration_info()
    return jsonify({
        'success': True,
        'restoration_weights_available': resto['available'],
        'restoration_weights_missing': resto['missing'],
        'note': 'Restoration requires DiffBIR framework. These weights are used by the desktop application for blind image restoration.',
        'weights_dir': str(WEIGHT_DIR),
    })


@app.route('/api/restoration/enhance', methods=['POST'])
def api_restoration_enhance():
    """Enhance/restore image quality using DiffBIR model (requires DiffBIR + PyTorch)."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Cannot decode image'}), 400

    try:
        # Try importing DiffBIR (only available in desktop environment)
        import importlib
        spec = importlib.util.find_spec('DiffBIR')
        if spec is None:
            return jsonify({
                'success': True,
                'restored_b64': image_to_base64(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                'restoration_applied': False,
                'note': 'DiffBIR not installed — returning original image. Install DiffBIR + basicsr for full restoration.',
            })

        # Full restoration pipeline (requires GPU and large memory)
        from DiffBIR import diff_bir
        ir_model = diff_bir(device='cpu', steps=50)
        restored = ir_model.restore(image)
        _, buf = cv2.imencode('.jpg', restored)
        import base64
        return jsonify({
            'success': True,
            'restored_b64': base64.b64encode(buf).decode(),
            'restoration_applied': True,
            'note': 'Image restored with DiffBIR v2.pth model.',
        })
    except Exception as e:
        return jsonify({
            'success': True,
            'restored_b64': image_to_base64(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            'restoration_applied': False,
            'note': f'Restoration unavailable: {str(e)}',
        })


@app.route('/api/detect/single', methods=['POST'])
def api_detect_single():
    """
    Detect stomata in a single uploaded image.

    Form fields:
      - image:       file upload (required)
      - conf:        float (default 0.5)
      - iou:         float (default 0.7)
      - scale:       float (default 100.0, um)
      - plant_type:  'dicotyledons' | 'monocotyledons' (default dicotyledons)
      - sample_type: 'destructive' | 'nondestructive'  (default nondestructive)
      - session_id:  str
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    params = _parse_detect_params(request.form)

    try:
        img = read_image_from_bytes(file.read())
    except Exception as e:
        return jsonify({'error': f'Failed to read image: {str(e)}'}), 400

    result = detect_image(img, params['conf'], params['iou'], params['scale'],
                          params['plant_type'], params['sample_type'])
    result['file_name'] = file.filename
    result['source'] = 'single'

    sess = get_session(params['session_id'])
    sess['results'].append(result)

    return jsonify({
        'success': True,
        'file_name': file.filename,
        'metrics': result['metrics'],
        'overlay_b64': result['overlay_b64'],
        'input_b64': result['input_b64'],
        'boxes': result['boxes'],
        'is_mock': result['is_mock'],
        'mock_reason': result.get('mock_reason'),
        'weight_path': result.get('weight_path'),
        'plant_type': params['plant_type'],
        'sample_type': params['sample_type'],
        'session_id': params['session_id'],
    })


@app.route('/api/detect/batch', methods=['POST'])
def api_detect_batch():
    """
    Detect stomata in multiple uploaded images.

    Form fields same as /api/detect/single, but with 'images' (multiple files).
    """
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400

    files = request.files.getlist('images')
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in files if Path(f.filename).suffix.lower() in img_extensions]
    if not files:
        return jsonify({'error': 'No valid image files found'}), 400

    params = _parse_detect_params(request.form)

    # Read all images first
    images_data = []
    for f in files:
        try:
            img = read_image_from_bytes(f.read())
            images_data.append((f.filename, img))
        except Exception:
            pass

    sess = get_session(params['session_id'])
    results = []
    total = len(images_data)

    futures = {}
    for fname, img in images_data:
        fut = thread_pool.submit(
            detect_image, img, params['conf'], params['iou'], params['scale'],
            params['plant_type'], params['sample_type']
        )
        futures[fut] = fname

    for i, fut in enumerate(as_completed(futures)):
        fname = futures[fut]
        try:
            r = fut.result()
            r['file_name'] = fname
            r['source'] = 'batch'
            r['index'] = i
            sess['results'].append(r)
            results.append({
                'file_name': fname,
                'success': True,
                'metrics': r['metrics'],
                'overlay_b64': r['overlay_b64'],
                'input_b64': r['input_b64'],
                'is_mock': r['is_mock'],
                'mock_reason': r.get('mock_reason'),
                'index': i,
            })
        except Exception as e:
            results.append({'file_name': fname, 'error': str(e), 'success': False})

    results.sort(key=lambda x: x.get('index', 0))

    return jsonify({
        'success': True,
        'total': total,
        'results': results,
        'plant_type': params['plant_type'],
        'sample_type': params['sample_type'],
        'session_id': params['session_id'],
    })


@app.route('/api/detect/video', methods=['POST'])
def api_detect_video():
    """
    Process uploaded video: extract and detect frames at interval.

    Form fields same as single + frame_interval (default 30).
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    params = _parse_detect_params(request.form)
    frame_interval = int(request.form.get('frame_interval', 30))

    video_path = UPLOAD_DIR / f'{uuid.uuid4().hex}_{file.filename}'
    file.save(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    sess = get_session(params['session_id'])
    frame_results = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            r = detect_image(frame, params['conf'], params['iou'], params['scale'],
                             params['plant_type'], params['sample_type'])
            r['file_name'] = f'{file.filename}_frame_{frame_idx:04d}'
            r['source'] = 'video'
            r['frame_index'] = frame_idx
            r['timestamp_sec'] = frame_idx / fps
            sess['results'].append(r)
            frame_results.append({
                'frame_index': frame_idx,
                'timestamp_sec': round(frame_idx / fps, 2),
                'metrics': r['metrics'],
                'overlay_b64': r['overlay_b64'],
                'input_b64': r.get('input_b64', ''),
                'file_name': f'{file.filename}_frame_{frame_idx:04d}',
                'is_mock': r['is_mock'],
                'mock_reason': r.get('mock_reason'),
            })
            processed += 1

        frame_idx += 1

    cap.release()
    try:
        video_path.unlink()
    except Exception:
        pass

    return jsonify({
        'success': True,
        'file_name': file.filename,
        'total_frames': total_frames,
        'detected_frames': processed,
        'fps': fps,
        'frame_interval': frame_interval,
        'results': frame_results,
        'plant_type': params['plant_type'],
        'sample_type': params['sample_type'],
        'session_id': params['session_id'],
    })


@app.route('/api/detect/frame', methods=['POST'])
def api_detect_frame():
    """
    Detect stomata on a base64-encoded frame (camera / live capture).

    JSON body:
      - image_b64:  base64 JPEG (required)
      - conf, iou, scale, plant_type, sample_type, session_id (optional)
    """
    data = request.get_json(silent=True) or {}
    image_b64 = data.get('image_b64', '')
    if not image_b64:
        return jsonify({'error': 'No image_b64 provided'}), 400

    params = _parse_detect_params(None, data)

    try:
        img_bytes = base64.b64decode(image_b64)
        img = read_image_from_bytes(img_bytes)
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400

    result = detect_image(img, params['conf'], params['iou'], params['scale'],
                          params['plant_type'], params['sample_type'])
    result['file_name'] = 'camera_frame'
    result['source'] = 'camera'

    sess = get_session(params['session_id'])
    sess['results'].append(result)

    return jsonify({
        'success': True,
        'metrics': result['metrics'],
        'overlay_b64': result['overlay_b64'],
        'is_mock': result['is_mock'],
        'mock_reason': result.get('mock_reason'),
        'weight_path': result.get('weight_path'),
        'plant_type': params['plant_type'],
        'sample_type': params['sample_type'],
        'session_id': params['session_id'],
    })


@app.route('/api/export/csv', methods=['POST'])
def api_export_csv():
    """Export session results as CSV. Matches app.py save_detect_result format."""
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id', 'default')

    if session_id not in sessions or not sessions[session_id]['results']:
        return jsonify({'error': 'No results to export'}), 404

    results = sessions[session_id]['results']
    header = [
        'file name',
        'stomata average height (um)', 'stomata average width (um)',
        'stomata aspect ratio',
        'aperture average height (um)', 'aperture average width (um)',
        'aperture aspect ratio',
        'stomata density (stomata * mm-2)', 'conductance (mol m-2 s-1)',
        'stoma count', 'aperture count', 'image area (mm2)',
        'is_mock', 'plant_type', 'sample_type',
    ]

    csv_path = RESULT_DIR / f'results_{session_id}_{int(time.time())}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            m = r.get('metrics', {})
            writer.writerow([
                r.get('file_name', ''),
                m.get('stoma_avg_height_um', 'nan'),
                m.get('stoma_avg_width_um', 'nan'),
                m.get('stoma_aspect_ratio', 'nan'),
                m.get('aperture_avg_height_um', 'nan'),
                m.get('aperture_avg_width_um', 'nan'),
                m.get('aperture_aspect_ratio', 'nan'),
                m.get('stoma_density_mm2', 'nan'),
                m.get('conductance', 'nan'),
                m.get('stoma_count', 'nan'),
                m.get('aperture_count', 'nan'),
                m.get('image_area_mm2', 'nan'),
                r.get('is_mock', True),
                r.get('plant_type', ''),
                r.get('sample_type', ''),
            ])

    return send_file(
        csv_path,
        mimetype='text/csv',
        as_attachment=True,
        attachment_filename=f'stomata_results_{session_id}.csv',
    )


@app.route('/api/session/results', methods=['GET'])
def api_session_results():
    session_id = request.args.get('session_id', 'default')
    sess = get_session(session_id)
    return jsonify({
        'session_id': session_id,
        'count': len(sess['results']),
        'results': [
            {
                'file_name': r.get('file_name'), 'source': r.get('source'),
                'metrics': r.get('metrics'),
                'overlay_b64': r.get('overlay_b64'),
                'input_b64': r.get('input_b64'),
                'is_mock': r.get('is_mock'),
                'mock_reason': r.get('mock_reason'),
                'plant_type': r.get('plant_type'),
                'sample_type': r.get('sample_type'),
            }
            for r in sess['results']
        ],
    })


@app.route('/api/session/reset', methods=['POST'])
def api_session_reset():
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id', 'default')
    if session_id in sessions:
        sessions[session_id]['results'] = []
    return jsonify({'success': True, 'session_id': session_id})


# ===========================================================================
# Agent chat endpoint  (MOCK - ready for Coze / LLM API integration)
# ===========================================================================
AGENT_SYSTEM_PROMPT = """You are the DIffCAD Agent, a plant science AI assistant specialized in
stomatal phenotyping. You help researchers:

1. Understand their input (image, batch, video, camera)
2. Plan the detection workflow
3. Invoke YOLO-OBB detection models
4. Review quality (contrast, scale, overlap)
5. Generate research-style reports

Keep answers concise and scientific."""

# ---- Conversational knowledge base ----
STOMATA_KNOWLEDGE = {
    "what_are_stomata": (
        "Stomata (singular: stoma) are microscopic pores on the surface of plant leaves and stems. "
        "Each pore is surrounded by a pair of guard cells that regulate its opening and closing. "
        "They are the primary sites of gas exchange — CO2 enters for photosynthesis, O2 exits, "
        "and water vapor escapes through transpiration. A typical leaf has 50–500 stomata per mm², "
        "varying by species, environment, and leaf side (adaxial vs. abaxial)."
    ),
    "function": (
        "Stomata serve three critical functions:\n"
        "1. **Gas exchange** — CO2 uptake for photosynthesis, O2 release\n"
        "2. **Transpiration** — Water vapor loss drives the transpiration stream, pulling nutrients from roots\n"
        "3. **Temperature regulation** — Evaporative cooling via controlled water loss\n"
        "Guard cells respond to light, CO2 concentration, humidity, and plant hormones (ABA) to open or close the pore."
    ),
    "detection_how": (
        "DIffCAD uses YOLO-OBB (Oriented Bounding Box) deep learning models to detect stomata and apertures. "
        "Unlike standard YOLO which predicts axis-aligned boxes, OBB predicts rotated boxes that better fit "
        "the elliptical shape of stomata. The model outputs: center coordinates (cx, cy), width, height, "
        "rotation angle, and class (stoma vs. aperture). Post-processing calculates density, size metrics, "
        "and stomatal conductance using the Brown-Escombe diffusion model."
    ),
    "dicot_vs_monocot": (
        "**Dicotyledons (dicots)**: Stomata are typically scattered randomly across the leaf surface. "
        "Guard cells are kidney-shaped. Examples: Arabidopsis, tomato, soybean, most broadleaf plants.\n"
        "**Monocotyledons (monocots)**: Stomata are arranged in parallel rows along the leaf axis. "
        "Guard cells are dumbbell-shaped. Examples: rice, wheat, maize, grasses.\n"
        "Choose the correct plant type in the agent panel for optimal detection accuracy."
    ),
    "density": (
        "Stomatal density (stomata/mm²) is a key phenotypic trait. It's calculated as: "
        "density = stomata_count / image_area_mm². Density varies widely:\n"
        "- Arabidopsis: ~100–300 stomata/mm²\n"
        "- Rice: ~300–700 stomata/mm²\n"
        "- Wheat: ~50–100 stomata/mm²\n"
        "Higher CO2 levels generally reduce stomatal density (CO2 fertilization effect). "
        "Drought stress often increases density but reduces individual stoma size."
    ),
    "conductance": (
        "Stomatal conductance (gs, mol H2O m⁻² s⁻¹) quantifies the rate of water vapor loss through stomata. "
        "DIffCAD estimates it using the Brown-Escombe diffusion model, which relates conductance to "
        "stomatal density, aperture dimensions, and pore depth. Typical values: 0.1–0.8 mol m⁻² s⁻¹. "
        "Conductance is a critical parameter for photosynthesis models and drought phenotyping."
    ),
    "scale_bar": (
        "Accurate scale (μm/pixel) is essential for correct measurements. You can set the scale in the "
        "Advanced panel. For microscope images: divide the field of view (μm) by image width (pixels). "
        "For images with a scale bar: measure the bar length in pixels and divide the known distance by it. "
        "Common reference: a 100 μm scale bar is about 224 pixels at 10× magnification."
    ),
    "nondestructive_vs_destructive": (
        "**Nondestructive sampling**: Uses leaf imprints (nail polish, dental paste), epidermal peels, "
        "or direct microscope imaging of living leaves. Preserves the plant for longitudinal studies.\n"
        "**Destructive sampling**: Involves harvesting leaf tissue, often with chemical clearing or fixation. "
        "Provides clearer images but terminates the sample.\n"
        "Choose the appropriate sample type in the agent panel to load the correct detection model."
    ),
}

CASUAL_RESPONSES = {
    "thanks": "You're welcome! Happy to help with your stomata research. Feel free to upload samples anytime.",
    "谢谢": "不客气！随时上传样本，我来帮您分析气孔数据。",
    "goodbye": "Goodbye! Your session data is saved. Come back anytime to continue your analysis.",
    "再见": "再见！您的会话数据已保存，随时回来继续分析。",
    "bye": "Goodbye! Your session data is saved. Come back anytime to continue your analysis.",
    "how_are_you": "I'm running at full capacity and ready to analyze your stomata samples! How can I help with your research today?",
    "你好吗": "我运行正常，随时准备分析您的样本！今天有什么研究需要帮助的吗？",
    "不错": "很高兴能帮上忙！如果有样本需要分析，或者想了解更多关于气孔检测的信息，随时告诉我。",
    "good": "Great to hear! If you have samples to analyze or want to learn more about stomata phenotyping, I'm here to help.",
}


def _detect_language(text):
    """Return 'zh' if text contains Chinese characters, 'en' otherwise."""
    for ch in text:
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            return 'zh'
    return 'en'


def _match_keywords(text, keywords):
    """Check if any keyword appears in text (case-insensitive, handles Chinese)."""
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def _generate_agent_reply(message, context):
    """
    Rich conversational agent with stomata domain knowledge.
    Returns (reply_text, is_mock).
    """
    lower = message.lower().strip()
    lang = _detect_language(message)
    metrics = context.get('metrics', {})
    batch_summary = context.get('batch_summary', '')
    video_summary = context.get('video_summary', '')
    source = context.get('source', '')
    has_results = bool(metrics and metrics.get('stoma_count') is not None)
    has_batch = bool(batch_summary)
    has_video = bool(video_summary)
    is_mock = context.get('is_mock', True)
    is_cn = (lang == 'zh')

    # === BATCH / VIDEO MULTI-ITEM CONTEXT ===
    plant_type = context.get('plant_type', 'dicotyledons')
    plant_label = 'Monocot' if plant_type == 'monocotyledons' else 'Dicot'
    plant_note = (
        '\n\n---\nDetection ran with the **' + plant_label + '** model.\n'
        'Dicot: scattered stomata, kidney-shaped guard cells — for broadleaf plants (Arabidopsis, soybean, tomato).\n'
        'Monocot: parallel-row stomata, dumbbell-shaped guard cells — for grasses & cereals (rice, wheat, maize).\n'
        '**If your sample doesn\'t match, switch the plant type chip and re-run detection.**'
        if not is_cn else
        '\n\n---\n当前使用 **' + plant_label + '** 模型进行检测。\n'
        '双子叶(Dicot)：气孔散乱分布，保卫细胞肾形 — 适用于拟南芥、大豆、番茄等阔叶植物。\n'
        '单子叶(Monocot)：气孔平行排列，保卫细胞哑铃形 — 适用于水稻、小麦、玉米等禾本科植物。\n'
        '**如果您的样本类型与当前选择不匹配，请切换芯片后重新检测。**'
    )

    if has_batch:
        if _match_keywords(lower, ['summary', 'report', 'summarize', '总结', '报告', '摘要',
                                    'analyze', '分析', '检测']):
            return (
                f"Batch Detection Complete\n{batch_summary}\n\n"
                f"Plant type: {context.get('plant_type', 'dicotyledons')}  |  "
                f"Sample: {context.get('sample_type', 'nondestructive')}\n"
                f"Scale: {context.get('scale', '100')} μm  |  "
                f"Confidence threshold: {context.get('conf', 0.5)}\n\n"
                + ('Note: results are MOCK — load model weights for publication-grade data.\n' if is_mock else '')
                + 'Open the Workspace panel to navigate individual images and export CSV.'
                + plant_note,
                True
            )
        # Default batch response — full summary
        return (
            f"Batch detection completed across all images.\n{batch_summary}\n\n"
            + ('MOCK results — load model weights for real YOLO-OBB detection.\n' if is_mock else '')
            + 'You can ask me for detailed analysis of individual images or a research report. '
            + 'Open Workspace to browse overlays.'
            + plant_note,
            True
        )

    if has_video:
        if _match_keywords(lower, ['summary', 'report', 'summarize', '总结', '报告', '摘要',
                                    'analyze', '分析', '检测']):
            return (
                f"Video Frame Analysis Complete\n{video_summary}\n\n"
                f"Plant type: {context.get('plant_type', 'dicotyledons')}  |  "
                f"Sample: {context.get('sample_type', 'nondestructive')}\n"
                f"Scale: {context.get('scale', '100')} μm  |  "
                f"Confidence threshold: {context.get('conf', 0.5)}\n\n"
                + ('Note: results are MOCK — load model weights for publication-grade data.\n' if is_mock else '')
                + 'Open the Workspace panel to inspect frame-by-frame overlays and export CSV.'
                + plant_note,
                True
            )
        return (
            f"Video frame analysis complete.\n{video_summary}\n\n"
            + ('MOCK results — load model weights for real YOLO-OBB detection.\n' if is_mock else '')
            + 'You can ask me for detailed analysis or a research report. '
            + 'Open Workspace to browse frame overlays.'
            + plant_note,
            True
        )

    # === DETECTION RESULTS CONTEXT (single image) ===
    if has_results:
        density = metrics.get('stoma_density_mm2', '?')
        stoma_n = metrics.get('stoma_count', '?')
        aperture_n = metrics.get('aperture_count', '?')
        cond = metrics.get('conductance', '?')
        sw = metrics.get('stoma_avg_width_um', '?')
        aw = metrics.get('aperture_avg_width_um', '?')

        if _match_keywords(lower, ['summary', 'report', 'summarize', '总结', '报告', '摘要']):
            return (
                f"Research Summary [{plant_label} model]\n"
                f"- Stomata density: {density} stomata/mm²\n"
                f"- Total stomata: {stoma_n}  |  Apertures: {aperture_n}\n"
                f"- Stomata avg width: {sw} μm  |  Aperture avg width: {aw} μm\n"
                f"- Conductance: {cond} mol m⁻² s⁻¹\n"
                + ('[MOCK DATA — connect model weights for publication-grade results]\n' if is_mock else '')
                + 'Open the Workspace panel to inspect overlays and export CSV.'
                + plant_note,
                True
            )
        if _match_keywords(lower, ['quality', 'check', 'reliability', '质量', '可靠性', '检查']):
            return (
                f"Quality Assessment [{plant_label}]\n"
                f"- Detection engine: {'MOCK (no weights loaded)' if is_mock else 'YOLO-OBB live model'}\n"
                f"- Scale: {context.get('scale', '100')} μm  |  Confidence threshold: {context.get('conf', 0.5)}\n"
                f"- Detected: {stoma_n} stomata, {aperture_n} aperture regions\n"
                f"- Manual review recommended for crowded or low-contrast regions.\n"
                + ('Recommendation: load a .pt weight file for real YOLO-OBB detection.' if is_mock else '')
                + plant_note,
                True
            )
        if _match_keywords(lower, ['explain', 'what', 'how', '解释', '说明', '怎么', '什么']):
            return (
                f"I detected {stoma_n} stomata and {aperture_n} aperture regions using "
                f"{'MOCK simulation' if is_mock else 'YOLO-OBB'} [{plant_label} model] on your {context.get('source', 'image')}. "
                f"Density = count / image_area = {density} stomata/mm². "
                f"Conductance ({cond} mol m⁻² s⁻¹) uses the Brown-Escombe diffusion model. "
                f"Stomata are oriented bounding boxes (OBB) — each has center, width, height, and rotation angle."
                + plant_note,
                True
            )
        # Generic result context
        if is_cn:
            return (
                f"[{plant_label}] 当前检测结果：{stoma_n} 个气孔，{aperture_n} 个开度区域，"
                f"密度 {density} stomata/mm²。您可以让我生成摘要、检查质量或解释检测原理。"
                + plant_note,
                True
            )
        return (
            f"[{plant_label}] Result: {stoma_n} stomata, {aperture_n} apertures, density {density} stomata/mm². "
            f"You can ask me for a summary, quality check, or explanation of how detection works."
            + plant_note,
            True
        )

    # === GREETINGS & IDENTITY ===
    if _match_keywords(lower, ['hello', 'hi', 'hey', '你好', '您好', '早上好', '晚上好',
                                '你是谁', 'who are you', '你叫什么', '介绍', '自我介绍',
                                '你的名字', 'what is your name', '你是什么']):
        if is_cn:
            return (
                "您好！我是 **DIffCAD Agent**，一个植物气孔检测与表型分析智能助手。\n\n"
                "我可以帮您：\n"
                "- 检测显微图像中的气孔和开度\n"
                "- 计算密度、导度、尺寸等表型指标\n"
                "- 处理批量图片、视频和实时摄像头\n"
                "- 导出 CSV 数据用于论文发表\n\n"
                "点击左下角的 **+** 按钮上传样本，或者直接跟我聊聊气孔生物学！",
                True
            )
        return (
            "Hello! I'm **DIffCAD Agent**, a plant stomata detection and phenotyping assistant.\n\n"
            "I can help you:\n"
            "- Detect stomata and apertures in microscope images\n"
            "- Calculate density, conductance, size metrics\n"
            "- Process batches, videos, and live camera feeds\n"
            "- Export CSV data for publications\n\n"
            "Click the **+** button to upload a sample, or ask me anything about stomata biology!",
            True
        )

    # === HELP / CAPABILITIES ===
    if _match_keywords(lower, ['help', '帮助', '功能', 'what can you do', '你能做什么', 'commands']):
        if is_cn:
            return (
                "**DIffCAD Agent 功能列表：**\n\n"
                "1. **检测气孔** — 上传图片/文件夹/视频/摄像头，使用 YOLO-OBB 模型检测\n"
                "2. **表型计算** — 密度、导度、气孔尺寸、开度尺寸、长宽比\n"
                "3. **批量处理** — 一次分析多张图片，自动汇总统计\n"
                "4. **质量审查** — 检查图像质量、比例尺、重叠区域\n"
                "5. **导出 CSV** — 一键导出数据用于统计分析和论文\n\n"
                "**对话话题：** 问我关于气孔生物学、检测原理、实验设计等问题！\n"
                "点击 **+** 按钮上传样本即可开始。",
                True
            )
        return (
            "**DIffCAD Agent Capabilities:**\n\n"
            "1. **Detection** — Upload images/folders/videos/camera for YOLO-OBB stomata detection\n"
            "2. **Phenotyping** — Density, conductance, stoma/aperture size, aspect ratio\n"
            "3. **Batch Processing** — Multi-image analysis with aggregated statistics\n"
            "4. **Quality Control** — Image quality, scale verification, overlap review\n"
            "5. **CSV Export** — One-click export for statistical analysis and publication\n\n"
            "**Ask me about:** stomata biology, detection methods, experimental design, plant physiology!\n"
            "Click the **+** button to upload a sample and get started.",
            True
        )

    # === STOMATA KNOWLEDGE Q&A ===
    if _match_keywords(lower, ['what are stomata', 'what is stomata', 'stomata definition',
                                '什么是气孔', '气孔是什么', '气孔定义']):
        return (STOMATA_KNOWLEDGE["what_are_stomata"], True)

    if _match_keywords(lower, ['function of stomata', 'stomata function', '气孔功能',
                                '气孔作用', 'stomata role']):
        return (STOMATA_KNOWLEDGE["function"], True)

    if _match_keywords(lower, ['how does detection work', 'how do you detect', 'detection method',
                                'yolo', 'obb', '检测原理', '怎么检测', '检测方法']):
        return (STOMATA_KNOWLEDGE["detection_how"], True)

    if _match_keywords(lower, ['dicot', 'monocot', '双子叶', '单子叶', 'plant type',
                                'dicot vs monocot', 'difference between']):
        return (STOMATA_KNOWLEDGE["dicot_vs_monocot"], True)

    if _match_keywords(lower, ['density', 'stomatal density', 'how many stomata',
                                '气孔密度', '密度']):
        return (STOMATA_KNOWLEDGE["density"], True)

    if _match_keywords(lower, ['conductance', 'gs', 'stomatal conductance',
                                '导度', '气孔导度']):
        return (STOMATA_KNOWLEDGE["conductance"], True)

    if _match_keywords(lower, ['scale', 'scale bar', 'pixel', 'μm', 'micron',
                                '比例尺', '标尺', '像素', '微米', 'scale setting']):
        return (STOMATA_KNOWLEDGE["scale_bar"], True)

    if _match_keywords(lower, ['nondestructive', 'destructive', '非破坏', '破坏性',
                                '无损', 'sampling method', 'sample type']):
        return (STOMATA_KNOWLEDGE["nondestructive_vs_destructive"], True)

    # === PLANT / BIOLOGY GENERAL ===
    if _match_keywords(lower, ['photosynthesis', '光合作用']):
        return (
            "Photosynthesis is the process by which plants convert light energy, CO2, and water into "
            "glucose and O2. Stomata are the entry points for CO2 — without them, photosynthesis cannot "
            "occur. Stomatal conductance directly limits photosynthetic rate under most conditions. "
            "C3 plants (rice, wheat, soy) typically have higher stomatal conductance than C4 plants (maize, sorghum).",
            True
        )

    if _match_keywords(lower, ['transpiration', '蒸腾']):
        return (
            "Transpiration is the loss of water vapor through stomata. It creates negative pressure that "
            "pulls water and dissolved nutrients from roots through the xylem (transpiration stream). "
            "About 97–99% of water absorbed by roots is lost through transpiration. Stomatal regulation "
            "balances the trade-off between CO2 uptake (photosynthesis) and water loss (transpiration).",
            True
        )

    if _match_keywords(lower, ['guard cell', '保卫细胞']):
        return (
            "Guard cells are specialized epidermal cells that surround each stoma. They change shape in "
            "response to turgor pressure: when swollen (high turgor), they bow apart and open the pore; "
            "when flaccid (low turgor), they relax and close it. Key regulators include: blue light "
            "(opens), ABA hormone (closes), high CO2 (closes), and potassium ion (K+) fluxes.",
            True
        )

    if _match_keywords(lower, ['aperture', '开度', 'pore size', '开度大小']):
        return (
            "The stomatal aperture is the pore opening between guard cells. Aperture size directly "
            "determines gas exchange capacity. Key metrics: aperture width (μm), aperture length (μm), "
            "and aperture area (often approximated as an ellipse: π × a × b). Aperture responds to "
            "light within minutes — blue light triggers rapid opening via phototropin signaling.",
            True
        )

    if _match_keywords(lower, ['aba', 'abscisic acid', '脱落酸', 'drought', '干旱',
                                'water stress', '水分胁迫']):
        return (
            "Abscisic acid (ABA) is the key drought-stress hormone that triggers stomatal closure. "
            "When roots detect dry soil, ABA is synthesized and transported to leaves, where it binds "
            "to guard cell receptors (PYR/PYL/RCAR), initiating a signaling cascade that releases Ca²⁺, "
            "activates anion channels, and causes K⁺ efflux — closing the pore within minutes. "
            "Stomatal density and ABA sensitivity are major targets for drought-resistance breeding.",
            True
        )

    if _match_keywords(lower, ['climate', 'co2', 'global warming', '气候变化', '二氧化碳']):
        return (
            "Rising atmospheric CO2 affects stomata in two ways:\n"
            "1. **Short-term**: High CO2 directly triggers partial stomatal closure (reducing conductance)\n"
            "2. **Long-term**: Plants grown at elevated CO2 often develop fewer stomata (lower density)\n"
            "This is studied using historical herbarium specimens and FACE (Free-Air CO2 Enrichment) "
            "experiments. Stomatal density is used as a paleo-CO2 proxy in fossil leaves.",
            True
        )

    # === EXPERIMENTAL GUIDANCE ===
    if _match_keywords(lower, ['how to image', 'imaging', 'microscope', '拍照', '显微镜',
                                'how to take', 'imaging protocol']):
        if is_cn:
            return (
                "**气孔成像建议：**\n"
                "1. 使用 10×–40× 物镜（10× 适合密度统计，40× 适合尺寸测量）\n"
                "2. 确保光照均匀，避免反光\n"
                "3. 使用比例尺或记录放大倍数以便后续校准\n"
                "4. 每个叶片至少拍摄 3–5 个视野以提高统计可靠性\n"
                "5. 对于活体成像，可使用指甲油印迹法或牙科糊剂制作表皮印迹\n"
                "6. 图像保存为 JPEG 或 PNG 格式，分辨率建议 1024×768 以上",
                True
            )
        return (
            "**Stomata Imaging Tips:**\n"
            "1. Use 10×–40× objective (10× for density, 40× for size measurements)\n"
            "2. Ensure even illumination, avoid glare\n"
            "3. Include a scale bar or record magnification for calibration\n"
            "4. Image 3–5 fields of view per leaf for statistical reliability\n"
            "5. For live imaging: nail polish imprints or dental paste epidermal peels work well\n"
            "6. Save as JPEG or PNG, resolution 1024×768 or higher recommended",
            True
        )

    # === CASUAL CONVERSATION ===
    for kw, resp in CASUAL_RESPONSES.items():
        if kw.lower() in lower:
            return (resp, True)

    if _match_keywords(lower, ['thank', 'thx', '谢谢', '感谢', '多谢']):
        return (CASUAL_RESPONSES["thanks"] if not is_cn else CASUAL_RESPONSES["谢谢"], True)

    if _match_keywords(lower, ['bye', 'goodbye', 'see you', '拜拜', '再见', '回头见']):
        return (CASUAL_RESPONSES["goodbye"] if not is_cn else CASUAL_RESPONSES["再见"], True)

    if _match_keywords(lower, ['how are you', 'how r u', '你怎么样', '你还好吗', 'how is it going']):
        return (CASUAL_RESPONSES["how_are_you"] if not is_cn else CASUAL_RESPONSES["你好吗"], True)

    if _match_keywords(lower, ['nice', 'cool', 'awesome', 'great', '太棒了', '不错', '很好', '厉害']):
        return (
            "Glad you think so! 😊 Ready to analyze more samples when you are."
            if not is_cn else
            "谢谢！😊 随时可以分析更多样本，或者问我关于气孔的任何问题。",
            True
        )

    if _match_keywords(lower, ['sorry', 'apologize', '对不起', '抱歉']):
        return (
            "No worries at all! Let's continue with your stomata analysis."
            if not is_cn else
            "没关系！我们继续分析气孔数据吧。",
            True
        )

    if _match_keywords(lower, ['joke', 'funny', '笑话', '搞笑', '幽默']):
        return (
            "Why did the guard cell go to the gym? ... To work on its stomatal conductance! 😄\n"
            "Alright, back to science — ready to analyze some samples?",
            True
        )

    if _match_keywords(lower, ['who made you', 'who created you', 'creator', '谁做的', '谁开发的', '开发者']):
        return (
            "I was developed as a research tool for plant scientists studying stomatal phenotyping. "
            "I combine YOLO-OBB computer vision with phenotype calculation to help researchers "
            "quantify stomatal traits from microscope images.",
            True
        )

    # === WORKFLOW / TOOL USAGE ===
    if _match_keywords(lower, ['workflow', 'plan', '流程', '步骤', '怎么用', 'how to use', '使用']):
        if is_cn:
            return (
                "**DIffCAD 使用流程：**\n"
                "1. 点击左下角 **+** 按钮上传样本（图片/文件夹/视频/摄像头）\n"
                "2. 选择植物类型（双子叶/单子叶）和采样方式（非破坏性/破坏性）\n"
                "3. 点击 **Run Detection** 运行检测\n"
                "4. 在 Workspace 中查看叠加结果和详细指标\n"
                "5. 导出 CSV 数据用于后续分析\n\n"
                "每一步都可以随时问我问题！",
                True
            )
        return (
            "**DIffCAD Workflow:**\n"
            "1. Click the **+** button to upload samples (image/folder/video/camera)\n"
            "2. Select plant type (dicot/monocot) and sampling method (nondestructive/destructive)\n"
            "3. Click **Run Detection** to start analysis\n"
            "4. Open **Workspace** to inspect overlays and detailed metrics\n"
            "5. Export CSV for further analysis\n\n"
            "Feel free to ask questions at any step!",
            True
        )

    # === FALLBACK — encourage upload ===
    if is_cn:
        return (
            "感谢您的消息！我是 DIffCAD Agent，专注于植物气孔检测和表型分析。\n\n"
            "您可以：\n"
            "- 问我关于气孔生物学、检测原理、实验设计的问题\n"
            "- 上传显微图像进行气孔检测\n"
            "- 让我帮您理解检测结果\n\n"
            "点击左下角的 **+** 按钮上传样本，或者试试问我「气孔是什么」「怎么检测气孔」等问题！",
            True
        )
    return (
        "Thanks for your message! I'm the DIffCAD Agent, focused on plant stomata detection and phenotyping.\n\n"
        "You can:\n"
        "- Ask me about stomata biology, detection methods, or experimental design\n"
        "- Upload microscope images for stomata detection\n"
        "- Get help interpreting your detection results\n\n"
        "Click the **+** button to upload a sample, or try asking me \"what are stomata\" or \"how does detection work\"!",
        True
    )


@app.route('/api/agent/chat', methods=['POST'])
def api_agent_chat():
    """
    Agent chat endpoint with rich conversational AI.

    *** MOCK IMPLEMENTATION with domain knowledge ***
    Replace with Coze / OpenAI / Anthropic API call for full intelligence.
    See [LLM-INTEGRATION] markers below.
    """
    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    context = data.get('context', {})

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # [LLM-INTEGRATION] Replace this block with real LLM API call:
    #   import openai  (or anthropic, coze SDK, etc.)
    #   response = openai.ChatCompletion.create(
    #       model="gpt-4",
    #       messages=[
    #           {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    #           {"role": "user", "content": message},
    #       ],
    #   )
    #   reply = response.choices[0].message.content
    #   return jsonify({'success': True, 'reply': reply, 'is_mock_agent': False, ...})

    reply, is_mock = _generate_agent_reply(message, context)

    return jsonify({
        'success': True,
        'reply': reply,
        'is_mock_agent': is_mock,
        'session_id': session_id,
    })


# ---------------------------------------------------------------------------
# Frontend serving
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Serve the landing page as the homepage."""
    return send_from_directory(str(BASE_DIR), 'landing_page.html')


@app.route('/agent')
def agent_page():
    """Serve the agent frontend."""
    return send_from_directory(str(BASE_DIR), 'stomata_agent_frontend.html')


@app.route('/icons/<path:filename>')
def serve_icons(filename):
    """Serve icon files from the icons/ directory."""
    icons_dir = BASE_DIR / 'icons'
    if icons_dir.is_dir():
        return send_from_directory(str(icons_dir), filename)
    return jsonify({'error': 'Icon not found'}), 404


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 60)
    print('  DIffCAD Agent Backend  v2.1.0')
    print('=' * 60)
    print(f'  Device:       {DEVICE}')
    print(f'  Torch:        {HAS_TORCH}')
    print(f'  ultralytics:  {HAS_YOLO}')

    scan = model_manager.scan_weights()
    print(f'  Weight dir:   {WEIGHT_DIR}  (exists: {scan["weight_dir_exists"]})')

    available = model_manager.available_combos()
    missing = model_manager.missing_weights()

    if available:
        print(f'  Available weights ({len(available)}):')
        for p, s in available:
            print(f'    - {p} + {s}')
    if missing:
        print(f'  Missing weights ({len(missing)}):')
        for p, s in missing:
            print(f'    - {p} + {s}  ->  {DETECTION_WEIGHTS.get((p,s), "?")}')

    resto = model_manager.get_restoration_info()
    if resto['available']:
        print(f'  Restoration weights available: {len(resto["available"])} (not loaded)')

    print(f'  Upload dir:   {UPLOAD_DIR}')
    print(f'  Result dir:   {RESULT_DIR}')
    print('=' * 60)

    # Preload models only on local dev (cloud deployments use lazy loading to save RAM)
    if os.environ.get('PRELOAD_MODELS', '').lower() == '1':
        if available:
            print('[init] Preloading available models ...')
            model_manager.preload_all()
            print('[init] Ready.')
    else:
        if available:
            print('[init] Lazy-load mode: models will be loaded on first request.')

    if not available:
        print('[init] WARNING: No detection weights available. '
              'All requests will use MOCK data.')
        print(f'[init] Expected weight files in: {WEIGHT_DIR}')

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
