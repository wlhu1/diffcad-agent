# -*- coding: utf-8 -*-
"""
ONNX-based detection worker.  No PyTorch / ultralytics dependency — uses
onnxruntime + opencv + numpy only.  Fits comfortably in 512 MB RAM.

Usage:
  python onnx_worker.py <image_path> <plant_type> <sample_type> [conf] [iou] [scale]

Output: JSON on stdout (single line)
Exit code: 0 on success, 1 on error
"""
import os
import sys
import json

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEIGHT_DIR = BASE_DIR / 'weights'
ONNX_WEIGHTS = {
    ('dicotyledons', 'destructive'):     'dicotyledons_destructive.onnx',
    ('dicotyledons', 'nondestructive'):  'dicotyledons_nondestructive.onnx',
}

# ---- ONNX model loading ----

_session_cache = None


def load_onnx_model(plant_type: str, sample_type: str):
    """Load the ONNX model with onnxruntime (lazy import for memory)."""
    import onnxruntime as ort

    key = (plant_type, sample_type)
    weight_name = ONNX_WEIGHTS.get(key)
    if not weight_name:
        raise ValueError(f'No ONNX model for: {plant_type} / {sample_type}')

    weight_path = WEIGHT_DIR / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f'ONNX model not found: {weight_path}')

    sess = ort.InferenceSession(str(weight_path), providers=['CPUExecutionProvider'])
    # Dry-run warm-up
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    sess.run(None, {'images': dummy})
    return sess


# ---- Preprocessing ----

def preprocess(image: np.ndarray, imgsz: int = 640):
    """Letterbox resize, normalize, and convert to NCHW tensor."""
    h0, w0 = image.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
    pad_h = imgsz - new_h
    pad_w = imgsz - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
    return tensor, (r, top, left, new_h, new_w)


# ---- Post-processing ----

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _get_covariance_matrix(boxes):
    """Convert OBB to Gaussian covariance components. (NumPy port of ultralytics)

    boxes: (N, 5)  [cx, cy, w, h, angle_rad]
    Returns: a, b, c  each (N,)
    """
    w = boxes[:, 2]
    h = boxes[:, 3]
    angle = boxes[:, 4]
    a = w * w / 12.0
    b = h * h / 12.0
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    cos2 = cos_a * cos_a
    sin2 = sin_a * sin_a
    return (a * cos2 + b * sin2,
            a * sin2 + b * cos2,
            (a - b) * cos_a * sin_a)


def _batch_probiou(boxes1, boxes2, eps=1e-7):
    """Probabilistic IoU matrix (batch). (NumPy port of ultralytics batch_probiou)

    boxes1: (N, 5)  boxes2: (M, 5)
    Returns: (N, M) probiou matrix.
    """
    # Centers: (N,) and (M,) — 1-d for broadcasting
    x1, y1 = boxes1[:, 0], boxes1[:, 1]  # (N,)
    x2, y2 = boxes2[:, 0], boxes2[:, 1]  # (M,)

    # Covariance components
    a1, b1, c1 = _get_covariance_matrix(boxes1)  # each (N,)
    a2, b2, c2 = _get_covariance_matrix(boxes2)  # each (M,)

    # Broadcast to (N, M)
    a1 = a1[:, None]; b1 = b1[:, None]; c1 = c1[:, None]  # (N, 1)
    a2 = a2[None, :]; b2 = b2[None, :]; c2 = c2[None, :]  # (1, M)

    dx = x2[None, :] - x1[:, None]  # (N, M)
    dy = y2[None, :] - y1[:, None]

    a_sum = a1 + a2
    b_sum = b1 + b2
    c_sum = c1 + c2
    det = a_sum * b_sum - c_sum * c_sum + eps

    t1 = ((a_sum * dy * dy + b_sum * dx * dx) / det) * 0.25
    t2 = ((c_sum * (-dx) * dy) / det) * 0.5

    safe1 = np.maximum(0.0, a1[:, 0] * b1[:, 0] - c1[:, 0] * c1[:, 0])[:, None]  # (N, 1)
    safe2 = np.maximum(0.0, a2[0, :] * b2[0, :] - c2[0, :] * c2[0, :])[None, :]  # (1, M)
    denom = 4.0 * np.sqrt(safe1 * safe2) + eps
    t3 = np.log(np.maximum(det / denom, eps)) * 0.5

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1.0 - hd  # (N, M)


def _nms_rotated(boxes, scores, conf_thr, iou_thr):
    """NMS with probiou (matching ultralytics behaviour).

    boxes: (N, 5)  [cx, cy, w, h, angle]
    scores: (N,)
    """
    keep_mask = scores >= conf_thr
    if not keep_mask.any():
        return np.array([], dtype=np.int64)

    indices = np.where(keep_mask)[0]
    boxes = boxes[indices]
    scores = scores[indices]
    n = len(boxes)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        remaining = order[1:]
        # Compute probiou between top box and remaining
        iou_row = _batch_probiou(boxes[i:i+1], boxes[remaining])[0]  # (M,)
        order = remaining[iou_row <= iou_thr]

    keep = np.array(keep, dtype=np.int64)
    return indices[keep]


def postprocess(predictions: np.ndarray, h0: int, w0: int,
                conf_thr: float, iou_thr: float,
                r: float, pad_top: int, pad_left: int):
    """Decode ONNX output (1, 7, 8400) → list of [cx, cy, w, h, angle, cls, conf]."""
    preds = predictions[0]  # (7, 8400)

    # Channels: [cx, cy, w, h, angle, c0_logit, c1_logit]
    cx_640, cy_640 = preds[0], preds[1]
    w_640, h_640 = preds[2], preds[3]
    angle = preds[4]
    c0_logit = preds[5]
    c1_logit = preds[6]

    # Confidence filtering on raw logits (matching ultralytics behaviour).
    # max across class logits is the effective detection score.
    max_logit = np.maximum(c0_logit, c1_logit)
    best_class = np.where(c0_logit >= c1_logit, 0, 1)

    # Stack boxes in 640×640 space for NMS
    boxes_640 = np.stack([cx_640, cy_640, w_640, h_640, angle], axis=1)

    # NMS (filtering by raw logit threshold internally)
    keep_idx = _nms_rotated(boxes_640, max_logit, conf_thr, iou_thr)

    if len(keep_idx) == 0:
        return []

    # Map coordinates from 640×640 model space back to original image
    result = []
    for idx in keep_idx:
        result.append([
            float((cx_640[idx] - pad_left) / r),
            float((cy_640[idx] - pad_top) / r),
            float(w_640[idx] / r),
            float(h_640[idx] / r),
            float(angle[idx]),
            int(best_class[idx]),
            float(_sigmoid(max_logit[idx])),  # confidence as probability
        ])
    return result


# ---- Detection entry ----

def detect(image: np.ndarray, model, conf: float, iou: float, scale: float):
    """Full ONNX detection pipeline."""
    h0, w0 = image.shape[:2]
    tensor, (r, pad_top, pad_left, new_h, new_w) = preprocess(image)

    outputs = model.run(None, {'images': tensor})
    predictions = outputs[0]

    boxes = postprocess(predictions, h0, w0, conf, iou, r, pad_top, pad_left)

    h, w = h0, w0
    metrics = _compute_metrics_from_boxes(boxes, h, w, scale)
    overlay = _render_overlay(image, boxes)

    import base64
    _, input_buf = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    _, overlay_buf = cv2.imencode('.jpg', overlay)

    return {
        'success': True,
        'metrics': metrics,
        'overlay_b64': base64.b64encode(overlay_buf).decode(),
        'input_b64': base64.b64encode(input_buf).decode(),
        'boxes': boxes,
        'is_mock': False,
        'mock_reason': None,
    }


def _compute_metrics_from_boxes(boxes, h, w, scale):
    """Compute stomata phenotyping metrics from OBB boxes. (Same logic as detection_worker)"""
    if not boxes:
        return _empty_metrics(h, w)

    stoma_boxes = [b for b in boxes if int(b[5]) == 0]
    aperture_boxes = [b for b in boxes if int(b[5]) == 1]

    stoma_heights = [b[3] * scale for b in stoma_boxes]
    stoma_widths = [b[2] * scale for b in stoma_boxes]

    aperture_heights = [b[3] * scale for b in aperture_boxes]
    aperture_widths = [b[2] * scale for b in aperture_boxes]

    image_area_mm2 = (h * w * scale * scale) / 1_000_000

    stoma_count = len(stoma_boxes)
    aperture_count = len(aperture_boxes)

    stoma_density = stoma_count / image_area_mm2 if image_area_mm2 > 0 else 0

    conductance = 0.0
    if aperture_count > 0 and stoma_count > 0:
        avg_aperture_area = (sum(aperture_heights) / aperture_count) * (sum(aperture_widths) / aperture_count)
        pore_area_fraction = (avg_aperture_area * aperture_count) / (image_area_mm2 * 1_000_000)
        conductance = round(pore_area_fraction * 1.6 * 0.025, 4)

    return {
        'stoma_count': stoma_count,
        'aperture_count': aperture_count,
        'stoma_avg_height_um': round(np.mean(stoma_heights), 2) if stoma_heights else 0,
        'stoma_avg_width_um': round(np.mean(stoma_widths), 2) if stoma_widths else 0,
        'stoma_aspect_ratio': round(np.mean(stoma_heights) / (np.mean(stoma_widths) or 1), 2) if stoma_boxes else 0,
        'aperture_avg_height_um': round(np.mean(aperture_heights), 2) if aperture_heights else 0,
        'aperture_avg_width_um': round(np.mean(aperture_widths), 2) if aperture_widths else 0,
        'aperture_aspect_ratio': round(np.mean(aperture_heights) / (np.mean(aperture_widths) or 1), 2) if aperture_boxes else 0,
        'stoma_density_mm2': round(stoma_density, 2),
        'conductance': conductance,
        'image_area_mm2': round(image_area_mm2, 4),
    }


def _empty_metrics(h, w):
    area = (h * w) / 1_000_000 if (h * w) > 0 else 0
    return {
        'stoma_count': 0, 'aperture_count': 0,
        'stoma_avg_height_um': 0, 'stoma_avg_width_um': 0,
        'stoma_aspect_ratio': 0, 'aperture_avg_height_um': 0,
        'aperture_avg_width_um': 0, 'aperture_aspect_ratio': 0,
        'stoma_density_mm2': 0, 'conductance': 0,
        'image_area_mm2': round(area, 4),
    }


def _render_overlay(image, boxes):
    """Render rotated boxes on image. (Same logic as detection_worker)"""
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    for box in boxes:
        cx, cy, bw, bh, angle, cls = box[:6]
        rect = ((cx, cy), (bw, bh), np.degrees(angle))
        color = (0, 255, 128) if int(cls) == 0 else (255, 200, 60)
        pts = cv2.boxPoints(rect)
        pts = np.int32(pts)
        cv2.drawContours(img, [pts], 0, color, 1)
    return img


# ---- CLI entry ----

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(json.dumps({'error': 'Usage: onnx_worker.py <image_path> <plant_type> <sample_type> [conf] [iou] [scale]'}))
        sys.exit(1)

    image_path = sys.argv[1]
    plant_type = sys.argv[2]
    sample_type = sys.argv[3]
    conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    iou = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
    scale = float(sys.argv[6]) if len(sys.argv) > 6 else 100.0

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'Cannot read image: {image_path}')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        model = load_onnx_model(plant_type, sample_type)
        result = detect(img_rgb, model, conf, iou, scale)
        result['weight_path'] = str(WEIGHT_DIR / ONNX_WEIGHTS[(plant_type, sample_type)])
        result['plant_type'] = plant_type
        result['sample_type'] = sample_type

        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({'error': str(e), 'success': False}, ensure_ascii=False))
        sys.exit(1)
