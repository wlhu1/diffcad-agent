# -*- coding: utf-8 -*-
"""
Standalone detection worker — runs in a subprocess so PyTorch memory is
fully released after each request.  Designed for cloud free-tier (512 MB RAM).

Usage:
  python detection_worker.py <image_path> <plant_type> <sample_type> [conf] [iou] [scale]

Output: JSON on stdout (single line)
Exit code: 0 on success, 1 on error
"""
import os
import sys
import json

# Memory-limit environment (set before any heavy import)
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
DETECTION_WEIGHTS = {
    ('dicotyledons', 'destructive'):     'dicotyledons_destructive.pt',
    ('dicotyledons', 'nondestructive'):  'dicotyledons_nondestructive.pt',
    # Monocotyledons models excluded from cloud deployment (memory constraint).
    # They remain available for local use.
}


def load_model(plant_type: str, sample_type: str):
    """Import torch + ultralytics and load YOLO model on demand."""
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    from ultralytics import YOLO
    import ultralytics.utils.loss as loss_mod
    if not hasattr(loss_mod, 'SoftmaxEQLV2Loss'):
        class SoftmaxEQLV2Loss(torch.nn.Module):
            def __init__(self, *args, **kwargs): super().__init__()
            def forward(self, *args, **kwargs): return torch.tensor(0.0)
        loss_mod.SoftmaxEQLV2Loss = SoftmaxEQLV2Loss

    weight_name = DETECTION_WEIGHTS.get((plant_type, sample_type))
    if not weight_name:
        raise ValueError(f'Unknown weights: {plant_type} / {sample_type}')

    weight_path = WEIGHT_DIR / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f'Weight file not found: {weight_path}')

    model = YOLO(str(weight_path), task='obb')
    # warm-up
    model(np.zeros((48, 48, 3), dtype=np.uint8), device='cpu')
    return model


def detect(image: np.ndarray, model, conf: float, iou: float, scale: float):
    """Run detection and build metrics."""
    import torch
    with torch.no_grad():
        results = model(image, conf=conf, iou=iou, device='cpu', verbose=False)

    if not results or len(results) == 0:
        return _empty_result(image, True, 'No results from model')

    result = results[0]
    if result.obb is None:
        return _empty_result(image, True, 'OBB output is None')

    boxes = []
    try:
        obb_data = result.obb.data
        if obb_data is not None and obb_data.shape[0] > 0:
            boxes = obb_data.cpu().numpy().tolist()
    except Exception:
        boxes = []

    h, w = image.shape[:2]
    metrics = _compute_metrics_from_boxes(boxes, h, w, scale)
    overlay = _render_overlay(image, boxes, False, None)

    _, input_buf = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    _, overlay_buf = cv2.imencode('.jpg', overlay)

    import base64
    return {
        'success': True,
        'metrics': metrics,
        'overlay_b64': base64.b64encode(overlay_buf).decode(),
        'input_b64': base64.b64encode(input_buf).decode(),
        'boxes': boxes,
        'is_mock': False,
        'mock_reason': None,
    }


def _empty_result(image, is_mock, reason):
    h, w = image.shape[:2]
    import base64
    _, input_buf = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return {
        'success': True,
        'metrics': {
            'stoma_count': 0, 'aperture_count': 0,
            'stoma_avg_height_um': 0, 'stoma_avg_width_um': 0,
            'stoma_aspect_ratio': 0, 'aperture_avg_height_um': 0,
            'aperture_avg_width_um': 0, 'aperture_aspect_ratio': 0,
            'stoma_density_mm2': 0, 'conductance': 0,
            'image_area_mm2': (h * w) / 1_000_000,
        },
        'overlay_b64': base64.b64encode(cv2.imencode('.jpg', image)[1]).decode(),
        'input_b64': base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1]).decode(),
        'boxes': [],
        'is_mock': is_mock,
        'mock_reason': reason,
    }


def _compute_metrics_from_boxes(boxes, h, w, scale):
    """Compute stomata phenotyping metrics from OBB detection boxes.

    Each box: [cx, cy, width, height, angle, class]
    class 0 = stoma, class 1 = aperture
    """
    if not boxes:
        return _empty_metrics(h, w)

    stoma_boxes = [b for b in boxes if int(b[5]) == 0]
    aperture_boxes = [b for b in boxes if int(b[5]) == 1]

    # Scale factor: convert pixels to µm
    stoma_heights = [b[3] * scale for b in stoma_boxes]  # box height in µm
    stoma_widths  = [b[2] * scale for b in stoma_boxes]  # box width in µm

    aperture_heights = [b[3] * scale for b in aperture_boxes]
    aperture_widths  = [b[2] * scale for b in aperture_boxes]

    image_area_mm2 = (h * w * scale * scale) / 1_000_000

    stoma_count = len(stoma_boxes)
    aperture_count = len(aperture_boxes)

    stoma_density = stoma_count / image_area_mm2 if image_area_mm2 > 0 else 0
    # Simplified conductance model
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


def _render_overlay(image, boxes, is_mock, mock_reason):
    """Minimal overlay rendering (mirrors backend render_overlay)."""
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

    if is_mock:
        cv2.putText(img, f'MOCK - {mock_reason or "no weights"}'[:50],
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
    return img


# ====== CLI entry ======
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(json.dumps({'error': 'Usage: detection_worker.py <image_path> <plant_type> <sample_type> [conf] [iou] [scale]'}))
        sys.exit(1)

    image_path = sys.argv[1]
    plant_type = sys.argv[2]
    sample_type = sys.argv[3]
    conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    iou  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
    scale = float(sys.argv[6]) if len(sys.argv) > 6 else 100.0

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'Cannot read image: {image_path}')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        model = load_model(plant_type, sample_type)
        result = detect(img_rgb, model, conf, iou, scale)
        result['weight_path'] = str(WEIGHT_DIR / DETECTION_WEIGHTS[(plant_type, sample_type)])
        result['plant_type'] = plant_type
        result['sample_type'] = sample_type

        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({'error': str(e), 'success': False}, ensure_ascii=False))
        sys.exit(1)
