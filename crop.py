from typing import Any, Union
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO
import math
import numpy as np


def load_clothing_model(
    repo_id: str = "kesimeg/yolov8n-clothing-detection",
    filename: str = "best.pt"
) -> YOLO:
    """Download (cached) and initialize the YOLO clothing model once."""
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(model_path)


def rank_prominence(
    conf: float,
    box: list[int],
    img_w: int,
    img_h: int,
    w_conf: float = 0.40,
    w_area: float = 0.35,
    w_pos: float = 0.25,
) -> float:
    """Calculates a visual prominence score using confidence, area, and centrality."""
    x1, y1, x2, y2 = box
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)

    # 1. Area ratio (relative to full image)
    img_area = img_w * img_h
    area_ratio = (box_w * box_h) / img_area if img_area > 0 else 0.0
    # Optional: apply sqrt to avoid extreme skew toward massive background boxes
    norm_area = min(1.0, math.sqrt(area_ratio))

    # 2. Centrality (distance from box center to image center)
    box_cx, box_cy = x1 + box_w / 2.0, y1 + box_h / 2.0
    img_cx, img_cy = img_w / 2.0, img_h / 2.0

    dist = math.hypot(box_cx - img_cx, box_cy - img_cy)
    max_dist = math.hypot(img_cx, img_cy)
    centrality = max(0.0, 1.0 - (dist / max_dist)) if max_dist > 0 else 1.0

    # 3. Weighted composite score
    return (w_conf * conf) + (w_area * norm_area) + (w_pos * centrality)


def generate_top_k_crops(
    image_source: Union[str, Image.Image, np.ndarray],
    model: YOLO,
    k: int = 5,
    conf_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """Runs garment detection and returns top-K bounding boxes ranked by visual prominence."""
    results = model(image_source, conf=conf_threshold, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    # Get image dimensions directly from metadata without touching pixel arrays
    height, width = result.orig_shape

    candidates = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        label = result.names[int(box.cls[0])]

        # Clamp to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        clamped_box = [x1, y1, x2, y2]

        score = rank_prominence(conf, clamped_box, width, height)

        candidates.append({
            "label": label,
            "confidence": round(conf, 4),
            "prominence_score": round(score, 4),
            "box_xyxy": clamped_box,
            "box_normalized": [
                round(y1 / height, 4),
                round(x1 / width, 4),
                round(y2 / height, 4),
                round(x2 / width, 4),
            ],
        })

    # Sort descending by composite prominence score
    candidates.sort(key=lambda item: item["prominence_score"], reverse=True)
    return candidates[:k]