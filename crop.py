from typing import Any, Union
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO


def load_clothing_model(
    repo_id: str = "kesimeg/yolov8n-clothing-detection",
    filename: str = "best.pt"
) -> YOLO:
    """Download (cached) and initialize the YOLO clothing model once."""
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(model_path)


def generate_top_k_crops(
    image_source: Union[str, Image.Image],
    model: YOLO,
    k: int = 3,
    conf_threshold: float = 0.25,
    show_crops: bool = False
) -> list[dict[str, Any]]:
    """Runs garment detection and extracts the top-K highest-confidence crops.

    Args:
        image_source: File path (str) or PIL Image object to detect.
        model: Preloaded YOLO model instance.
        k: Maximum number of highest-confidence crops to extract.
        conf_threshold: Minimum confidence score to consider.
        show_crops: If True, opens each cropped image in a viewer window.

    Returns:
        List of dicts containing:
            - 'crop': PIL.Image.Image (RGB)
            - 'label': str
            - 'confidence': float
            - 'box_xyxy': list[int] [xmin, ymin, xmax, ymax]
            - 'box_2d': list[int] [ymin, xmin, ymax, xmax] (0-1000 normalized)
    """
    # 1. Run inference
    results = model(image_source, conf=conf_threshold, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    # 2. Extract original BGR image and dimensions
    img_bgr = result.orig_img
    height, width = img_bgr.shape[:2]

    # 3. Sort boxes by confidence in descending order and slice top K
    sorted_boxes = sorted(
        result.boxes,
        key=lambda b: float(b.conf[0]),
        reverse=True
    )[:k]

    detected_crops = []

    # 4. Extract crops and metadata
    for i, box in enumerate(sorted_boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        label = result.names[int(box.cls[0])]

        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        # Slice BGR numpy array and convert to RGB PIL Image
        crop_bgr = img_bgr[y1:y2, x1:x2]
        crop_rgb = Image.fromarray(crop_bgr[..., ::-1])

        # Normalized coordinates for schema compatibility [ymin, xmin, ymax, xmax]
        box_2d_norm = [
            int((y1 / height) * 1000),
            int((x1 / width) * 1000),
            int((y2 / height) * 1000),
            int((x2 / width) * 1000),
        ]

        if show_crops:
            crop_rgb.show(title=f"Top_{i+1}_{label}_{confidence:.2f}")

        detected_crops.append({
            "crop": crop_rgb,
            "label": label,
            "confidence": round(confidence, 4),
            "box_xyxy": [x1, y1, x2, y2],
            "box_2d": box_2d_norm,
        })

    return detected_crops