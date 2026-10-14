import os
import math
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
import torch
import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

from crop import generate_top_k_crops, load_clothing_model

# ---------------------------------------------------------
# 1. Configuration & Qdrant Connection
# ---------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "fashion_catalog" # Updated to new collection name
IMAGE_PATH = os.path.join("images", "outfit.jpg")

client = QdrantClient(url=QDRANT_URL)

# ---------------------------------------------------------
# 2. Ensure Keyword Index Exists on 'category'
# ---------------------------------------------------------
# Qdrant requires a keyword payload index to perform group_by operations
try:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category", # Updated to index on the new category field
        field_schema=PayloadSchemaType.KEYWORD
    )
except Exception:
    pass  # Index already exists

# ---------------------------------------------------------
# 3. Load Model & Preprocessor
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
model = model.to(device).eval()

# Extract SigLIP's internal temperature and bias for confidence calculation
logit_scale = model.logit_scale.exp().item()
logit_bias = model.logit_bias.item()

# ---------------------------------------------------------
# 4. Encode Image
# ---------------------------------------------------------
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

IMG = Image.open(IMAGE_PATH).convert("RGB")
yolo_model = load_clothing_model()
crops = generate_top_k_crops(IMG, yolo_model, 3,conf_threshold=0.4)
img_w, img_h = IMG.size

for i,crop_data in enumerate(crops):
    x1, y1, x2, y2 = crop_data["box_xyxy"]

    # Tightened padding to 5% to reduce ViT background noise
    pad_w = int((x2 - x1) * 0.05)
    pad_h = int((y2 - y1) * 0.05)

    crop_box = (
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(img_w, x2 + pad_w),
        min(img_h, y2 + pad_h),
    )

    crop = IMG.crop(crop_box)
    crop.save(f"{i}.jpg")
    image = preprocess_val(crop).unsqueeze(0).to(device)
    

    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                image_features = model.encode_image(image, normalize=True)
        else:
            image_features = model.encode_image(image, normalize=True)
            
        query_vector = image_features.squeeze(0).cpu().to(torch.float32).numpy().tolist()

    # ---------------------------------------------------------
    # 5. Query: Top-3 Overall Title Matches
    # ---------------------------------------------------------
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3  # Retrieve the top 3 closest matches overall
    )

    # ---------------------------------------------------------
    # 6. Display Structured Metadata
    # ---------------------------------------------------------
    print(f"=== Extracted Matches for Crop ===")
    print(f"{'Category':<15} | {'Title Match':<55} | {'Raw Score':<9}")
    print("-" * 100)

    extracted_attributes = []

    for hit in search_results.points:
        title = hit.payload.get("title")
        category_name = hit.payload.get("category", "N/A")
        raw_score = hit.score
        
        extracted_attributes.append({
            "category": category_name, 
            "title": title, 
        })
        
        print(f"{category_name:<15} | {title:<55} | {raw_score:.4f}")

    print("\n\n")

client.close()