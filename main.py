from dotenv import load_dotenv
# 1. Environment variables (must load before DB connection reads MONGO_CONNECTION_STRING)
load_dotenv()

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from ultralytics import YOLO
from PIL import Image

from attribute import getAutoProcessor, loadQwen2bQuantized
from crop import load_clothing_model, generate_top_k_crops
from db import get_collection, get_db
from clip import load_clip_model


# ---------------------------------------------------------------------
# Global resources (populated during lifespan startup)
# ---------------------------------------------------------------------
collection: Optional[Collection] = None
clip_model: Optional[SentenceTransformer] = None
qwen_model: Optional[Qwen2VLForConditionalGeneration] = None
qwen_processor: Optional[AutoProcessor] = None
yolo_model: Optional[YOLO] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection, clip_model, qwen_model, qwen_processor, yolo_model

    # 2. MongoDB collection
    db = get_db()
    collection = get_collection(db)

    # 3. Models
    clip_model = load_clip_model()
    qwen_model = loadQwen2bQuantized()
    qwen_processor = getAutoProcessor()
    yolo_model = load_clothing_model()

    yield

    # Teardown
    db.client.close()
    collection = None
    clip_model = None
    qwen_model = None
    qwen_processor = None
    yolo_model = None


app = FastAPI(lifespan=lifespan)

@app.get("/")
def greeting():
    return "Hi thanks for visiting the base endpoint of curation engine"

@app.post("/crop")
def get_crops(file: UploadFile = File(...)):
    img = Image.open(file.file)
    crops = generate_top_k_crops(img,model=yolo_model)
    return [
        {
            "label": str(c["label"]),
            "confidence": round(float(c["confidence"]), 4),
            "crop_location": [int(v) for v in c["box_xyxy"]],
        }
        for c in crops
    ]