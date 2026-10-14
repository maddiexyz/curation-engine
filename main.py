from dotenv import load_dotenv
load_dotenv()

from attribute import analyze_crop_features
from requests import HTTPError

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer
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
yolo_model: Optional[YOLO] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection, clip_model, yolo_model

    # 2. MongoDB collection
    db = get_db()
    collection = get_collection(db)

    # 3. Models
    clip_model = load_clip_model()
    yolo_model = load_clothing_model()

    yield

    # Teardown
    db.client.close()
    collection = None
    clip_model = None
    yolo_model = None


app = FastAPI(lifespan=lifespan)
