from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = SentenceTransformer("clip-ViT-B-32")
    yield
    model = None

app = FastAPI(lifespan=lifespan)

class TextPayload(BaseModel):
    text: str

@app.post("/embed/text")
def embed_text(payload: TextPayload):
    vector = model.encode(payload.text).tolist()
    return {"vector": vector}

@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    vector = model.encode(image).tolist()
    return {"vector": vector}