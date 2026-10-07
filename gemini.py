import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from models import OutfitAnalysis

load_dotenv()

def generateImageMetadata(img: Image.Image) -> OutfitAnalysis:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=img,
        config=types.GenerateContentConfig(
            system_instruction = """
            You are a fashion cataloger and visual search indexer.
            Analyze every clothing item, pair of shoes, or accessory worn in the image:
            1. TITLES & QUERIES: Synthesize a specific e-commerce title and an actionable 4-7 keyword search query for each item
            (format: [Brand] [Subject/Motif] [Key Feature] [Item Type]).
            2. GRAPHICS & SUBJECTS: If an item features artwork, licensed characters, bands, sports emblems, or prominent typography,
            capture the specific subject in `subject_or_graphic`. If the item is plain or non-graphic, set this field to null.
            3. BRANDS: Only extract the brand if visible logos, marks, or unmistakable designs confirm it. Otherwise, set brand to
            null.
            4. TEXTURES & CUTS: Differentiate fabrics (denim, knit, leather, satin, fleece) and silhouettes (oversized, cropped, slim,
            wide-leg).""",
            response_mime_type="application/json",
            response_schema=OutfitAnalysis,
            temperature=0.1,
        ),
    )   
    return OutfitAnalysis.model_validate_json(response.text)

from pathlib import Path
folder = Path("inputs/")

metadatas = {}

for file in folder.iterdir():
    if file.is_file():
        img = Image.open(file)
        metadata = generateImageMetadata(img)
        metadatas[file.name] = metadata.model_dump(mode="json")

print(metadatas)

with open("output.json", "w") as json_file:
    json.dump(metadatas, json_file, indent=4)