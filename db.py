import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from dotenv import load_dotenv
from pprint import pprint
from pathlib import Path

client = chromadb.PersistentClient(path="./chroma_db")
image_loader = ImageLoader()
multimodal_ef = OpenCLIPEmbeddingFunction()

collection = client.get_or_create_collection(
    name="product_catalog",
    embedding_function=multimodal_ef,
    data_loader=image_loader,
)

# with open("output.json", "r", encoding="utf-8") as f:
#     catalog_data = json.load(f)

# ids = []
# uris = []
# metadatas = []

# for filename, metadata in catalog_data.items():
#     image_path = f"inputs/{filename}"

#     if not os.path.exists(image_path):
#         print(f"Skipping {filename}: file not found at {image_path}")
#         continue

#     garments = metadata.get("garments", [])
#     if not garments:
#         continue

#     first_garment = garments[0]

#     desc = {
#         "title": first_garment.get("title", ""),
#         "category": first_garment.get("category", ""),
#         "brand": first_garment.get("brand") or "",  # Avoid None
#         "primary_color": first_garment.get("primary_color", ""),
#     }

#     ids.append(filename)
#     uris.append(image_path)
#     metadatas.append(desc)

# if ids:
#     collection.add(
#         ids=ids,
#         uris=uris,
#         metadatas=metadatas
#     )
#     print(f"Successfully added {len(ids)} items to ChromaDB.")


# results = collection.query(
#     query_texts=["sweatshirt"],
#     n_results=1
# ) 

# pprint(results)


for file in Path("query images").iterdir():
    results_img = collection.query(
        query_uris=[file],
        n_results=1
    )
    print(file.name)
    pprint(results_img)