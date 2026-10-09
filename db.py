from pymongo import MongoClient
import requests

from dotenv import load_dotenv

load_dotenv()

MONGO_CONNECTION_STRING="mongodb://localhost:27017/?directConnection=true"

client = MongoClient(MONGO_CONNECTION_STRING)

db = client["curation-engine"]
collection = db["catalog"]

# from pymongo.operations import SearchIndexModel

# index = SearchIndexModel(
#     definition={
#         "fields": [
#             {"type": "vector", "path": "image_embedding", "numDimensions": 512, "similarity": "cosine"},
#             {"type": "vector", "path": "text_embedding", "numDimensions": 512, "similarity": "cosine"},
#         ]
#     },
#     name="catalog_vector_index",
#     type="vectorSearch",
# )

# collection.create_search_index(index)

# import time

# while True:
#     indexes = list(collection.list_search_indexes("catalog_vector_index"))
#     if indexes and indexes[0].get("queryable"):
#         break
#     time.sleep(2)
# print("Index is ready")

CLIP_URL = "http://localhost:8000"

# res = requests.post(f"{CLIP_URL}/embed/text", json={"text": "Superdry white hoodie black logo pullover sweatshirt"})
# text_vector = res.json()["vector"]

# with open("sweatshirt.png", "rb") as img:
#     res = requests.post(f"{CLIP_URL}/embed/image", files={"file": img})
# image_vector = res.json()["vector"]

# result = collection.insert_one(
#     {
#         "title" : "Superdry Logo Print Cotton Blend Hoodie",
#         "image_embedding" : image_vector,
#         "text_embedding": text_vector 
#     }
# )
# print(result.inserted_id)

def query_vector(collection, index_name, field, vector, limit=5):
    return list(collection.aggregate([
        {
            "$vectorSearch": {
                "index": index_name,
                "path": field,
                "queryVector": vector,
                "numCandidates": limit * 15,
                "limit": limit
            }
        },
        {"$project": {"score": {"$meta": "vectorSearchScore"}, "text_embedding": 0, "image_embedding": 0}}
    ]))



res = requests.post(f"{CLIP_URL}/embed/text", json={"text": "white hoodie"})
text_vector = res.json()["vector"]

results = query_vector(collection, "catalog_vector_index", "text_embedding", text_vector)
print(results)