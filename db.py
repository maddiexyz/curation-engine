import os
import time
from typing import Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import SearchIndexModel


# ---------------------------------------------------------------------
# 1. Get DB Instance
# ---------------------------------------------------------------------
def get_db(
    connection_string: Optional[str] = None,
    db_name: str = "curation-engine",
) -> Database:
    """Connects to MongoDB and returns a Database instance."""
    uri = connection_string or os.getenv(
        "MONGO_CONNECTION_STRING",
        "mongodb://localhost:27017/?directConnection=true",
    )
    client: MongoClient = MongoClient(uri)
    return client[db_name]


# ---------------------------------------------------------------------
# 2. Get Collection
# ---------------------------------------------------------------------
def get_collection(
    db: Database,
    collection_name: str = "catalog",
) -> Collection:
    """Returns a specific collection from the database instance."""
    return db[collection_name]


# ---------------------------------------------------------------------
# 3. Create Index
# ---------------------------------------------------------------------
def create_vector_index(
    collection: Collection,
    index_name: str = "catalog_vector_index",
    dimensions: int = 512,
    similarity: str = "cosine",
) -> None:
    """Submits a vectorSearch index definition for text and image embeddings."""
    existing_indexes = [
        idx.get("name")
        for idx in collection.list_search_indexes()
        if "name" in idx
    ]
    if index_name in existing_indexes:
        return

    index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "image_embedding",
                    "numDimensions": dimensions,
                    "similarity": similarity,
                },
                {
                    "type": "vector",
                    "path": "text_embedding",
                    "numDimensions": dimensions,
                    "similarity": similarity,
                },
            ]
        },
        name=index_name,
        type="vectorSearch",
    )
    collection.create_search_index(index_model)


# ---------------------------------------------------------------------
# 4. Check Index Ready
# ---------------------------------------------------------------------
def is_index_ready(
    collection: Collection,
    index_name: str = "catalog_vector_index",
) -> bool:
    """Checks if the search index exists and is currently queryable."""
    indexes = list(collection.list_search_indexes(index_name))
    if indexes and indexes[0].get("queryable"):
        return True
    return False


def wait_for_index_ready(
    collection: Collection,
    index_name: str = "catalog_vector_index",
    timeout: int = 60,
    poll_interval: int = 2,
) -> bool:
    """Polls until the vector search index is queryable or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_index_ready(collection, index_name):
            return True
        time.sleep(poll_interval)
    return False


# ---------------------------------------------------------------------
# 5. Insert Item
# ---------------------------------------------------------------------
def insert_item(
    collection: Collection,
    item_document: dict[str, Any],
) -> str:
    """Inserts an item into the collection and returns the inserted string ID."""
    result = collection.insert_one(item_document)
    return str(result.inserted_id)


# ---------------------------------------------------------------------
# 6. Query
# ---------------------------------------------------------------------
def query_vector(
    collection: Collection,
    vector: list[float],
    field: str = "text_embedding",
    index_name: str = "catalog_vector_index",
    limit: int = 5,
    filter_criteria: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Executes a $vectorSearch aggregation pipeline against the collection."""
    search_stage: dict[str, Any] = {
        "index": index_name,
        "path": field,
        "queryVector": vector,
        "numCandidates": limit * 15,
        "limit": limit,
    }

    if filter_criteria:
        search_stage["filter"] = filter_criteria

    pipeline = [
        {"$vectorSearch": search_stage},
        {
            "$project": {
                "score": {"$meta": "vectorSearchScore"},
                "text_embedding": 0,
                "image_embedding": 0,
            }
        },
    ]

    return list(collection.aggregate(pipeline))