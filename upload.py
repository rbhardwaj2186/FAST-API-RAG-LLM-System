import os
import aiofiles
from fastapi import UploadFile
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
import numpy as np

# Define the default chunk size (50MB)
DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50  # 50 megabytes

# Qdrant setup
QDRANT_URL = "http://172.25.163.6:6333"
COLLECTION_NAME = "knowledgebase"
VECTOR_SIZE = 768  # Update this based on your embedding dimensionality
DISTANCE_METRIC = "Cosine"

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL)

# Ensure the collection exists in Qdrant
if not client.get_collection(COLLECTION_NAME):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC),
    )

async def save_file(file: UploadFile) -> str:
    """
    Save the uploaded file to the uploads directory in chunks.

    Args:
        file (UploadFile): The uploaded file to save.

    Returns:
        str: The full path to the saved file.
    """
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)

    try:
        async with aiofiles.open(filepath, "wb") as f:
            while chunk := await file.read(DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Failed to save file: {e}")

    return filepath

async def process_and_upload_to_qdrant(filepath: str):
    """
    Process the saved file and upload its data to Qdrant.

    Args:
        filepath (str): The path to the saved file.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Generate a dummy embedding (replace with actual embedding logic)
        embedding = np.random.rand(VECTOR_SIZE).tolist()

        # Upload the data to the Qdrant collection
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": os.path.basename(filepath),  # Use the filename as a unique ID
                    "vector": embedding,
                    "payload": {"content": content},
                }
            ],
        )
        print(f"Uploaded {filepath} to Qdrant collection '{COLLECTION_NAME}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to process and upload file to Qdrant: {e}")
