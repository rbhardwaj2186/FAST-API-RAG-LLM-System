import os
import aiofiles
from fastapi import UploadFile

# Define the default chunk size (50MB)
DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50  # 50 megabytes

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
