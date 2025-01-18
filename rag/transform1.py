import re
from typing import Any, AsyncGenerator

import aiofiles
from transformers import AutoModel, AutoTokenizer

# Load the embedding model
embedder = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
)


async def load(filepath: str, chunk_size: int = 20000) -> AsyncGenerator[str, Any]:
    """
    Load the contents of a file in chunks asynchronously.

    Args:
        filepath (str): Path to the file to load.
        chunk_size (int): Size of each chunk in bytes.

    Yields:
        str: Text chunks from the file.
    """
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {filepath}: {e}")


def clean(text: str) -> str:
    """
    Clean and normalize text by removing unnecessary whitespace and artifacts.

    Args:
        text (str): Input text to clean.

    Returns:
        str: Cleaned text.
    """
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)  # Replace multiple spaces with a single space
    t = re.sub(r"\. ,", "", t)
    t = t.replace("..", ".")
    t = t.replace(". .", ".")
    cleaned_text = t.strip()
    return cleaned_text


def embed(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text.

    Args:
        text (str): Text to embed.

    Returns:
        list[float]: List of float values representing the text embedding.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            embeddings = embedder(**inputs).last_hidden_state.mean(dim=1).squeeze()
        return embeddings.tolist()
    except Exception as e:
        raise RuntimeError(f"Error generating embedding: {e}")
