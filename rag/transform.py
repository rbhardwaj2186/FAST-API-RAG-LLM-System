import re
import torch
from typing import Any, AsyncGenerator

import aiofiles
from transformers import AutoModel, AutoTokenizer

# Debug prints for GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clear GPU memory if any
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load the embedding model
embedder = AutoModel.from_pretrained(
    "distilbert-base-uncased",
    trust_remote_code=True,
    low_cpu_mem_usage=False,
    torch_dtype=torch.float16
)
embedder = embedder.to(device)
embedder.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased",
    trust_remote_code=True
)

model = AutoModel.from_pretrained("distilbert-base-uncased")



async def load(filepath: str, chunk_size: int = 20000) -> AsyncGenerator[str, Any]:
    """
    Load the contents of a file in chunks asynchronously.
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
    """
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\. ,", "", t)
    t = t.replace("..", ".")
    t = t.replace(". .", ".")
    return t.strip()


def embed(text: str) -> list[float]:
    """Generate embedding vector for text."""
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(torch.int64).to(device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = embedder(**inputs).last_hidden_state.mean(dim=1).squeeze()
            embeddings = embeddings.cpu().float()

        return embeddings.tolist()
    except Exception as e:
        raise RuntimeError(f"Error generating embedding: {e}")

def generate_query_vector(query: str):
    """
    Generates a vector embedding for a given query using DistilBERT.
    """
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state  # Token embeddings
        attention_mask = inputs['attention_mask']  # Attention mask

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        query_vector = (sum_embeddings / sum_mask).squeeze().tolist()

    return query_vector