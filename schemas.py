from pydantic import BaseModel

class TextModelRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

class TextModelResponse(BaseModel):
    content: str
    ip: str

# Define the request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Define the structure of individual search results
class SearchResult(BaseModel):
    original_text: str
    source: str
    score: float

# Define the response model
class SearchResponse(BaseModel):
    results: list[SearchResult]