from fastapi import FastAPI, Request, Body, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.responses import JSONResponse
from dependencies import get_rag_content
from schemas import TextModelRequest, TextModelResponse, SearchResponse, SearchResult, QueryRequest
from rag.repository import VectorRepository
from vllm_service import generate_text
from rag.extractor import pdf_text_extractor
from rag.service import vector_service
from rag.transform import generate_query_vector  # Import embedding logic
import logging
import requests
import asyncio
import os
import aiofiles

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

repo = VectorRepository(host="172.25.163.6", port=6333)

@app.post("/search", response_model=SearchResponse)
async def search(query_request: QueryRequest):
    try:
        # Generate query vector
        query_vector = generate_query_vector(query_request.query)

        # Perform search (use await here)
        search_results = await repo.search(
            collection_name="knowledgebase",
            query_vector=query_vector,
            retrieval_limit=query_request.top_k,
            score_threshold=0.1,  # Adjust the threshold as needed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Format results
    results = [
        SearchResult(
            original_text=result.payload.get("original_text", "N/A"),
            source=result.payload.get("source", "N/A"),
            score=result.score,
        )
        for result in search_results
    ]

    return SearchResponse(results=results)

@app.delete("/delete-collection/{collection_name}")
async def delete_collection_endpoint(collection_name: str):
    success = await repo.delete_collection(collection_name)
    if success:
        return {"message": f"Collection '{collection_name}' successfully deleted."}
    else:
        return {"error": f"Failed to delete collection '{collection_name}'."}

# Save uploaded files
async def save_file(file: UploadFile) -> str:
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)

    try:
        async with aiofiles.open(filepath, "wb") as f:
            while chunk := await file.read(1024 * 1024 * 50):  # 50 MB chunk size
                await f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Failed to save file: {e}")

    return filepath

@app.on_event("startup")
def check_qdrant_connection():
    try:
        # Using host.docker.internal for Qdrant in WSL
        response = requests.get("http://172.25.163.6:6333/")
        logger.info("Connected to Qdrant successfully!")
        logger.info(f"Qdrant response: {response.json()}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")

# Upload endpoint
@app.post("/upload")
async def file_upload_controller(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported for upload.",
        )
    try:
        # Save the uploaded file
        filepath = await save_file(file)

        # Extract text from the PDF and store embeddings in Qdrant
        background_tasks.add_task(pdf_text_extractor, filepath)
        background_tasks.add_task(
            vector_service.store_file_content_in_db,
            filepath.replace(".pdf", ".txt"),
            512,
            "knowledgebase",
            768,
        )

        return {"filename": file.filename, "message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {e}",
        )

@app.post("/generate/text", response_model=TextModelResponse)
async def serve_text_to_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    rag_content: str = Depends(get_rag_content),
) -> TextModelResponse:
    """
    Generates text using the LLM with RAG-augmented content.

    Args:
        request (Request): The HTTP request object.
        body (TextModelRequest): The request body containing the prompt and temperature.
        rag_content (str): Relevant content retrieved via RAG.

    Returns:
        TextModelResponse: Generated text and client IP.
    """
    try:
        # Construct augmented prompt
        prompt = f"{body.prompt}\n{rag_content}"
        logger.info("Augmented prompt successfully created")

        # Generate text using the LLM
        output = await generate_text(prompt, body.temperature)
        logger.info("Text successfully generated")

        return TextModelResponse(content=output, ip=request.client.host)

    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate text: {e}",
        )

@app.get("/collections")
async def list_collections():
    try:
        response = await repo.db_client.get_collections()
        collections = [collection.name for collection in response.collections]
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch collections.")


# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}