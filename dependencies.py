from fastapi import Body, Depends, HTTPException
from rag import vector_service
from rag.transform import embed
from schemas import TextModelRequest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_rag_content(body: TextModelRequest = Body(...)) -> str:
    """
    Retrieves relevant content from the vector database using RAG.

    Args:
        body (TextModelRequest): The request body containing the prompt.

    Returns:
        str: Concatenated relevant content from the vector database.
    """
    try:
        # Perform semantic search using the query vector
        query_vector = embed(body.prompt)
        rag_content = await vector_service.search(
            collection_name="knowledgebase",
            query_vector=query_vector,
            retrieval_limit=3,
            score_threshold=0.7,
        )

        # Concatenate original text from search results
        rag_content_str = "\n".join([c.payload["original_text"] for c in rag_content])
        logger.info("RAG content successfully retrieved")
        return rag_content_str

    except Exception as e:
        logger.error(f"Error retrieving RAG content: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve relevant content: {e}",
        )