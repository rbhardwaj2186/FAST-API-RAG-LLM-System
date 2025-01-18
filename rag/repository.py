import requests
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import ScoredPoint


class VectorRepository:
    """
    Repository layer for interacting with the Qdrant vector database.
    """

    def __init__(self, host: str = "172.25.163.6", port: int = 6333) -> None:
        """
        Initialize the Qdrant client.

        Args:
            host (str): Hostname of the Qdrant server.
            port (int): Port of the Qdrant server.
        """
        self.db_client = AsyncQdrantClient(host=host, port=port)

    async def create_collection(self, collection_name: str, size: int) -> bool:
        """
        Create or recreate a collection in Qdrant.

        Args:
            collection_name (str): Name of the collection.
            size (int): Dimension of the embedding vectors.

        Returns:
            bool: True if the collection was created successfully, False otherwise.
        """
        vectors_config = models.VectorParams(size=size, distance=models.Distance.COSINE)

        try:
            response = await self.db_client.get_collections()
            collection_exists = any(
                collection.name == collection_name for collection in response.collections
            )

            if collection_exists:
                logger.debug(f"Collection {collection_name} already exists - recreating it")
                await self.db_client.delete_collection(collection_name)

            logger.debug(f"Creating collection {collection_name}")
            await self.db_client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False

    async def delete_collection(self, name: str) -> bool:
        """
        Delete a collection from Qdrant.

        Args:
            name (str): Name of the collection to delete.

        Returns:
            bool: True if the collection was deleted successfully, False otherwise.
        """
        try:
            logger.debug(f"Deleting collection {name}")
            await self.db_client.delete_collection(name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            return False

    async def create(
        self,
        collection_name: str,
        embedding_vector: list[float],
        original_text: str,
        source: str,
    ) -> None:
        """
        Add a new vector to a collection in Qdrant.

        Args:
            collection_name (str): Name of the collection.
            embedding_vector (list[float]): Embedding vector for the data.
            original_text (str): Original text associated with the vector.
            source (str): Source metadata for the vector.

        Returns:
            None
        """
        try:
            response = await self.db_client.count(collection_name=collection_name)
            logger.debug(
                f"Creating a new vector with ID {response.count} inside the {collection_name}"
            )

            await self.db_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=response.count,
                        vector=embedding_vector,
                        payload={
                            "source": source,
                            "original_text": original_text,
                        },
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Failed to create vector in {collection_name}: {e}")
            raise RuntimeError(f"Error creating vector in {collection_name}: {e}")

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        retrieval_limit: int,
        score_threshold: float,
    ) -> list[ScoredPoint]:
        """
        Perform a semantic search in Qdrant.

        Args:
            collection_name (str): Name of the collection to search in.
            query_vector (list[float]): Query embedding vector.
            retrieval_limit (int): Maximum number of results to return.
            score_threshold (float): Minimum score threshold for results.

        Returns:
            list[ScoredPoint]: List of scored points matching the query.
        """
        try:
            logger.debug(
                f"Searching for relevant items in the {collection_name} collection"
            )
            vectors = await self.db_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=retrieval_limit,
                score_threshold=score_threshold,
            )
            logger.debug(f"Search results: {vectors}")
            return vectors
        except Exception as e:
            logger.error(f"Failed to search in {collection_name}: {e}")
            raise RuntimeError(f"Error searching in {collection_name}: {e}")

    async def list_collections(self) -> list[str]:
        """
        List all collections in the Qdrant vector database.

        Returns:
            list[str]: List of collection names.
        """
        try:
            response = await self.db_client.get_collections()
            collections = [collection.name for collection in response.collections]
            logger.debug(f"Collections retrieved: {collections}")
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise RuntimeError(f"Error listing collections: {e}")