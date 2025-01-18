import os
from loguru import logger
from .repository import VectorRepository
from .transform import clean, embed, load


class VectorService(VectorRepository):
    """
    Service layer extending the repository for processing and storing content in the database.
    """

    def __init__(self):
        """
        Initialize the VectorService with the repository's client setup.
        """
        super().__init__()

    async def store_file_content_in_db(
            self,
            filepath: str,
            chunk_size: int = 512,
            collection_name: str = "knowledgebase",
            collection_size: int = 768,
    ) -> None:
        try:
            # Ensure the collection exists or create it
            await self.create_collection(collection_name, collection_size)
            logger.info(f"Storing content from {filepath} into collection '{collection_name}'")

            # Process the file in chunks
            async for chunk in load(filepath, chunk_size):
                logger.debug(f"Processing chunk: '{chunk[:20]}...'")

                # Clean and embed the chunk
                embedding_vector = embed(clean(chunk))
                filename = os.path.basename(filepath)

                # Store the vector in the collection
                await self.create(collection_name, embedding_vector, chunk, filename)

            logger.info(f"File content from {filepath} successfully stored in the database.")

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise RuntimeError(f"File not found: {filepath}")

        except Exception as e:
            logger.error(f"An error occurred while processing {filepath}: {e}")
            raise RuntimeError(f"Failed to store content from {filepath}: {e}")


# Singleton instance of VectorService for use across the application
vector_service = VectorService()
