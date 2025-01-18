from qdrant_client import QdrantClient
import numpy as np

# Initialize the client
client = QdrantClient(url="http://localhost:6333")  # Replace with your Qdrant host if different

# Verify collections
collections = client.get_collections()
print("Available collections:", collections.collections)

# Retrieve collection info
collection_info = client.get_collection(collection_name="knowledgebase")
print("Collection Info:", collection_info)

# Generate a dummy query vector (Replace with actual embedding from your model)
query_vector = np.random.rand(768).tolist()  # Ensure this matches your vector dimension (768)

# Perform vector search (Corrected 'limit' argument)
search_results = client.search(
    collection_name="knowledgebase",
    query_vector=query_vector,
    limit=5  # Retrieve top 5 results
)

# Display results
for result in search_results:
    print(f"Match: {result.payload}, Score: {result.score}")
