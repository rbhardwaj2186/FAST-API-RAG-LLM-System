from qdrant_client import QdrantClient
client = QdrantClient("http://172.25.6:6333")
response = client.count(collection_name="knowledgebase")
print(response)
