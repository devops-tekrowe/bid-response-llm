

# This will remove all chunks in that collection.
# This is useful if you want to start fresh or if you want to ingest new PDFs.
# You can recreate the collection later when you want to ingest new PDFs.

from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
client.delete_collection(collection_name="rag_docs")
