from qdrant_client import QdrantClient
from rag.core.vectordb import QdrantWrapper
from rag.core.embeddings import create_embeddings
from configuration.load import config

# Instantiate the Qdrant client
client = QdrantClient(url=config["vectordb"]["client"])

# Instantiate the QdrantManager
qdrant_manager = QdrantWrapper(
    client=client, embedding_function=create_embeddings
)
