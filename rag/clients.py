from qdrant_client import QdrantClient

from configuration.load import config
from rag.core.models import create_embeddings_audio, create_embeddings_image
from rag.core.vectordb import QdrantWrapper

# Instantiate the Qdrant client
client = QdrantClient(url=config["vectordb"]["client"])

# Instantiate the QdrantManager
qdrant_manager = QdrantWrapper(
    client=client,
    embedding_function_audios=create_embeddings_audio,
    embedding_function_images=create_embeddings_image,
)
