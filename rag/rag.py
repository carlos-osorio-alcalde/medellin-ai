from qdrant_client.http import models

from configuration.load import config
from rag.core.data import create_audio_to_upsert
from rag.core.embeddings import MODEL_DIM
from rag.core.vectordb import QdrantWrapper


def upsert_full_audio_data(qdrant_manager: QdrantWrapper):
    """Upsert all audio data to the collection."""
    # Check if the collection exists
    if not qdrant_manager.check_collection(config["vectordb"]["collection"]):
        qdrant_manager.create_collection(
            collection_name=config["vectordb"]["collection"],
            vectors_config=models.VectorParams(
                size=MODEL_DIM, distance=models.Distance.COSINE
            ),
        )

    # Upsert vectors to the collection
    data_to_upsert = create_audio_to_upsert()
    qdrant_manager.upsert_vectors(
        collection_name=config["vectordb"]["collection"],
        data_to_upsert=data_to_upsert,
    )


def search_similar_audios(qdrant_manager: QdrantWrapper, value: str):
    """Search audio by text."""
    search_results = qdrant_manager.search_vectors(
        collection_name=config["vectordb"]["collection"], value=value
    )

    return search_results
