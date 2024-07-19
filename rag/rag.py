from typing import Generator, List

from qdrant_client.conversions import common_types as types
from qdrant_client.http import models

from rag.core.data import DataToUpsert
from rag.core.models import Model
from rag.core.vectordb import QdrantWrapper


def create_collections(
    qdrant_manager: QdrantWrapper, model: Model, model_dim: int
):
    """Create collections in the vectordb."""
    collection_name = QdrantWrapper.model_collections[model]
    if not qdrant_manager.check_collection(collection_name):
        qdrant_manager.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=model_dim,
                distance=models.Distance.COSINE,
            ),
        )


def upsert_full_data(
    qdrant_manager: QdrantWrapper,
    collection_name: str,
    model_dim: int,
    data_to_upsert: Generator[DataToUpsert, None, None],
):
    """Upsert all data to the specified collection."""
    # Check if the collection exists
    if not qdrant_manager.check_collection(collection_name):
        qdrant_manager.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=model_dim, distance=models.Distance.COSINE
            ),
        )

    # Upsert vectors to the collection
    qdrant_manager.upsert_vectors(
        collection_name=collection_name,
        data_to_upsert=data_to_upsert,
    )


def search_similar_items(
    qdrant_manager: QdrantWrapper,
    collection_name: str,
    value: str,
) -> List[types.ScoredPoint]:
    """Search audio by text."""
    search_results = qdrant_manager.search_vectors(
        collection_name=collection_name, value=value
    )

    return search_results
