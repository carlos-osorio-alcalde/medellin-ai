import os
import uuid
from typing import Callable, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.core.data import DataToUpsert
from rag.core.embeddings import CLAPModality
from rag.core.logger import logger


class QdrantWrapper:
    def __init__(self, client: QdrantClient, embedding_function: Callable):
        self._client = client
        self._embeddings_function = embedding_function

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection of vectors in Qdrant."""
        self._client.delete_collection(collection_name=collection_name)

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Dict[str, models.VectorParams],
    ) -> None:
        """Create a collection of vectors in Qdrant."""
        self._client.create_collection(
            collection_name=collection_name, vectors_config=vectors_config
        )

    def check_collection(self, collection_name: str) -> bool:
        """Check if a collection exists in Qdrant."""
        return self._client.collection_exists(
            collection_name=collection_name
        )

    def _compute_embeddings(self, value: str) -> List[float]:
        """Compute embeddings for the given value."""
        if os.path.exists(value):
            embedding = self._embeddings_function(
                modality=CLAPModality.AUDIO, audio_paths=[value]
            )
            embedding_type = "audio"
        elif isinstance(value, str) and not os.path.exists(value):
            embedding = self._embeddings_function(
                modality=CLAPModality.TEXT, texts_list=[value]
            )
            embedding_type = "text"

        logger.info(f"Computing {embedding_type} embeddings for {value}")

        return embedding

    def _create_point_for_qdrant(
        self, data: DataToUpsert
    ) -> models.PointStruct:
        """Create a list of points for upserting in Qdrant."""
        logger.info(f"Creating embeddings for {data.value}")
        embedding = self._compute_embeddings(data.value)

        return models.PointStruct(
            id=str(uuid.uuid4()),
            payload=data.payload,
            vector=embedding,
        )

    def upsert_vectors(
        self, collection_name: str, data_to_upsert: List[DataToUpsert]
    ) -> None:
        """Upsert vectors to a collection in Qdrant."""
        for data in data_to_upsert:
            point = self._create_point_for_qdrant(data)
            self._client.upsert(
                collection_name=collection_name, points=[point]
            )

    def search_vectors(self, value: str, collection_name: str, top: int = 5):
        """Search for vectors in Qdrant."""
        embedding = self._compute_embeddings(value)
        return self._client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top,
        )
