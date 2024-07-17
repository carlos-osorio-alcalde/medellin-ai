import os
import uuid
from logging import Logger
from typing import Callable, Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.http import models

from configuration.load import config
from rag.core.data import DataToUpsert
from rag.core.logger import logger
from rag.core.models import Modalities, Model


class QdrantWrapper:
    # Name of the collection based on the model
    collection_models = {
        config["vectordb"]["collection_audio"]: Model.CLAP,
        config["vectordb"]["collection_image"]: Model.CLIP,
    }

    # Name of the model based on the collection
    model_collections = {
        Model.CLAP: config["vectordb"]["collection_audio"],
        Model.CLIP: config["vectordb"]["collection_image"],
    }

    def __init__(
        self,
        client: QdrantClient,
        embedding_function_audios: Callable,
        embedding_function_images: Callable,
        logger: Logger = logger,
    ):
        self._client = client
        self._embeddings_function_audio = embedding_function_audios
        self._embeddings_function_image = embedding_function_images
        self._logger = logger

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
        self._logger.info(f"Created collection {collection_name} in Qdrant")

    def check_collection(self, collection_name: str) -> bool:
        """Check if a collection exists in Qdrant."""
        return self._client.collection_exists(
            collection_name=collection_name
        )

    def _compute_embeddings(self, model: str, value: str) -> List[float]:
        """Compute embeddings for the given value."""
        embedding, embedding_type = None, None

        if model == Model.CLAP:
            embedding, embedding_type = self._compute_clap_embeddings(value)
        elif model == Model.CLIP:
            embedding, embedding_type = self._compute_clip_embeddings(value)
        else:
            raise ValueError(f"Unsupported model type: {model}")

        self._logger.info(
            f"Computing {embedding_type} embeddings for {value}"
        )
        return embedding

    def _compute_clap_embeddings(
        self, value: str
    ) -> Tuple[List[float], str]:
        if os.path.exists(value):
            return (
                self._embeddings_function_audio(
                    modality=Modalities.AUDIO, audio_paths=[value]
                ),
                "audio",
            )
        elif isinstance(value, str):
            return (
                self._embeddings_function_audio(
                    modality=Modalities.TEXT, texts_list=[value]
                ),
                "text",
            )

    def _compute_clip_embeddings(
        self, value: str
    ) -> Tuple[List[float], str]:
        if os.path.exists(value):
            return (
                self._embeddings_function_image(
                    modality=Modalities.IMAGE, image_paths=[value]
                ),
                "image",
            )
        elif isinstance(value, str):
            return (
                self._embeddings_function_image(
                    modality=Modalities.TEXT, texts_list=[value]
                ),
                "text",
            )

    def _create_point_for_qdrant(
        self, model: Model, data: DataToUpsert
    ) -> models.PointStruct:
        """Create a list of points for upserting in Qdrant."""
        self._logger.info(f"Creating embeddings for {data.value}")
        embedding = self._compute_embeddings(model=model, value=data.value)

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
            point = self._create_point_for_qdrant(
                model=QdrantWrapper.collection_models[collection_name],
                data=data,
            )
            self._client.upsert(
                collection_name=collection_name, points=[point]
            )
            self._logger.info(
                f"Upserted data to Qdrant: {data.payload['filename']}"
            )

    def search_vectors(
        self, collection_name: str, value: str, top: int = 5
    ) -> List[types.ScoredPoint]:
        """Search for vectors in Qdrant."""
        embedding = self._compute_embeddings(
            model=QdrantWrapper.collection_models[collection_name],
            value=value,
        )
        return self._client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top,
        )
