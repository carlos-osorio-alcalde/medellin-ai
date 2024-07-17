from configuration.load import config
from rag import upsert_full_data
from rag.clients import qdrant_manager
from rag.core.data import (audio_payload_generator, create_data_to_upsert,
                           image_payload_generator)
from rag.core.models import MODEL_AUDIO_DIM, MODEL_IMAGE_DIM


def upsert_embeddings(media_info: dict):
    """Upsert embeddings to the vectordb."""
    for info in media_info:
        upsert_full_data(
            qdrant_manager=qdrant_manager,
            collection_name=info["collection_name"],
            model_dim=info["model_dim"],
            data_to_upsert=create_data_to_upsert(
                directory=info["directory"],
                file_extension=info["file_extension"],
                payload_generator=info["payload_generator"],
            ),
        )


if __name__ == "__main__":
    # Create and upload the audio embedding to qdrant
    media_info = [
        {
            "collection_name": config["vectordb"]["collection_audio"],
            "model_dim": MODEL_AUDIO_DIM,
            "directory": config["audio"]["directory"],
            "file_extension": config["audio"]["file_extension"],
            "payload_generator": audio_payload_generator,
        },
        {
            "collection_name": config["vectordb"]["collection_image"],
            "model_dim": MODEL_IMAGE_DIM,
            "directory": config["image"]["directory"],
            "file_extension": config["image"]["file_extension"],
            "payload_generator": image_payload_generator,
        },
    ]
    upsert_embeddings(media_info)
