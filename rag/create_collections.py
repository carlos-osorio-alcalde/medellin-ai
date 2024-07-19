import argparse

from rag import create_collections
from rag.clients import qdrant_manager
from rag.core.models import MODEL_AUDIO_DIM, MODEL_IMAGE_DIM, Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create collections")
    parser.add_argument(
        "-c",
        "--collection",
        type=str,
        help="Select the collection to create: audio or image",
        choices=["audio", "image", "both"],
    )

    args = parser.parse_args()

    # Create the collections for audio
    if args.collection == "audio" or args.collection == "both":
        create_collections(
            qdrant_manager=qdrant_manager,
            model=Model.CLAP,
            model_dim=MODEL_AUDIO_DIM,
        )

    if args.collection == "image" or args.collection == "both":
        create_collections(
            qdrant_manager=qdrant_manager,
            model=Model.CLIP,
            model_dim=MODEL_IMAGE_DIM,
        )
