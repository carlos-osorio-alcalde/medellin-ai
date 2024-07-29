import argparse

from rag import search_similar_items
from rag.clients import qdrant_manager
from rag.core.models import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search for similar audios or images"
    )
    parser.add_argument(
        "-v", "--value", type=str, help="Text query to search for"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Specify model: CLIP or CLAP",
    )

    args = parser.parse_args()
    search_results = search_similar_items(
        qdrant_manager=qdrant_manager,
        collection_name=qdrant_manager.model_collections[
            Model.CLAP if args.model.upper() == "CLAP" else Model.CLIP
        ],
        value=args.value,
    )
