from configuration.load import config
from rag.data import create_audio_to_upsert
from rag.embeddings import MODEL_DIM, create_embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rag.vectordb import QdrantWrapper


def upsert_full_audio_data():
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


def search_audio_by_text(text: str):
    """Search audio by text."""
    search_results = qdrant_manager.search_vectors(
        collection_name=config["vectordb"]["collection"], value=text
    )

    return search_results


if __name__ == "__main__":
    # Instantiate the Qdrant client
    client = QdrantClient(url=config["vectordb"]["client"])

    # Instantiate the QdrantManager
    qdrant_manager = QdrantWrapper(
        client=client, embedding_function=create_embeddings
    )

    # Upsert the full audio data
    # upsert_full_audio_data()

    # Get the text and search
    text = input("Enter the text to search: ")
    search_results = search_audio_by_text(text)
    print(search_results)
