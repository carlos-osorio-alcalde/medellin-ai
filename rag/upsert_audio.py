from rag import upsert_full_audio_data
from rag.clients import qdrant_manager

if __name__ == "__main__":
    upsert_full_audio_data(qdrant_manager=qdrant_manager)
