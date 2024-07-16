from rag import search_similar_audios
from rag.clients import qdrant_manager
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for similar audios")
    parser.add_argument("value", type=str, help="Text to search for")

    args = parser.parse_args()
    search_results = search_similar_audios(qdrant_manager, args.value)
    print(search_results)
