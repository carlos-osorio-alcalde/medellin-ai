from rag import create_collections
from rag.clients import qdrant_manager
from rag.core.models import MODEL_AUDIO_DIM, MODEL_IMAGE_DIM, Model

if __name__ == "__main__":
    # Create the collections for audio
    create_collections(
        qdrant_manager=qdrant_manager,
        model=Model.CLAP,
        model_dim=MODEL_AUDIO_DIM,
    )

    # Create the collections for image
    create_collections(
        qdrant_manager=qdrant_manager,
        model=Model.CLIP,
        model_dim=MODEL_IMAGE_DIM,
    )
