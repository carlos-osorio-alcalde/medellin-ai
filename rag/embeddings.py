from enum import Enum
from typing import List, Optional, Union

from msclap import CLAP

# Instantiate the model
model = CLAP()
MODEL_DIM = model.args.d_proj


class CLAPModality(Enum):
    """Enum for specifying the modality of the data for which embeddings"""

    TEXT = "text"
    AUDIO = "audio"


def create_embeddings(
    modality: CLAPModality,
    audio_paths: Optional[Union[str, List[str]]] = None,
    texts_list: Optional[Union[str, List[str]]] = None,
) -> List[float]:
    """Create embeddings for audio or text data based on the given modality."""
    if modality == CLAPModality.TEXT and texts_list is not None:
        embedding = model.get_text_embeddings(texts_list)

    elif modality == CLAPModality.AUDIO and audio_paths is not None:
        embedding = model.get_audio_embeddings(audio_paths)

    else:
        raise ValueError(
            """Invalid modality or missing required arguments
            for the specified modality."""
        )

    return embedding[0]


if __name__ == "__main__":
    # Create embeddings for text data
    texts = ["Hello, how are you?", "I am doing great!"]
    embeddings = create_embeddings(
        texts_list=texts, modality=CLAPModality.TEXT
    )
    print("text:", embeddings)

    # Create embeddings for audio data
    audio_files = [
        "data/audio/1-7057-A-12.wav",
        "data/audio/3-151089-A-30.wav",
    ]
    embeddings = create_embeddings(
        audio_paths=audio_files, modality=CLAPModality.AUDIO
    )
    print("sound:", embeddings)
