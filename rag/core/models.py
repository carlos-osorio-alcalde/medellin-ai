import os
from enum import Enum
from typing import List, Optional, Union

import torch
from dotenv import load_dotenv
from groq import Groq
from msclap import CLAP
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from configuration.load import config

# Instantiate the model for audio
model_audio = CLAP()
MODEL_AUDIO_DIM = model_audio.args.d_proj

# Instantiate the model for images
model_path = config["huggingface"]["image_text_model"]
model_image = AutoModel.from_pretrained(model_path)
processor_image = AutoImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
MODEL_IMAGE_DIM = model_image.projection_dim

# Instantiate the LLM client
load_dotenv()
llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class Model(Enum):
    """
    Enum for specifying the model to be used for
    creating embeddings.
    """

    CLIP = "CLIP"
    CLAP = "CLAP"


class Modalities(Enum):
    """
    Enum for specifying the modality of the data
    for which embeddings are required.
    """

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


def create_embeddings_audio(
    modality: Modalities,
    audio_paths: Optional[Union[str, List[str]]] = None,
    texts_list: Optional[Union[str, List[str]]] = None,
) -> List[float]:
    """Create embeddings for audio or text data based on the given modality."""
    if modality == Modalities.TEXT and texts_list is not None:
        embedding = model_audio.get_text_embeddings(texts_list)

    elif modality == Modalities.AUDIO and audio_paths is not None:
        embedding = model_audio.get_audio_embeddings(audio_paths)

    else:
        raise ValueError(
            """Invalid modality or missing required arguments
            for the specified modality."""
        )

    return embedding[0]


def create_embeddings_image(
    modality: Modalities,
    image_paths: Optional[Union[str, List[str]]] = None,
    texts_list: Optional[Union[str, List[str]]] = None,
) -> List[float]:
    """Create embeddings for image or text data based on the given modality."""
    if modality == Modalities.TEXT and texts_list is not None:
        inputs = tokenizer(
            texts_list, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            embedding = model_image.get_text_features(**inputs)

        return embedding.tolist()[0]

    elif modality == Modalities.IMAGE and image_paths is not None:
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = [Image.open(image_path) for image_path in image_paths]
        inputs = processor_image(images=images, return_tensors="pt")
        with torch.no_grad():
            embedding = model_image.get_image_features(**inputs)

        return embedding.tolist()[0]

    else:
        raise ValueError(
            """Invalid modality or missing required arguments
            for the specified modality."""
        )


def create_caption_audio(audio_path: Union[str, List[str]]) -> str:
    """Create caption for the given audio."""
    model_caption = CLAP(version="clapcap")
    return model_caption.generate_caption(audio_path)


if __name__ == "__main__":
    path = "data/images/image_0.png"
    text_embeddings = create_embeddings_image(
        modality=Modalities.TEXT, texts_list="Hi everyone"
    )
    image_embeddings = create_embeddings_image(
        modality=Modalities.IMAGE, image_paths=path
    )

    print("Size: ", len(text_embeddings), len(image_embeddings))
