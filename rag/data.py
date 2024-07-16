from typing import Dict, Optional, Union, Generator
import pandas as pd
import os
from configuration.load import config
from dataclasses import dataclass


@dataclass
class DataToUpsert:
    value: str
    payload: Optional[Dict[str, Union[str, int]]] = None


# Load the metadata
metadata = pd.read_csv(config["local"]["metadata"], sep=",")


def get_category_from_filename(filename: str) -> str:
    """Get the category from the filename."""
    try:
        return metadata[metadata["filename"] == filename]["category"].values[
            0
        ]
    except IndexError:
        return "unknown"


def create_audio_to_upsert() -> Generator:
    """Create a list of audio data to upsert."""
    # Get the audio in data/audio path
    files = [
        f
        for f in os.listdir(config["local"]["audios"])
        if f.endswith(".wav")
    ]

    for file in files:
        category = get_category_from_filename(file)

        yield DataToUpsert(
            value=os.path.join(config["local"]["audios"], file),
            payload={
                "type": "sound",
                "filename": file,
                "category": category,
            },
        )
