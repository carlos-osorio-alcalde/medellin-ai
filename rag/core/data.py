import os
from dataclasses import dataclass
from typing import Callable, Dict, Generator, Optional, Union

import pandas as pd

from configuration.load import config


@dataclass
class DataToUpsert:
    value: str
    payload: Optional[Dict[str, Union[str, int]]] = None


# Load the metadata
metadata_audios = pd.read_csv(config["local"]["metadata_audios"], sep=",")


def get_category_from_filename(filename: str) -> str:
    """Get the category from the filename."""
    try:
        return metadata_audios[metadata_audios["filename"] == filename][
            "category"
        ].values[0]
    except IndexError:
        return "unknown"


def create_data_to_upsert(
    directory: str,
    file_extension: str,
    payload_generator: Callable[[str], Dict[str, str]],
) -> Generator[DataToUpsert, None, None]:
    """Create a list of data to upsert."""
    # Get the files in the specified directory
    files = [f for f in os.listdir(directory) if f.endswith(file_extension)]

    for file in files:
        yield DataToUpsert(
            value=os.path.join(directory, file),
            payload=payload_generator(file),
        )


def audio_payload_generator(file: str) -> Dict[str, str]:
    """Generate payload for audio files."""
    category = get_category_from_filename(file)
    return {
        "type": "sound",
        "filename": file,
        "category": category,
    }


def image_payload_generator(file: str) -> Dict[str, str]:
    """Generate payload for image files."""
    return {
        "type": "image",
        "filename": file,
    }
