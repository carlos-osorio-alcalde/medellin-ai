from typing import Optional, Generator
import boto3
import os
from datasets import load_dataset
from configuration.load import config
import pathlib

s3 = boto3.client("s3")
BUCKET = config["s3"]["bucket"]
S3_PATH = config["s3"]["path"]

LOCAL_PATH_AUDIOS = config["local"]["audios"]


def search_audios(category: Optional[str] = None) -> Generator:
    """Search for audio files in the specified category."""
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=S3_PATH)
    contents = response.get("Contents", [])
    for content in contents:
        key = content["Key"]
        if category is None or category in key:
            yield key


def download_audios(category: Optional[str] = None) -> None:
    """Download all audio files from the specified category."""
    audios = search_audios(category)

    for audio in audios:
        s3.download_file(
            BUCKET,
            audio,
            f"{LOCAL_PATH_AUDIOS}/{pathlib.Path(audio).name}",
        )


def download_images() -> None:
    """Download the dataset of the images
    from HuggingFace and save as image files."""
    # Load the dataset from HuggingFace
    dataset = load_dataset(config["huggingface"]["images_dataset_path"])

    # Ensure the target directory exists
    target_dir = config["local"]["images"]
    os.makedirs(target_dir, exist_ok=True)

    # Save each image to the target directory
    for i, example in enumerate(dataset["train"]):
        try:
            print(f"Saving image {i}")
            image = example["image"]
            image.save(os.path.join(target_dir, f"image_{i}.png"))
        except Exception as e:
            print(f"Error saving image {i}: {e}")
            pass


if __name__ == "__main__":
    # Download all audio files from the specified category
    download_images()
