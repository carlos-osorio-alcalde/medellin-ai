from typing import Optional, Generator
import boto3
from configuration.load import config
import pathlib

s3 = boto3.client("s3")
BUCKET = config["s3"]["bucket"]
S3_PATH = config["s3"]["path"]

LOCAL_PATH = config["local"]["path"]


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
        print("Downloading", audio)
        s3.download_file(
            BUCKET, audio, f"{LOCAL_PATH}/{pathlib.Path(audio).name}"
        )


if __name__ == "__main__":
    # Download all audio files from the specified category
    download_audios()
