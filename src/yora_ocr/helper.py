import glob
from pathlib import Path
from pydantic import BaseModel


def find_images(yora_dataset_path_str: str) -> list[str]:
    yora_dataset_path = Path(yora_dataset_path_str).resolve()
    return glob.glob(f"{yora_dataset_path}/**/*.jpg", recursive=True)


def find_json_files(yora_dataset_path_str: str) -> list[str]:
    yora_dataset_path = Path(yora_dataset_path_str).resolve()
    return glob.glob(f"{yora_dataset_path}/**/*.json", recursive=True)


def model_to_json(model: type[BaseModel]):
    return model.model_json_schema()
