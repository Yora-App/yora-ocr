import pytest

from yora_ocr.main import find_images
from yora_ocr.ocr import ocr
from yora_ocr.schema import Purchase

@pytest.mark.parametrize("test_image", find_images())
def test_dataset(test_image):
    json_path= test_image.replace(".jpg", ".json")
    with open(json_path, "r") as file:
        text = file.read(json_path)
    expected = Purchase.model_validate_json(text)
    real = ocr(test_image)
    assert real == expected