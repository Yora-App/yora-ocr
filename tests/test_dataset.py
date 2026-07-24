import pytest

from yora_ocr.helper import find_json_files
from yora_ocr.ocr import ocr
from yora_ocr.schema import Purchase


@pytest.mark.parametrize("test_json", find_json_files("/home/julian/Nextcloud/yora_dataset/"))
def test_dataset(test_json):
    with open(test_json, "r") as file:
        text = file.read()
    expected = Purchase.model_validate_json(text)

    jpg_path = test_json.replace(".json", ".jpg")
    real = ocr(jpg_path)

    assert real == expected
