import pytest

from pathlib import Path

from yora_ocr.helper import find_json_files
from yora_ocr.ocr import ocr
from yora_ocr.schema import Purchase


@pytest.fixture(
    scope="session",
    params=find_json_files("/home/julian/Nextcloud/yora_dataset/"),
    ids=lambda path: Path(path).name,
)
def dataset_entry(request):
    json_path = request.param
    jpg_path = json_path.replace(".json", ".jpg")

    with open(json_path) as file:
        expected = Purchase.model_validate_json(file.read())

    real = ocr(jpg_path)

    return expected, real


def test_all(dataset_entry):
    expected, real = dataset_entry

    assert real == expected


def test_price_tags(dataset_entry):
    expected, real = dataset_entry

    # total price
    assert real.total == expected.total

    # check the length of items first
    assert len(real.items) == len(expected.items)

    # collect price list of expected
    expected_price_list = []
    for expected_shop_item in expected.items.values():
        expected_price_list.append(expected_shop_item.price)

    # compare with
    for real_shop_item in real.items.values():
        try:
            expected_price_list.remove(real_shop_item.price)
        except ValueError:
            pytest.fail(
                f"Parsed price of {real_shop_item.price} not found in expected output"
            )
