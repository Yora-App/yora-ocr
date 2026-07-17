import re

from paddleocr import PaddleOCR
from paddlex.inference.pipelines.ocr.result import OCRResult

from .schema import Purchase


def ocr(path: str) -> Purchase:

    result_dict = {
        "items": {},
    }

    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    result: list[OCRResult] = ocr.predict(path)
    for res in result:
        res.save_to_img("output")
        res.save_to_json("output")

        for price_box_str, price_box_dim in zip(res["rec_texts"], res["rec_boxes"]):
            # box_dim: [x_min, y_min, x_max, y_max]
            # (x_min, y_min) is the top-left coordinate and (x_max, y_max) is the bottom-right coordinate
            if match := re.fullmatch(r"(\d+[.,]\d{2}).?", price_box_str.strip()):
                price_y_center = (price_box_dim[1] + price_box_dim[3]) / 2
                price_height = price_box_dim[3] - price_box_dim[1]

                for item_box_str, item_box_dim in zip(
                    res["rec_texts"], res["rec_boxes"]
                ):
                    # item should be left of price
                    if item_box_dim[2] < price_box_dim[0]:
                        item_y_center = (item_box_dim[1] + item_box_dim[3]) / 2
                        item_height = item_box_dim[3] - item_box_dim[1]

                        # tolerance: half of the larger height
                        tolerance = max(price_height, item_height) / 2

                        if abs(item_y_center - price_y_center) <= tolerance:
                            price_string = match.group(1).replace(",", ".")
                            result_dict["items"][item_box_str] = {
                                "price": price_string,
                                "price_per_unit": price_string,
                            }

    result_check = Purchase.model_validate(result_dict)
    return result_check
