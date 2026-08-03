from paddleocr import PaddleOCR
from paddlex.inference.pipelines.ocr.result import OCRResult

from yora_ocr.price_detection import find_item_prices_and_total_price

from .schema import Purchase


def ocr(path: str) -> Purchase:

    result_dict = {
        "items": {},
        "total": 0,
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

        debug_overlay_path = "output/" + path.rsplit("/")[-1].replace(
            ".jpg", "_debug_overlay.jpg"
        )
        _, item_prices, total_price = find_item_prices_and_total_price(
            res["rec_polys"], res["rec_texts"], path, debug_overlay_path
        )
        result_dict["total"] = total_price
        for i, price in enumerate(item_prices):
            result_dict["items"][f"item{i}"] = {
                "price": price,
                "price_per_unit": price,
            }

    result_check = Purchase.model_validate(result_dict)
    return result_check
