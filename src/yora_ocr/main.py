from .ocr import ocr
from .helper import find_images


for image in find_images():
    res = ocr(image)
    json_path = image.replace(".jpg", ".json")
    with open(json_path, "w") as file:
        file.write(res.model_dump_json())