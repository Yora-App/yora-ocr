from ocr import ocr
import glob


def find_images() -> list[str]:
    return glob.glob("/mnt/c/Users/Annalena Frey/Nextcloud/yora_dataset/**/*.jpg", recursive=True)


for image in find_images():
    res = ocr(image)
    json_path = image.replace(".jpg", ".json")
    with open(json_path, "w") as file:
        file.write(res.model_dump_json())