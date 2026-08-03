from .ocr import ocr
from .helper import find_images
from pathlib import Path

images = find_images("/home/julian/Nextcloud/yora_dataset/")
print(f"Collected {len(images)} images from yora dataset")
for image in images:
    json_path = f"output/{image.replace('.jpg', '.json').rsplit('/')[-1]}"
    if Path(json_path).exists():
        print(f"File {json_path} already exists, skipping...")
        continue

    res = ocr(image)
    with open(json_path, "w") as file:
        file.write(res.model_dump_json(indent=2))

    print(f"Successfully wrote ocr output to {json_path}")
