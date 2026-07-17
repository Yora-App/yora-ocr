import glob


def find_images() -> list[str]:
    return glob.glob("/home/julian/Nextcloud/yora_dataset/**/*.jpg", recursive=True)


def find_json_files() -> list[str]:
    return glob.glob("/home/julian/Nextcloud/yora_dataset/**/*.json", recursive=True)
