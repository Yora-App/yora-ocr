import glob

def find_images() -> list[str]:
    return glob.glob("/mnt/c/Users/Annalena Frey/Nextcloud/yora_dataset/**/*.jpg", recursive=True)
