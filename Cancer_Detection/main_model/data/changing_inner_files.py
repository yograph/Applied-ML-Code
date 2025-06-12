import shutil
from pathlib import Path

def flatten_and_prefix(root_dir: str):
    """
    Moves every image out of its numbered subfolder into root_dir,
    renaming it from:
       <orig_folder>/<image_name>.ext
    to:
       <orig_folder>_<image_name>.ext
    and then deletes the now-empty subfolders.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"{root_dir} is not a directory")

    # go through each immediate subdirectory
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        folder_name = sub.name

        # move & rename each image
        for img in sub.iterdir():
            if not img.is_file():
                continue
            if img.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}:
                continue

            # build new name: "5_cat.png", "28_dog.jpg", etc.
            new_name = f"{folder_name}_{img.name}"
            dst = root / new_name

            # if there's a name collision, append a counter
            if dst.exists():
                stem, suffix = Path(new_name).stem, Path(new_name).suffix
                i = 1
                while True:
                    candidate = root / f"{stem}_{i}{suffix}"
                    if not candidate.exists():
                        dst = candidate
                        break
                    i += 1

            img.rename(dst)

        # delete the empty folder tree
        shutil.rmtree(sub)

if __name__ == "__main__":
    flatten_and_prefix("train_images_processed_512")
    flatten_and_prefix("images_png")
    print("All images moved and renamed. Subfolders removed.")
