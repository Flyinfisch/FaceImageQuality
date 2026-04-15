from pathlib import Path
import cv2
from face_image_quality import SER_FIQ

# Folder that contains all your images
DATA_DIR = Path("data")

# Allowed image types
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_files(folder: Path):
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def main():
    if not DATA_DIR.exists():
        print(f"Folder not found: {DATA_DIR.resolve()}")
        return

    image_files = get_image_files(DATA_DIR)

    if not image_files:
        print(f"No images found in: {DATA_DIR.resolve()}")
        return

    # Use CPU
    serfiq = SER_FIQ(gpu=None)

    for i, image_path in enumerate(image_files, start=1):
        img = cv2.imread(str(image_path))

        if img is None:
            print(f"image {i}: could not read file")
            continue

        aligned = serfiq.apply_mtcnn(img)

        if aligned is None:
            print(f"image {i}: no face detected")
            continue

        score = serfiq.get_score(aligned)
        print(f"image {i}: SER-FIQ score = {score:.4f}")


if __name__ == "__main__":
    main()