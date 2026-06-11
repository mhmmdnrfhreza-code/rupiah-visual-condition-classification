from pathlib import Path

import cv2
import numpy as np

from src.config import IMAGE_SIZE


def _read_image(path: Path) -> np.ndarray:
    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("File tidak dapat dibaca sebagai gambar.")
    return image


def preprocess_image(image_path: str | Path, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize an image and convert it to uint8 grayscale for GLCM."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")

    image = _read_image(path)
    resized = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return np.clip(grayscale, 0, 255).astype(np.uint8)
