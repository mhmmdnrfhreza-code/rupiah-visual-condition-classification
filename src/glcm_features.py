from pathlib import Path

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from src.config import GLCM_ANGLES, GLCM_DISTANCES, GLCM_FEATURE_NAMES, GLCM_LEVELS, GLCM_PROPERTIES
from src.preprocessing import preprocess_image


def extract_glcm_features(grayscale_image: np.ndarray) -> dict[str, float]:
    """Extract averaged GLCM texture features from a grayscale uint8 image."""
    if grayscale_image.ndim != 2:
        raise ValueError("GLCM membutuhkan gambar grayscale 2 dimensi.")

    image = np.clip(grayscale_image, 0, GLCM_LEVELS - 1).astype(np.uint8)
    glcm = graycomatrix(
        image,
        distances=list(GLCM_DISTANCES),
        angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    features: dict[str, float] = {}
    for property_name, feature_name in zip(GLCM_PROPERTIES, GLCM_FEATURE_NAMES, strict=True):
        values = graycoprops(glcm, property_name)[0]
        features[feature_name] = float(np.nan_to_num(values, nan=0.0).mean())
    return features


def extract_glcm_features_from_path(image_path: str | Path) -> dict[str, float]:
    return extract_glcm_features(preprocess_image(image_path))
