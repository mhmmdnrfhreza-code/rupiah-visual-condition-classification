from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

IMAGE_SIZE = (256, 256)
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_UPLOAD_SIZE = 8 * 1024 * 1024

CONDITION_LABELS = ("normal", "dirty", "scuffed", "scuffed-dirty", "torn")

GLCM_DISTANCES = (1,)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_LEVELS = 256
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GLCM_FEATURE_NAMES = tuple(f"{name.lower()}_mean" for name in GLCM_PROPERTIES)

MODEL_FILENAME = "model.pkl"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"
FEATURES_FILENAME = "features.csv"
METRICS_FILENAME = "metrics.json"
REPORT_FILENAME = "classification_report.txt"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
