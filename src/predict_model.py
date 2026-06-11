from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR, GLCM_FEATURE_NAMES, LABEL_ENCODER_FILENAME, MODEL_FILENAME
from src.glcm_features import extract_glcm_features_from_path
from src.utils import format_confidence, get_display_label, get_visual_status


@lru_cache(maxsize=4)
def load_artifacts(artifacts_dir: str | Path = ARTIFACTS_DIR):
    artifacts_path = Path(artifacts_dir)
    model_path = artifacts_path / MODEL_FILENAME
    encoder_path = artifacts_path / LABEL_ENCODER_FILENAME

    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            "Model belum tersedia. Jalankan `python -m src.train_model --dataset dataset --artifacts artifacts`."
        )

    return joblib.load(model_path), joblib.load(encoder_path)


def predict_image(image_path: str | Path, artifacts_dir: str | Path = ARTIFACTS_DIR) -> dict[str, object]:
    features = extract_glcm_features_from_path(image_path)
    model, label_encoder = load_artifacts(str(Path(artifacts_dir)))

    feature_vector = pd.DataFrame([{name: features[name] for name in GLCM_FEATURE_NAMES}])
    encoded_prediction = model.predict(feature_vector)[0]
    condition = str(label_encoder.inverse_transform([encoded_prediction])[0])

    confidence: float | None = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector)[0]
        class_index = list(model.classes_).index(encoded_prediction)
        confidence = float(probabilities[class_index])

    status = get_visual_status(condition)
    return {
        "image_path": str(image_path),
        "condition": condition,
        "condition_display": get_display_label(condition),
        "visual_status": status.status,
        "status_description": status.description,
        "status_class": status.css_class,
        "confidence": confidence,
        "confidence_display": format_confidence(confidence),
        "features": features,
    }
