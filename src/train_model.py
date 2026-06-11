import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    ARTIFACTS_DIR,
    CONFUSION_MATRIX_FILENAME,
    DATASET_DIR,
    FEATURES_FILENAME,
    GLCM_FEATURE_NAMES,
    LABEL_ENCODER_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    REPORT_FILENAME,
)
from src.dataset_loader import build_feature_dataframe


def train_model(dataset_dir: str | Path = DATASET_DIR, artifacts_dir: str | Path = ARTIFACTS_DIR) -> dict[str, object]:
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    dataframe = build_feature_dataframe(dataset_dir)
    dataframe.to_csv(artifacts_path / FEATURES_FILENAME, index=False)

    x = dataframe.loc[:, list(GLCM_FEATURE_NAMES)]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataframe["label"])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    class_names = [str(name) for name in label_encoder.classes_]
    report_text = classification_report(y_test, predictions, target_names=class_names)
    report_dict = classification_report(y_test, predictions, target_names=class_names, output_dict=True)
    matrix = confusion_matrix(y_test, predictions)

    joblib.dump(model, artifacts_path / MODEL_FILENAME)
    joblib.dump(label_encoder, artifacts_path / LABEL_ENCODER_FILENAME)
    (artifacts_path / REPORT_FILENAME).write_text(report_text, encoding="utf-8")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(Path(dataset_dir)),
        "total_samples": int(len(dataframe)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "accuracy": accuracy,
        "classes": class_names,
        "feature_names": list(GLCM_FEATURE_NAMES),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
    }
    (artifacts_path / METRICS_FILENAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(matrix, class_names, artifacts_path / CONFUSION_MATRIX_FILENAME)
    return metrics


def save_confusion_matrix(matrix, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Kondisi Visual Rupiah")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training model klasifikasi kondisi visual uang Rupiah.")
    parser.add_argument("--dataset", default=str(DATASET_DIR), help="Path folder dataset.")
    parser.add_argument("--artifacts", default=str(ARTIFACTS_DIR), help="Path folder output artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(args.dataset, args.artifacts)
    print(f"Training selesai. Accuracy: {metrics['accuracy']:.4f}")
    print(f"Artifacts disimpan di: {Path(args.artifacts).resolve()}")


if __name__ == "__main__":
    main()
