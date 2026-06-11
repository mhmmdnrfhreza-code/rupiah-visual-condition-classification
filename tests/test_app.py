import io
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
import app as app_module
from app import create_app
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import GLCM_FEATURE_NAMES, LABEL_ENCODER_FILENAME, MODEL_FILENAME


CONDITION_LABELS = ("normal", "dirty", "scuffed", "scuffed-dirty", "torn")


def write_test_artifacts(artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    training_features = pd.DataFrame(
        [
            {name: 0.0 for name in GLCM_FEATURE_NAMES},
            {name: 1.0 for name in GLCM_FEATURE_NAMES},
            {name: 2.0 for name in GLCM_FEATURE_NAMES},
            {name: 3.0 for name in GLCM_FEATURE_NAMES},
        ]
    )
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(["normal", "normal", "torn", "torn"])
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(training_features, encoded_labels)
    joblib.dump(model, artifacts_dir / MODEL_FILENAME)
    joblib.dump(label_encoder, artifacts_dir / LABEL_ENCODER_FILENAME)


class FlaskUploadTests(unittest.TestCase):
    def test_home_page_loads_upload_form(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(
                {
                    "TESTING": True,
                    "UPLOAD_FOLDER": str(Path(temp_dir) / "uploads"),
                    "ARTIFACTS_DIR": str(Path(temp_dir) / "artifacts"),
                }
            )

            response = app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Identifikasi Kelayakan Visual", response.data)
        self.assertIn(b'name="file"', response.data)
        self.assertIn(b'href="/model"', response.data)
        self.assertIn(b"Model", response.data)

    def test_model_page_renders_required_model_information(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_dir = temp_path / "dataset"
            artifacts_dir = temp_path / "artifacts"
            artifacts_dir.mkdir()
            for label in CONDITION_LABELS:
                image_dir = dataset_dir / "1000" / label
                image_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 24), color=(80, 140, 200)).save(image_dir / f"{label}.png")
            (artifacts_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "accuracy": 0.75,
                        "total_samples": 5,
                        "train_samples": 4,
                        "test_samples": 1,
                        "classification_report": {
                            "macro avg": {
                                "precision": 0.70,
                                "recall": 0.72,
                                "f1-score": 0.71,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            app = create_app(
                {
                    "TESTING": True,
                    "DATASET_DIR": str(dataset_dir),
                    "ARTIFACTS_DIR": str(artifacts_dir),
                    "UPLOAD_FOLDER": str(temp_path / "uploads"),
                }
            )

            response = app.test_client().get("/model")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Informasi Model", response.data)
        self.assertIn(b"Random Forest Classifier", response.data)
        self.assertIn(b"contrast_mean", response.data)
        self.assertIn(b"Distribusi Dataset", response.data)
        self.assertIn(b"75.00%", response.data)
        self.assertIn(b"Confusion Matrix", response.data)
        for label in CONDITION_LABELS:
            self.assertIn(label.encode(), response.data)
        self.assertIn("Keterbatasan Model".encode(), response.data)

    def test_model_example_visual_styles_show_full_landscape_images(self):
        css = Path("static/css/style.css").read_text(encoding="utf-8")

        self.assertIn(".example-image-frame", css)
        self.assertIn("aspect-ratio: 2.15 / 1", css)
        self.assertIn("object-fit: contain", css)

    def test_predict_rejects_invalid_file_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(
                {
                    "TESTING": True,
                    "UPLOAD_FOLDER": str(Path(temp_dir) / "uploads"),
                    "ARTIFACTS_DIR": str(Path(temp_dir) / "artifacts"),
                }
            )

            response = app.test_client().post(
                "/predict",
                data={"file": (io.BytesIO(b"not an image"), "notes.txt")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Format file tidak didukung", response.data)

    def test_predict_rejects_invalid_image_content_without_internal_detail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            flask_app = create_app(
                {
                    "TESTING": True,
                    "UPLOAD_FOLDER": str(Path(temp_dir) / "uploads"),
                    "ARTIFACTS_DIR": str(Path(temp_dir) / "artifacts"),
                }
            )

            response = flask_app.test_client().post(
                "/predict",
                data={"file": (io.BytesIO(b"not a real image"), "sample.png")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Gambar gagal dibaca".encode(), response.data)
        self.assertNotIn(b"Traceback", response.data)

    def test_predict_hides_unexpected_internal_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            flask_app = create_app(
                {
                    "TESTING": True,
                    "UPLOAD_FOLDER": str(Path(temp_dir) / "uploads"),
                    "ARTIFACTS_DIR": str(Path(temp_dir) / "artifacts"),
                }
            )
            flask_app.logger.disabled = True

            def raise_internal_error(_image_path, _artifacts_dir):
                raise RuntimeError("internal path D:\\secret\\model.pkl")

            original_predict_image = app_module.predict_image
            app_module.predict_image = raise_internal_error
            try:
                response = flask_app.test_client().post(
                    "/predict",
                    data={"file": (io.BytesIO(b"fake image bytes"), "sample.png")},
                    content_type="multipart/form-data",
                    follow_redirects=True,
                )
            finally:
                app_module.predict_image = original_predict_image

        self.assertEqual(response.status_code, 200)
        self.assertIn("Gambar gagal diproses".encode(), response.data)
        self.assertNotIn(b"D:\\secret", response.data)

    def test_predict_accepts_valid_image_and_renders_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            upload_dir = temp_path / "uploads"
            artifacts_dir = temp_path / "artifacts"
            image_path = temp_path / "sample.png"
            Image.new("RGB", (32, 24), color=(80, 140, 200)).save(image_path)
            write_test_artifacts(artifacts_dir)
            app = create_app(
                {
                    "TESTING": True,
                    "UPLOAD_FOLDER": str(upload_dir),
                    "ARTIFACTS_DIR": str(artifacts_dir),
                }
            )

            with image_path.open("rb") as image_file:
                response = app.test_client().post(
                    "/predict",
                    data={"file": (image_file, "sample.png")},
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Hasil Prediksi", response.data)
        self.assertIn(b"contrast_mean", response.data)
        self.assertIn(b"asm_mean", response.data)
        self.assertIn("bukan validasi keaslian uang".encode(), response.data)


if __name__ == "__main__":
    unittest.main()
