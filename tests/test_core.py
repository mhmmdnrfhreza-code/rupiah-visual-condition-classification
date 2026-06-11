import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import GLCM_FEATURE_NAMES, IMAGE_SIZE, LABEL_ENCODER_FILENAME, MODEL_FILENAME
from src.dataset_loader import load_image_records
from src.glcm_features import extract_glcm_features
from src.predict_model import load_artifacts, predict_image
from src.preprocessing import preprocess_image
from src.utils import get_visual_status


class CorePipelineTests(unittest.TestCase):
    def test_preprocess_image_returns_uint8_grayscale_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (32, 24), color=(30, 120, 210)).save(image_path)

            image = preprocess_image(image_path)

        self.assertEqual(image.shape, IMAGE_SIZE)
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreaterEqual(int(image.min()), 0)
        self.assertLessEqual(int(image.max()), 255)

    def test_extract_glcm_features_returns_expected_feature_names(self):
        image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))

        features = extract_glcm_features(image)

        self.assertEqual(list(features.keys()), list(GLCM_FEATURE_NAMES))
        for value in features.values():
            self.assertIsInstance(value, float)

    def test_dataset_loader_reads_nominal_condition_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            image_dir = dataset_dir / "1000" / "normal"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), color="white").save(image_dir / "1k_b_normal_001.png")

            records = load_image_records(dataset_dir)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].nominal, "1000")
        self.assertEqual(records[0].label, "normal")
        self.assertEqual(records[0].path.name, "1k_b_normal_001.png")

    def test_visual_status_mapping_matches_prd(self):
        self.assertEqual(get_visual_status("normal").status, "Layak")
        self.assertEqual(get_visual_status("scuffed").status, "Kurang layak")
        self.assertEqual(get_visual_status("dirty").status, "Kurang layak")
        self.assertEqual(get_visual_status("scuffed-dirty").status, "Kurang layak")
        self.assertEqual(get_visual_status("torn").status, "Tidak layak")

    def test_predict_image_uses_feature_names_without_sklearn_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "sample.png"
            artifacts_dir = temp_path / "artifacts"
            artifacts_dir.mkdir()
            Image.new("RGB", (32, 24), color=(80, 140, 200)).save(image_path)

            training_features = pd.DataFrame(
                [
                    {name: 0.0 for name in GLCM_FEATURE_NAMES},
                    {name: 1.0 for name in GLCM_FEATURE_NAMES},
                    {name: 2.0 for name in GLCM_FEATURE_NAMES},
                    {name: 3.0 for name in GLCM_FEATURE_NAMES},
                ]
            )
            labels = ["normal", "normal", "torn", "torn"]
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(training_features, encoded_labels)
            joblib.dump(model, artifacts_dir / MODEL_FILENAME)
            joblib.dump(label_encoder, artifacts_dir / LABEL_ENCODER_FILENAME)

            load_artifacts.cache_clear()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = predict_image(image_path, artifacts_dir)
            load_artifacts.cache_clear()

        feature_name_warnings = [item for item in caught if "valid feature names" in str(item.message)]
        self.assertEqual(feature_name_warnings, [])
        self.assertIn(result["condition"], {"normal", "torn"})


if __name__ == "__main__":
    unittest.main()
