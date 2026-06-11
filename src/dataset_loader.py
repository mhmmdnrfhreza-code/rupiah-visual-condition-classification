from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import ALLOWED_IMAGE_EXTENSIONS, CONDITION_LABELS
from src.glcm_features import extract_glcm_features_from_path


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    nominal: str
    label: str


def load_image_records(dataset_dir: str | Path) -> list[ImageRecord]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_path}")

    records: list[ImageRecord] = []
    for nominal_dir in sorted(item for item in dataset_path.iterdir() if item.is_dir()):
        for label_dir in sorted(item for item in nominal_dir.iterdir() if item.is_dir()):
            if label_dir.name not in CONDITION_LABELS:
                continue
            for image_path in sorted(label_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower().lstrip(".") in ALLOWED_IMAGE_EXTENSIONS:
                    records.append(ImageRecord(path=image_path, nominal=nominal_dir.name, label=label_dir.name))
    return records


def build_feature_dataframe(dataset_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in load_image_records(dataset_dir):
        features = extract_glcm_features_from_path(record.path)
        rows.append(
            {
                "path": str(record.path),
                "nominal": record.nominal,
                "label": record.label,
                **features,
            }
        )

    if not rows:
        raise ValueError("Dataset tidak berisi gambar yang dapat diproses.")
    return pd.DataFrame(rows)


def summarize_records(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        summary.setdefault(record.nominal, {})
        summary[record.nominal][record.label] = summary[record.nominal].get(record.label, 0) + 1
    return summary
