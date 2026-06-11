import json
from pathlib import Path
from typing import Any

from src.config import (
    ARTIFACTS_DIR,
    CONFUSION_MATRIX_FILENAME,
    DATASET_DIR,
    GLCM_FEATURE_NAMES,
    METRICS_FILENAME,
)
from src.dataset_loader import load_image_records


PREDICTION_CLASSES = [
    {
        "label": "normal",
        "meaning": "Uang terlihat baik atau bersih",
        "status": "Layak",
    },
    {
        "label": "dirty",
        "meaning": "Uang terlihat kotor",
        "status": "Kurang layak",
    },
    {
        "label": "scuffed",
        "meaning": "Uang terlihat kusut atau lecek",
        "status": "Kurang layak",
    },
    {
        "label": "scuffed-dirty",
        "meaning": "Uang terlihat kusut/lecek sekaligus kotor",
        "status": "Kurang layak",
    },
    {
        "label": "torn",
        "meaning": "Uang terlihat sobek atau rusak",
        "status": "Tidak layak",
    },
]

GLCM_FEATURES = [
    {"name": "contrast_mean", "description": "Mengukur perbedaan intensitas piksel."},
    {"name": "dissimilarity_mean", "description": "Mengukur ketidakmiripan tekstur."},
    {"name": "homogeneity_mean", "description": "Mengukur keseragaman tekstur."},
    {"name": "energy_mean", "description": "Mengukur keteraturan tekstur."},
    {"name": "correlation_mean", "description": "Mengukur hubungan antar piksel."},
    {"name": "asm_mean", "description": "Mengukur tingkat keteraturan distribusi tekstur."},
]

VISUAL_EXAMPLES = [
    {
        "label": "normal",
        "status": "Layak",
        "image": "examples/normal.png",
        "description": "Permukaan uang terlihat bersih dan tidak rusak.",
    },
    {
        "label": "dirty",
        "status": "Kurang layak",
        "image": "examples/dirty.png",
        "description": "Terdapat noda atau area kotor pada permukaan uang.",
    },
    {
        "label": "scuffed",
        "status": "Kurang layak",
        "image": "examples/scuffed.png",
        "description": "Permukaan uang terlihat aus, lecet, atau kusam.",
    },
    {
        "label": "scuffed-dirty",
        "status": "Kurang layak",
        "image": "examples/scuffed-dirty.png",
        "description": "Permukaan uang terlihat aus dan kotor.",
    },
    {
        "label": "torn",
        "status": "Tidak layak",
        "image": "examples/torn.png",
        "description": "Uang terlihat sobek atau mengalami kerusakan fisik.",
    },
]


def build_model_page_context(
    dataset_dir: str | Path = DATASET_DIR,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
) -> dict[str, Any]:
    artifacts_path = Path(artifacts_dir)
    metrics = read_metrics(artifacts_path / METRICS_FILENAME)
    distribution = get_dataset_distribution(dataset_dir)
    split_summary = {
        "Training": metrics.get("train_samples"),
        "Testing": metrics.get("test_samples"),
        "Total": metrics.get("total_samples") or distribution["total"],
    }
    return {
        "summary": get_model_summary(),
        "prediction_classes": PREDICTION_CLASSES,
        "glcm_features": GLCM_FEATURES,
        "dataset_distribution": distribution,
        "split_summary": split_summary,
        "metrics": get_metric_cards(metrics),
        "confusion_matrix": {
            "path": "images/confusion_matrix.png",
            "source_exists": (artifacts_path / CONFUSION_MATRIX_FILENAME).exists(),
        },
        "visual_examples": VISUAL_EXAMPLES,
        "workflow_steps": [
            "Dataset uang",
            "Resize gambar",
            "Konversi grayscale",
            "Ekstraksi Fitur GLCM",
            "Training Random Forest",
            "Evaluasi model",
            "Prediksi gambar upload",
        ],
        "limitations": [
            ("Dataset terbatas", "Akurasi bisa kurang stabil."),
            ("Pencahayaan berbeda", "Hasil prediksi bisa berubah."),
            ("Background tidak seragam", "Fitur tekstur dapat terganggu."),
            ("Kelas mirip", "Model bisa salah membedakan beberapa kondisi visual."),
            ("GLCM berbasis tekstur", "Tidak selalu ideal untuk mendeteksi sobekan besar."),
        ],
        "feature_names": list(GLCM_FEATURE_NAMES),
    }


def get_model_summary() -> list[tuple[str, str]]:
    return [
        ("Metode Ekstraksi Fitur", "GLCM"),
        ("Model Klasifikasi", "Random Forest Classifier"),
        ("Input Model", "Fitur tekstur citra grayscale"),
        ("Output Model", "Kondisi visual uang"),
        ("Jumlah Kelas", "5 kelas"),
        ("Ukuran Citra", "256 x 256"),
        ("Sudut GLCM", "0 derajat, 45 derajat, 90 derajat, 135 derajat"),
        ("Distance GLCM", "1"),
        ("Format Gambar", "JPG, JPEG, PNG"),
    ]


def read_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def get_dataset_distribution(dataset_dir: str | Path) -> dict[str, Any]:
    counts = {item["label"]: 0 for item in PREDICTION_CLASSES}
    try:
        records = load_image_records(dataset_dir)
    except FileNotFoundError:
        records = []

    for record in records:
        if record.label in counts:
            counts[record.label] += 1
    return {
        "classes": [{"label": label, "count": count} for label, count in counts.items()],
        "total": sum(counts.values()),
    }


def get_metric_cards(metrics: dict[str, Any]) -> list[dict[str, str]]:
    macro_average = metrics.get("classification_report", {}).get("macro avg", {})
    return [
        {"label": "Accuracy", "value": format_percent(metrics.get("accuracy"))},
        {"label": "Precision", "value": format_percent(macro_average.get("precision"))},
        {"label": "Recall", "value": format_percent(macro_average.get("recall"))},
        {"label": "F1-Score", "value": format_percent(macro_average.get("f1-score"))},
    ]


def format_percent(value: Any) -> str:
    if value is None:
        return "Tidak tersedia"
    return f"{float(value) * 100:.2f}%"
