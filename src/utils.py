from dataclasses import dataclass


@dataclass(frozen=True)
class VisualStatus:
    status: str
    description: str
    css_class: str


VISUAL_STATUS_MAP = {
    "normal": VisualStatus("Layak", "Uang terlihat baik atau bersih.", "status-good"),
    "scuffed": VisualStatus("Kurang layak", "Uang terlihat kusut atau lecek.", "status-warning"),
    "dirty": VisualStatus("Kurang layak", "Uang terlihat kotor.", "status-warning"),
    "scuffed-dirty": VisualStatus("Kurang layak", "Uang terlihat kusut/lecek sekaligus kotor.", "status-warning"),
    "torn": VisualStatus("Tidak layak", "Uang terlihat sobek atau rusak.", "status-danger"),
}


DISPLAY_LABELS = {
    "normal": "Normal",
    "dirty": "Dirty",
    "scuffed": "Scuffed",
    "scuffed-dirty": "Scuffed-dirty",
    "torn": "Torn",
}


def get_visual_status(label: str) -> VisualStatus:
    try:
        return VISUAL_STATUS_MAP[label]
    except KeyError as exc:
        raise ValueError(f"Label kondisi tidak dikenal: {label}") from exc


def get_display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def format_confidence(value: float | None) -> str:
    if value is None:
        return "Tidak tersedia"
    return f"{value * 100:.2f}%"
