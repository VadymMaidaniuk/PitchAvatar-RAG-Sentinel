from __future__ import annotations

from pathlib import Path

from pitchavatar_rag_sentinel.datasets.models import RetrievalDataset


def load_dataset(path: str | Path) -> RetrievalDataset:
    dataset_path = Path(path)
    return RetrievalDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))


def discover_datasets(root: str | Path) -> list[Path]:
    datasets_root = Path(root)
    return sorted(datasets_root.rglob("*.json"))

