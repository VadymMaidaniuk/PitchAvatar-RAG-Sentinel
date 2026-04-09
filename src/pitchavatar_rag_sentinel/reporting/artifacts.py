from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pitchavatar_rag_sentinel.config import SentinelSettings


class ArtifactWriter:
    def __init__(self, settings: SentinelSettings) -> None:
        self._root = Path(settings.artifacts_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def prepare_run_dir(self, run_id: str, dataset_id: str) -> Path:
        run_dir = self._root / run_id / dataset_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

