from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline


def test_preview_dataset_source_cli_works_on_temp_markdown(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_path = tmp_path / "sample.md"
    source_path.write_text("# Alpha\nFirst section.\n\n## Beta\nSecond section.", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [sys.executable, "scripts/preview_dataset_source.py", str(source_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Source file: sample.md" in result.stdout
    assert "Sections: 2" in result.stdout
    assert "section_001_alpha | Alpha" in result.stdout
    assert "section_002_beta | Beta" in result.stdout
