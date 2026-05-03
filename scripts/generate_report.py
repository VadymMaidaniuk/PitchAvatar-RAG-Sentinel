from __future__ import annotations

import argparse
from pathlib import Path

from pitchavatar_rag_sentinel.reporting.artifacts import (
    find_latest_artifact_dir,
    load_artifact_report,
)
from pitchavatar_rag_sentinel.reporting.report import write_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a read-only HTML report from existing RAG Sentinel artifacts."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Dataset artifact directory containing summary.json.",
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest dataset artifact under --artifacts-root.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/runs"),
        help="Artifact root used with --latest. Defaults to artifacts/runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_dir = find_latest_artifact_dir(args.artifacts_root) if args.latest else args.run_dir
    report = load_artifact_report(artifact_dir)
    output_path = write_html_report(report)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
