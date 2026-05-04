from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pitchavatar_rag_sentinel.reporting.trends import write_trends_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a read-only trends report from existing RAG Sentinel artifacts."
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/runs"),
        help="Artifact root to scan recursively. Defaults to artifacts/runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path. Defaults to <artifacts-root>/trends_report.html.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to the HTML output path with .csv suffix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        html_path, csv_path = write_trends_report(
            args.artifacts_root,
            output_path=args.output,
            csv_output_path=args.csv_output,
        )
    except (OSError, ValueError) as exc:
        print(f"Trends report generation failed: {exc}", file=sys.stderr)
        return 1

    print(html_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
