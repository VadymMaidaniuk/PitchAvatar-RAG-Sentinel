from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitchavatar_rag_sentinel.clients.opensearch_helper import OpenSearchHelper
from pitchavatar_rag_sentinel.clients.rag_client import RagServiceClient
from pitchavatar_rag_sentinel.config import get_settings
from pitchavatar_rag_sentinel.datasets.loader import load_dataset
from pitchavatar_rag_sentinel.executors.retrieval_flow import RetrievalFlowExecutor
from pitchavatar_rag_sentinel.reporting.artifacts import ArtifactWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dataset-driven RAG retrieval flow: seed -> search -> evaluate -> cleanup."
    )
    parser.add_argument("dataset", type=Path, help="Path to a retrieval dataset JSON file.")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the JSON run summary to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    dataset = load_dataset(args.dataset)
    opensearch_helper = OpenSearchHelper(settings)
    opensearch_helper.ensure_test_index()
    rag_client = RagServiceClient(settings)
    artifact_writer = ArtifactWriter(settings)
    executor = RetrievalFlowExecutor(
        settings=settings,
        rag_client=rag_client,
        opensearch_helper=opensearch_helper,
        artifact_writer=artifact_writer,
    )

    try:
        summary = executor.run_dataset(dataset)
    finally:
        rag_client.close()

    if args.summary:
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=True))

    return 0 if summary.all_queries_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

