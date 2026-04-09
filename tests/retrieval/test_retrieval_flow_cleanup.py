from __future__ import annotations

from pathlib import Path

import pytest

import pitchavatar_rag_sentinel.executors.retrieval_flow as retrieval_flow_module
from pitchavatar_rag_sentinel.config import SentinelSettings
from pitchavatar_rag_sentinel.datasets.models import RetrievalDataset
from pitchavatar_rag_sentinel.executors.retrieval_flow import RetrievalFlowExecutor
from pitchavatar_rag_sentinel.reporting.artifacts import ArtifactWriter


class _Response:
    def __init__(self, *, success: bool, message: str = "ok") -> None:
        self.success = success
        self.message = message


class FakeRagClient:
    def __init__(self) -> None:
        self.deleted_document_ids: list[str] = []

    def upsert_content(self, *, document_id: str, content: str, metadata: dict[str, str]) -> _Response:
        return _Response(success=True, message=f"upserted {document_id}")

    def delete_document(self, document_id: str) -> _Response:
        self.deleted_document_ids.append(document_id)
        return _Response(success=True, message=f"deleted {document_id}")


class FakeOpenSearchHelper:
    def __init__(self) -> None:
        self.absent_waits: list[str] = []

    def wait_until_document_present(
        self,
        document_id: str,
        *,
        min_chunks: int = 1,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 1.0,
    ) -> int:
        raise TimeoutError(f"{document_id} never became visible")

    def wait_until_document_absent(
        self,
        document_id: str,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.absent_waits.append(document_id)

    def cleanup_document(self, document_id: str) -> None:
        raise AssertionError("fallback cleanup should not be used in this scenario")


@pytest.fixture()
def settings(tmp_path: Path) -> SentinelSettings:
    return SentinelSettings(
        grpc_target="qa-rag-service-dev.pitchavatar.com:443",
        grpc_secure=True,
        opensearch_url="https://opensearch-dev.pitchavatar.com",
        opensearch_username="unused",
        opensearch_password="unused",
        opensearch_write_alias="qa-sentinel-write",
        opensearch_read_alias="qa-sentinel-read",
        opensearch_physical_index="qa-sentinel-000001",
        auto_create_index=False,
        delete_fallback_to_opensearch=False,
        namespace="qa-test",
        artifacts_dir=str(tmp_path / "artifacts"),
    )


def test_cleanup_still_runs_for_document_that_times_out_during_seed(
    monkeypatch: pytest.MonkeyPatch,
    settings: SentinelSettings,
) -> None:
    monkeypatch.setattr(
        retrieval_flow_module,
        "unique_document_id",
        lambda namespace, slug: f"{namespace}-{slug}",
    )

    rag_client = FakeRagClient()
    opensearch_helper = FakeOpenSearchHelper()
    executor = RetrievalFlowExecutor(
        settings=settings,
        rag_client=rag_client,
        opensearch_helper=opensearch_helper,
        artifact_writer=ArtifactWriter(settings),
    )
    dataset = RetrievalDataset.model_validate(
        {
            "dataset_id": "sample",
            "description": "cleanup regression",
            "documents": [
                {
                    "key": "doc_a",
                    "content": "test content",
                    "metadata": {"user_id": "user-alpha", "type": "txt"},
                    "min_expected_chunks": 1,
                }
            ],
            "queries": [],
        }
    )

    with pytest.raises(RuntimeError, match="Dataset run failed"):
        executor.run_dataset(dataset)

    assert rag_client.deleted_document_ids == ["qa-test-sample-doc_a"]
    assert opensearch_helper.absent_waits == ["qa-test-sample-doc_a"]
