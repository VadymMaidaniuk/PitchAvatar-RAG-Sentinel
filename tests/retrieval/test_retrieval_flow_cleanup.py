from __future__ import annotations

import json
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


class CleanupFailingOpenSearchHelper:
    def wait_until_document_present(
        self,
        document_id: str,
        *,
        min_chunks: int = 1,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 1.0,
    ) -> int:
        return 1

    def wait_until_document_absent(
        self,
        document_id: str,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        raise AssertionError(f"{document_id} cleanup visibility failed")

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
        opensearch_allowed_targets=[
            "qa-sentinel-write",
            "qa-sentinel-read",
            "qa-sentinel-000001",
        ],
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


def test_summary_artifact_is_written_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    settings: SentinelSettings,
) -> None:
    monkeypatch.setattr(
        retrieval_flow_module,
        "unique_document_id",
        lambda namespace, slug: f"{namespace}-{slug}",
    )

    executor = RetrievalFlowExecutor(
        settings=settings,
        rag_client=FakeRagClient(),
        opensearch_helper=CleanupFailingOpenSearchHelper(),
        artifact_writer=ArtifactWriter(settings),
    )
    dataset = RetrievalDataset.model_validate(
        {
            "dataset_id": "cleanup-fails",
            "description": "summary artifact cleanup failure regression",
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

    summary = executor.run_dataset(dataset)

    summary_path = Path(summary.run_dir) / "summary.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    cleanup_result = payload["cleanup_results"][0]
    cleanup_error = cleanup_result["cleanup_errors"][0]
    assert cleanup_result["cleanup_status"] == "failed"
    assert cleanup_result["cleanup_method"] == "grpc_delete"
    assert cleanup_result["cleanup_verified"] is False
    assert cleanup_error["method"] == "grpc_delete"
    assert cleanup_error["error_type"] == "builtins.AssertionError"
    assert "cleanup visibility failed" in cleanup_error["error_repr"]
    assert "AssertionError" in cleanup_error["traceback"]
