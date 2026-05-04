from __future__ import annotations

import json
from pathlib import Path

import pytest

from pitchavatar_rag_sentinel.config import SentinelSettings
from scripts import run_dataset

pytestmark = pytest.mark.offline


def make_settings(**overrides: object) -> SentinelSettings:
    values = {
        "grpc_target": "qa-rag-service-dev.pitchavatar.com:443",
        "grpc_secure": True,
        "grpc_server_name": "qa-rag-service-dev.pitchavatar.com",
        "opensearch_url": "https://opensearch-dev.pitchavatar.com",
        "opensearch_username": "unused",
        "opensearch_password": "unused",
        "opensearch_write_alias": "qa-sentinel-write",
        "opensearch_read_alias": "qa-sentinel-read",
        "opensearch_physical_index": "qa-sentinel-000001",
        "opensearch_allowed_targets": [
            "qa-sentinel-write",
            "qa-sentinel-read",
            "qa-sentinel-000001",
        ],
        "auto_create_index": False,
        "namespace": "qa-test",
        "artifacts_dir": "artifacts/runs",
    }
    values.update(overrides)
    return SentinelSettings(_env_file=None, **values)


def write_dataset(path: Path, *, queries: list[dict] | None = None) -> None:
    payload = {
        "dataset_id": "dry-run-sample",
        "description": "dry-run regression dataset",
        "documents": [
            {
                "key": "doc_a",
                "content": "Dry-run test content.",
                "metadata": {"user_id": "qa-test", "type": "txt"},
            }
        ],
        "queries": queries
        if queries is not None
        else [
            {
                "query_id": "q_one",
                "query": "dry-run test",
                "alpha": 0.5,
                "expectations": {"expected_top1": "doc_a"},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(autouse=True)
def block_network_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_client_construction(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not construct network clients or artifact writers")

    monkeypatch.setattr(run_dataset, "OpenSearchHelper", fail_client_construction)
    monkeypatch.setattr(run_dataset, "RagServiceClient", fail_client_construction)
    monkeypatch.setattr(run_dataset, "ArtifactWriter", fail_client_construction)


def test_dry_run_validates_dataset_without_grpc_or_opensearch_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(dataset_path)
    monkeypatch.setattr(run_dataset, "get_settings", make_settings)
    monkeypatch.setattr(
        "sys.argv",
        ["run_dataset.py", str(dataset_path), "--dry-run"],
    )

    assert run_dataset.main() == 0


def test_dry_run_fails_on_duplicate_query_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "duplicate-query-id.json"
    write_dataset(
        dataset_path,
        queries=[
            {
                "query_id": "q_duplicate",
                "query": "first query",
                "alpha": 0.5,
                "expectations": {"expected_top1": "doc_a"},
            },
            {
                "query_id": "q_duplicate",
                "query": "second query",
                "alpha": 0.5,
                "expectations": {"expected_top1": "doc_a"},
            },
        ],
    )
    monkeypatch.setattr(run_dataset, "get_settings", make_settings)
    monkeypatch.setattr(
        "sys.argv",
        ["run_dataset.py", str(dataset_path), "--dry-run"],
    )

    assert run_dataset.main() == 2

    captured = capsys.readouterr()
    assert "Dataset run setup failed:" in captured.err
    assert "query_id values must be unique" in captured.err
    assert "Traceback" not in captured.err


def test_dry_run_fails_on_unsafe_opensearch_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(dataset_path)
    monkeypatch.setattr(
        run_dataset,
        "get_settings",
        lambda: make_settings(opensearch_write_alias="unexpected-shared-index"),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_dataset.py", str(dataset_path), "--dry-run"],
    )

    assert run_dataset.main() == 2

    captured = capsys.readouterr()
    assert "Dataset run setup failed:" in captured.err
    assert "non-allowlisted OpenSearch target" in captured.err
    assert "Traceback" not in captured.err


def test_dry_run_output_contains_dataset_and_case_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "dataset.json"
    write_dataset(dataset_path)
    monkeypatch.setattr(run_dataset, "get_settings", make_settings)
    monkeypatch.setattr(
        "sys.argv",
        ["run_dataset.py", str(dataset_path), "--dry-run"],
    )

    assert run_dataset.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset_id"] == "dry-run-sample"
    assert payload["document_count"] == 1
    assert payload["query_count"] == 1
    assert payload["grpc"]["target"] == "qa-rag-service-dev.pitchavatar.com:443"
    assert payload["opensearch"]["write_target"] == "qa-sentinel-write"
    assert payload["opensearch"]["delete_fallback_enabled"] is False
    assert payload["cleanup"]["fail_on_cleanup_error"] is True
    assert payload["opensearch"]["allowed_targets"] == [
        "qa-sentinel-000001",
        "qa-sentinel-read",
        "qa-sentinel-write",
    ]
