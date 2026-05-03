from __future__ import annotations

import json
from pathlib import Path

import pytest

from pitchavatar_rag_sentinel.reporting.artifacts import (
    ArtifactLoader,
    load_artifact_report,
    load_query_artifacts,
)
from pitchavatar_rag_sentinel.reporting.report import render_html_report, write_html_report
from scripts import generate_report

pytestmark = pytest.mark.offline


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_synthetic_artifacts(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifacts" / "runs" / "run-001" / "dataset-alpha"
    write_json(
        artifact_dir / "summary.json",
        {
            "run_id": "run-001",
            "dataset_id": "dataset-alpha",
            "run_passed": False,
            "all_queries_passed": False,
            "cleanup_failed": True,
            "cleanup_warning": "cleanup failed but report generation should continue",
            "metrics": {
                "query_pass_rate": 0.5,
                "top1_document_accuracy": 0.0,
                "document_hit_rate_at_k": 1.0,
                "top1_chunk_match_rate": None,
                "chunk_hit_rate_at_k": None,
                "forbidden_doc_violation_rate": 0.0,
                "forbidden_chunk_violation_rate": None,
                "empty_query_pass_rate": None,
                "total_run_ms": 120.0,
                "seed_total_ms": 10.0,
                "search_total_ms": 40.0,
                "cleanup_total_ms": 70.0,
                "p50_search_ms": 20.0,
                "p95_search_ms": 30.0,
                "max_search_ms": 30.0,
            },
            "query_results": [
                {"query_id": "q_pass", "passed": True},
                {"query_id": "q_fail", "passed": False},
            ],
            "cleanup_results": [
                {
                    "runtime_document_id": "runtime-doc-a",
                    "cleanup_status": "failed",
                    "cleanup_method": "grpc_delete",
                    "cleanup_verified": False,
                    "cleanup_errors": [
                        {
                            "method": "grpc_delete",
                            "error_type": "builtins.TimeoutError",
                            "error_repr": "TimeoutError('not absent')",
                        }
                    ],
                }
            ],
        },
    )
    write_json(
        artifact_dir / "queries" / "q_pass.json",
        {
            "query_id": "q_pass",
            "request": {"query": "known good query"},
            "response": {
                "results": [
                    {
                        "document_id": "runtime-doc-a",
                        "page_content": "expected chunk",
                        "metadata": {},
                        "score": 1.0,
                    }
                ]
            },
            "evaluation": {
                "passed": True,
                "checks": [{"name": "expected_top1", "passed": True, "details": "ok"}],
                "returned_document_ids": ["runtime-doc-a"],
                "result_count": 1,
            },
        },
    )
    write_json(
        artifact_dir / "queries" / "q_fail.json",
        {
            "query_id": "q_fail",
            "request": {"query": "known failing query"},
            "response": {
                "results": [
                    {
                        "document_id": "runtime-doc-wrong",
                        "page_content": "wrong chunk",
                        "metadata": {},
                        "score": 0.5,
                    }
                ]
            },
            "evaluation": {
                "passed": False,
                "checks": [
                    {
                        "name": "expected_top1",
                        "passed": False,
                        "level": "document",
                        "details": "actual_top1=runtime-doc-wrong",
                        "matched_document_id": None,
                        "failure_reason": "top result did not match expected document",
                    }
                ],
                "returned_document_ids": ["runtime-doc-wrong"],
                "result_count": 1,
            },
        },
    )
    return artifact_dir


def test_artifact_loader_loads_synthetic_summary_json(tmp_path: Path) -> None:
    artifact_dir = write_synthetic_artifacts(tmp_path)
    loader = ArtifactLoader(tmp_path / "artifacts" / "runs")

    runs = loader.list_runs()
    datasets = loader.list_datasets(runs[0].path)
    report = loader.load_report(artifact_dir)

    assert [run.run_id for run in runs] == ["run-001"]
    assert [dataset.dataset_id for dataset in datasets] == ["dataset-alpha"]
    assert report.run_id == "run-001"
    assert report.dataset_id == "dataset-alpha"
    assert report.run_passed is False
    assert report.metrics["query_pass_rate"] == 0.5


def test_artifact_loader_loads_query_artifacts(tmp_path: Path) -> None:
    artifact_dir = write_synthetic_artifacts(tmp_path)

    query_payloads = load_query_artifacts(artifact_dir)
    report = load_artifact_report(artifact_dir)

    assert {payload["query_id"] for payload in query_payloads} == {"q_pass", "q_fail"}
    assert report.query_results[0].query_text == "known good query"
    assert report.query_results[1].top_document_id == "runtime-doc-wrong"


def test_failed_queries_are_detected_correctly(tmp_path: Path) -> None:
    artifact_dir = write_synthetic_artifacts(tmp_path)

    report = load_artifact_report(artifact_dir)

    assert [query.query_id for query in report.failed_query_results] == ["q_fail"]
    assert report.failed_query_results[0].failed_checks[0]["name"] == "expected_top1"


def test_report_generator_script_creates_report_html(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_dir = write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["generate_report.py", "--run-dir", str(artifact_dir)],
    )

    assert generate_report.main() == 0

    assert (artifact_dir / "report.html").is_file()


def test_missing_optional_fields_do_not_crash_report_generator(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "runs" / "run-minimal" / "dataset-minimal"
    write_json(
        artifact_dir / "summary.json",
        {
            "run_id": "run-minimal",
            "dataset_id": "dataset-minimal",
            "query_results": [{"query_id": "q_legacy", "passed": False}],
        },
    )

    report = load_artifact_report(artifact_dir)
    report_path = write_html_report(report)

    assert report_path.is_file()
    assert "No retrieval metrics" in report_path.read_text(encoding="utf-8")


def test_report_includes_run_passed_query_pass_rate_and_failed_query_id(
    tmp_path: Path,
) -> None:
    artifact_dir = write_synthetic_artifacts(tmp_path)

    report = load_artifact_report(artifact_dir)
    html = render_html_report(report)

    assert "run_passed" in html
    assert "query_pass_rate" in html
    assert "q_fail" in html
