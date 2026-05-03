from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from apps.sentinel_console import (
    all_query_rows,
    cleanup_failed_text,
    failed_check_names,
    failed_query_rows,
    format_metric_value,
    metric_rows,
    query_detail_payload,
    sort_runs_latest_first,
    summary_rows,
    timing_rows,
)
from pitchavatar_rag_sentinel.reporting.artifacts import (
    ArtifactDatasetRef,
    ArtifactRunRef,
    load_artifact_report,
)

pytestmark = pytest.mark.offline


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_console_artifacts(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifacts" / "runs" / "run-console" / "dataset-console"
    write_json(
        artifact_dir / "summary.json",
        {
            "run_id": "run-console",
            "dataset_id": "dataset-console",
            "run_passed": False,
            "all_queries_passed": False,
            "cleanup_failed": False,
            "metrics": {
                "query_pass_rate": 0.5,
                "top1_document_accuracy": None,
                "total_run_ms": 12.3,
            },
            "query_results": [
                {"query_id": "q_pass", "passed": True},
                {"query_id": "q_fail", "passed": False},
            ],
        },
    )
    write_json(
        artifact_dir / "queries" / "q_pass.json",
        {
            "query_id": "q_pass",
            "request": {"query": "passing query"},
            "response": {"results": [{"document_id": "doc-a"}]},
            "evaluation": {
                "passed": True,
                "checks": [{"name": "expected_top1", "passed": True}],
                "returned_document_ids": ["doc-a"],
                "result_count": 1,
            },
        },
    )
    write_json(
        artifact_dir / "queries" / "q_fail.json",
        {
            "query_id": "q_fail",
            "request": {"query": "failing query"},
            "response": {"results": [{"document_id": "doc-wrong"}]},
            "evaluation": {
                "passed": False,
                "checks": [
                    {
                        "name": "expected_top1",
                        "passed": False,
                        "details": "actual_top1=doc-wrong",
                        "failure_reason": "expected top result was not returned",
                    }
                ],
                "returned_document_ids": ["doc-wrong"],
                "result_count": 1,
            },
        },
    )
    return artifact_dir


def test_format_metric_value_handles_none_float_and_int() -> None:
    assert format_metric_value(None) == "n/a"
    assert format_metric_value(0.5000) == "0.5"
    assert format_metric_value(3) == "3"


def test_failed_check_helpers_extract_names_and_reasons(tmp_path: Path) -> None:
    report = load_artifact_report(write_console_artifacts(tmp_path))
    failed_query = report.failed_query_results[0]

    assert failed_check_names(failed_query) == "expected_top1"
    assert failed_query_rows(report) == [
        {
            "query_id": "q_fail",
            "query": "failing query",
            "failed_checks": "expected_top1",
            "failure_reasons": "expected top result was not returned",
            "top_document_id": "doc-wrong",
        }
    ]


def test_cleanup_failed_false_displays_as_ok_not_failed(tmp_path: Path) -> None:
    report = load_artifact_report(write_console_artifacts(tmp_path))

    assert cleanup_failed_text(report.cleanup_failed) == "ok"
    assert {"field": "cleanup_failed", "value": "ok"} in summary_rows(report)
    assert {"field": "cleanup_failed", "value": "failed"} not in summary_rows(report)


def test_cleanup_failed_true_displays_as_failed(tmp_path: Path) -> None:
    artifact_dir = write_console_artifacts(tmp_path)
    summary_path = artifact_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["cleanup_failed"] = True
    write_json(summary_path, payload)
    report = load_artifact_report(artifact_dir)

    assert cleanup_failed_text(report.cleanup_failed) == "failed"
    assert {"field": "cleanup_failed", "value": "failed"} in summary_rows(report)


def test_sort_runs_latest_first_uses_summary_mtime(tmp_path: Path) -> None:
    old_dataset_path = tmp_path / "runs" / "run-old" / "dataset"
    new_dataset_path = tmp_path / "runs" / "run-new" / "dataset"
    write_json(old_dataset_path / "summary.json", {"run_id": "run-old"})
    write_json(new_dataset_path / "summary.json", {"run_id": "run-new"})
    os.utime(old_dataset_path / "summary.json", (100.0, 100.0))
    os.utime(new_dataset_path / "summary.json", (200.0, 200.0))
    runs = [
        ArtifactRunRef(
            run_id="run-old",
            path=old_dataset_path.parent,
            datasets=[
                ArtifactDatasetRef(
                    run_id="run-old",
                    dataset_id="dataset",
                    path=old_dataset_path,
                )
            ],
        ),
        ArtifactRunRef(
            run_id="run-new",
            path=new_dataset_path.parent,
            datasets=[
                ArtifactDatasetRef(
                    run_id="run-new",
                    dataset_id="dataset",
                    path=new_dataset_path,
                )
            ],
        ),
    ]

    assert [run.run_id for run in sort_runs_latest_first(runs)] == ["run-new", "run-old"]


def test_console_table_rows_are_derived_from_loaded_report(tmp_path: Path) -> None:
    report = load_artifact_report(write_console_artifacts(tmp_path))

    assert metric_rows(report)[:2] == [
        {"metric": "query_pass_rate", "value": "0.5"},
        {"metric": "top1_document_accuracy", "value": "n/a"},
    ]
    assert all_query_rows(report) == [
        {
            "query_id": "q_pass",
            "status": "passed",
            "returned_results": 1,
            "failed_checks": "none",
            "top_document_id": "doc-a",
        },
        {
            "query_id": "q_fail",
            "status": "failed",
            "returned_results": 1,
            "failed_checks": "expected_top1",
            "top_document_id": "doc-wrong",
        },
    ]
    assert query_detail_payload(report.query_results[0])["query_id"] == "q_pass"


def test_missing_metrics_produce_standard_metric_rows_with_na(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "runs" / "run-old" / "dataset-old"
    write_json(
        artifact_dir / "summary.json",
        {
            "run_id": "run-old",
            "dataset_id": "dataset-old",
            "query_results": [],
        },
    )
    report = load_artifact_report(artifact_dir)

    rows = metric_rows(report)

    assert {"metric": "query_pass_rate", "value": "n/a"} in rows
    assert {"metric": "top1_document_accuracy", "value": "n/a"} in rows
    assert len(rows) >= 8


def test_missing_timing_metrics_produce_standard_timing_rows_with_na(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "runs" / "run-old" / "dataset-old"
    write_json(
        artifact_dir / "summary.json",
        {
            "run_id": "run-old",
            "dataset_id": "dataset-old",
            "metrics": {"query_pass_rate": 1.0},
            "query_results": [],
        },
    )
    report = load_artifact_report(artifact_dir)

    rows = timing_rows(report)

    assert {"metric": "total_run_ms", "value": "n/a"} in rows
    assert {"metric": "p95_search_ms", "value": "n/a"} in rows
    assert len(rows) >= 7
