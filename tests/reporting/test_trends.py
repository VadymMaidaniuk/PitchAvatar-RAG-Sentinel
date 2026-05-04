from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from apps.sentinel_console import (
    failed_query_chart_rows,
    filter_trend_history,
    latest_dataset_status_rows,
    trend_chart_rows,
    trend_dataset_options,
    trend_table_rows,
)
from pitchavatar_rag_sentinel.reporting.artifacts import (
    latest_run_by_dataset,
    load_run_history,
)
from pitchavatar_rag_sentinel.reporting.trends import render_trends_html, write_trends_csv
from scripts import generate_trends_report

pytestmark = pytest.mark.offline


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_trend_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts" / "runs"
    alpha_old = root / "qa-dataset-alpha-1700000000000-aaaaaaaa" / "dataset-alpha"
    alpha_new = root / "qa-dataset-alpha-1700000100000-bbbbbbbb" / "dataset-alpha"
    beta_old = root / "legacy-run" / "dataset-beta"

    write_json(
        alpha_old / "summary.json",
        {
            "run_id": "qa-dataset-alpha-1700000000000-aaaaaaaa",
            "dataset_id": "dataset-alpha",
            "run_passed": False,
            "all_queries_passed": False,
            "cleanup_failed": False,
            "metrics": {
                "query_pass_rate": 0.5,
                "top1_document_accuracy": 0.0,
                "document_hit_rate_at_k": 1.0,
                "top1_chunk_match_rate": 0.25,
                "chunk_hit_rate_at_k": 0.5,
                "forbidden_doc_violation_rate": 0.0,
                "forbidden_chunk_violation_rate": 0.0,
                "total_run_ms": 100.0,
                "p95_search_ms": 40.0,
            },
            "query_results": [
                {"query_id": "q_pass", "passed": True},
                {"query_id": "q_fail", "passed": False},
            ],
        },
    )
    write_json(
        alpha_new / "summary.json",
        {
            "run_id": "qa-dataset-alpha-1700000100000-bbbbbbbb",
            "dataset_id": "dataset-alpha",
            "run_passed": True,
            "all_queries_passed": True,
            "cleanup_failed": False,
            "metrics": {
                "query_pass_rate": 1.0,
                "top1_document_accuracy": 1.0,
                "chunk_hit_rate_at_k": 1.0,
                "total_run_ms": 80.0,
                "p95_search_ms": 25.0,
            },
            "query_results": [{"query_id": "q_pass", "passed": True}],
        },
    )
    write_json(
        beta_old / "summary.json",
        {
            "run_id": "legacy-run",
            "dataset_id": "dataset-beta",
            "query_results": [{"query_id": "q_legacy", "passed": False}],
        },
    )
    (alpha_new / "report.html").write_text("<html>alpha report</html>", encoding="utf-8")
    os.utime(beta_old / "summary.json", (100.0, 100.0))
    return root


def test_run_history_loader_finds_multiple_runs(tmp_path: Path) -> None:
    root = write_trend_artifacts(tmp_path)

    history = load_run_history(root)

    assert len(history) == 3
    assert {row.dataset_id for row in history} == {"dataset-alpha", "dataset-beta"}


def test_run_history_loader_handles_old_artifacts_without_metrics(tmp_path: Path) -> None:
    root = write_trend_artifacts(tmp_path)

    beta_row = next(row for row in load_run_history(root) if row.dataset_id == "dataset-beta")

    assert beta_row.query_pass_rate is None
    assert beta_row.top1_document_accuracy is None
    assert beta_row.total_queries == 1
    assert beta_row.created_at_source == "summary_mtime"


def test_latest_run_per_dataset_is_selected_correctly(tmp_path: Path) -> None:
    root = write_trend_artifacts(tmp_path)

    latest = latest_run_by_dataset(load_run_history(root))

    assert latest["dataset-alpha"].run_id == "qa-dataset-alpha-1700000100000-bbbbbbbb"
    assert latest["dataset-beta"].run_id == "legacy-run"


def test_failed_query_count_is_calculated_correctly(tmp_path: Path) -> None:
    root = write_trend_artifacts(tmp_path)

    alpha_old = next(
        row
        for row in load_run_history(root)
        if row.run_id == "qa-dataset-alpha-1700000000000-aaaaaaaa"
    )

    assert alpha_old.failed_queries == 1
    assert alpha_old.total_queries == 2


def test_trends_report_html_is_generated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = write_trend_artifacts(tmp_path)
    output_path = tmp_path / "trend-output" / "trends.html"
    csv_path = tmp_path / "trend-output" / "trends.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_trends_report.py",
            "--artifacts-root",
            str(root),
            "--output",
            str(output_path),
            "--csv-output",
            str(csv_path),
        ],
    )

    assert generate_trends_report.main() == 0

    html = output_path.read_text(encoding="utf-8")
    assert "Artifact Run History" in html
    assert "Latest Status Per Dataset" in html
    assert "dataset-alpha" in html
    assert "report.html" in html


def test_trends_csv_is_generated(tmp_path: Path) -> None:
    root = write_trend_artifacts(tmp_path)
    csv_path = tmp_path / "trends.csv"

    write_trends_csv(load_run_history(root), csv_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert "query_pass_rate" in rows[0]
    assert {row["dataset_id"] for row in rows} == {"dataset-alpha", "dataset-beta"}


def test_streamlit_trend_helpers_filter_and_format_rows(tmp_path: Path) -> None:
    root = write_trend_artifacts(tmp_path)
    history = load_run_history(root)

    filtered = filter_trend_history(history, ["dataset-alpha"])

    assert trend_dataset_options(history) == ["dataset-alpha", "dataset-beta"]
    assert [row.dataset_id for row in filtered] == ["dataset-alpha", "dataset-alpha"]
    assert latest_dataset_status_rows(filtered)[0]["run_passed"] == "passed"
    assert trend_table_rows(filtered, artifacts_root=root)[0]["report_html"].endswith(
        "report.html"
    )
    assert trend_chart_rows(filtered, "query_pass_rate")[0]["query_pass_rate"] == 0.5
    assert failed_query_chart_rows(filtered)[0]["failed_queries"] == 1


def test_render_trends_html_handles_empty_history() -> None:
    html = render_trends_html([])

    assert "No artifact summaries were found." in html
