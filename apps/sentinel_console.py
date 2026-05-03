from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pitchavatar_rag_sentinel.reporting.artifacts import (
    ArtifactDatasetRef,
    ArtifactLoader,
    ArtifactRunRef,
    ArtifactRunReport,
    QueryArtifactReport,
    TIMING_METRIC_NAMES,
)
from pitchavatar_rag_sentinel.reporting.report import SUMMARY_METRIC_NAMES


def sort_runs_latest_first(runs: list[ArtifactRunRef]) -> list[ArtifactRunRef]:
    return sorted(
        runs,
        key=lambda run: (_run_latest_mtime(run), run.run_id),
        reverse=True,
    )


def format_metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def failed_check_names(query: QueryArtifactReport) -> str:
    names = [str(check.get("name")) for check in query.failed_checks if check.get("name")]
    return "\n".join(names) if names else "none"


def failure_reasons(query: QueryArtifactReport) -> str:
    reasons = [
        str(check.get("failure_reason") or check.get("details"))
        for check in query.failed_checks
        if check.get("failure_reason") or check.get("details")
    ]
    return "\n".join(reasons)


def query_detail_payload(query: QueryArtifactReport) -> dict[str, Any]:
    if query.raw_artifact:
        return query.raw_artifact
    return {
        "query_id": query.query_id,
        "request": query.request,
        "response": query.response,
        "evaluation": query.evaluation,
    }


def summary_rows(report: ArtifactRunReport) -> list[dict[str, str]]:
    return [
        {"field": "run_id", "value": report.run_id},
        {"field": "dataset_id", "value": report.dataset_id},
        {"field": "run_passed", "value": _status_text(report.run_passed)},
        {"field": "all_queries_passed", "value": _status_text(report.all_queries_passed)},
        {"field": "cleanup_failed", "value": cleanup_failed_text(report.cleanup_failed)},
        {"field": "cleanup_warning", "value": report.cleanup_warning or ""},
    ]


def metric_rows(report: ArtifactRunReport) -> list[dict[str, str]]:
    return _named_value_rows(report.metrics, SUMMARY_METRIC_NAMES)


def timing_rows(report: ArtifactRunReport) -> list[dict[str, str]]:
    return _named_value_rows(report.metrics, TIMING_METRIC_NAMES, unit="ms")


def failed_query_rows(report: ArtifactRunReport) -> list[dict[str, str | int | None]]:
    return [
        {
            "query_id": query.query_id,
            "query": query.query_text,
            "failed_checks": failed_check_names(query),
            "failure_reasons": failure_reasons(query),
            "top_document_id": query.top_document_id,
        }
        for query in report.failed_query_results
    ]


def all_query_rows(report: ArtifactRunReport) -> list[dict[str, str | int | None]]:
    return [
        {
            "query_id": query.query_id,
            "status": "passed" if query.passed else "failed",
            "returned_results": query.result_count,
            "failed_checks": failed_check_names(query),
            "top_document_id": query.top_document_id,
        }
        for query in report.query_results
    ]


def cleanup_rows(report: ArtifactRunReport) -> list[dict[str, str | bool | None]]:
    return [
        {
            "runtime_document_id": _optional_str(result.get("runtime_document_id")),
            "cleanup_status": _optional_str(result.get("cleanup_status")),
            "cleanup_method": _optional_str(result.get("cleanup_method")),
            "cleanup_verified": result.get("cleanup_verified"),
            "errors": _compact_json(result.get("cleanup_errors") or result.get("error")),
        }
        for result in report.cleanup_details
    ]


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="RAG Sentinel Console", layout="wide")
    st.title("RAG Sentinel Console")
    st.caption("Read-only local artifact viewer. No RAG runs or cleanup actions are available here.")

    artifacts_root_text = st.sidebar.text_input("Artifacts root", value="artifacts/runs")
    artifacts_root = Path(artifacts_root_text)
    if not artifacts_root.exists():
        st.info(f"Artifact root does not exist: {artifacts_root}")
        return

    loader = ArtifactLoader(artifacts_root)
    runs = sort_runs_latest_first(loader.list_runs())
    if not runs:
        st.info(f"No artifact runs with summary.json were found under {artifacts_root}.")
        return

    selected_run = st.sidebar.selectbox(
        "Run",
        runs,
        format_func=lambda run: run.run_id,
    )
    if selected_run is None:
        st.info("No run selected.")
        return

    datasets = selected_run.datasets
    selected_dataset = st.sidebar.selectbox(
        "Dataset",
        datasets,
        format_func=lambda dataset: dataset.dataset_id,
    )
    if selected_dataset is None:
        st.info("No dataset selected.")
        return

    report = loader.load_report(selected_dataset.path)
    _render_report(st, report, selected_run, selected_dataset)


def _render_report(
    st: Any,
    report: ArtifactRunReport,
    selected_run: ArtifactRunRef,
    selected_dataset: ArtifactDatasetRef,
) -> None:
    st.subheader("Summary")
    summary_columns = st.columns(3)
    summary_columns[0].metric("Run passed", _status_text(report.run_passed))
    summary_columns[1].metric("All queries passed", _status_text(report.all_queries_passed))
    summary_columns[2].metric("Cleanup failed", cleanup_failed_text(report.cleanup_failed))
    st.dataframe(summary_rows(report), hide_index=True, use_container_width=True)

    st.subheader("Metrics")
    rows = metric_rows(report)
    if not _has_any_metric(report.metrics, SUMMARY_METRIC_NAMES):
        st.info("No metrics were found in summary.json.")
    st.dataframe(rows, hide_index=True, use_container_width=True)

    st.subheader("Timing")
    rows = timing_rows(report)
    if not _has_any_metric(report.metrics, TIMING_METRIC_NAMES):
        st.info("No timing metrics were found in summary.json.")
    st.dataframe(rows, hide_index=True, use_container_width=True)

    st.subheader("Failed Queries")
    rows = failed_query_rows(report)
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.success("No failed queries were recorded.")

    st.subheader("All Queries")
    rows = all_query_rows(report)
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No query results were found.")

    st.subheader("Query Details")
    if report.query_results:
        selected_query = st.selectbox(
            "Query",
            report.query_results,
            format_func=lambda query: query.query_id,
        )
        if selected_query is not None:
            st.write("Evaluation checks")
            st.dataframe(selected_query.checks, hide_index=True, use_container_width=True)
            detail_tabs = st.tabs(["Raw JSON", "Request", "Response", "Evaluation"])
            detail_tabs[0].json(query_detail_payload(selected_query))
            detail_tabs[1].json(selected_query.request)
            detail_tabs[2].json(selected_query.response)
            detail_tabs[3].json(selected_query.evaluation)
    else:
        st.info("No query artifacts are available for detail view.")

    st.subheader("Cleanup")
    if report.cleanup_warning:
        st.warning(report.cleanup_warning)
    st.metric("Cleanup failed", cleanup_failed_text(report.cleanup_failed))
    rows = cleanup_rows(report)
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
        with st.expander("Cleanup details JSON"):
            st.json(report.cleanup_details)
    else:
        st.info("No cleanup details were found.")

    with st.expander("Selected artifact paths"):
        st.json(
            {
                "artifacts_root": str(selected_run.path.parent),
                "run_path": str(selected_run.path),
                "dataset_path": str(selected_dataset.path),
            }
        )


def _named_value_rows(
    values: dict[str, Any],
    names: tuple[str, ...],
    *,
    unit: str | None = None,
) -> list[dict[str, str]]:
    rows = []
    for name in names:
        value = format_metric_value(values.get(name))
        rows.append(
            {
                "metric": name,
                "value": f"{value} {unit}" if unit and value != "n/a" else value,
            }
        )
    return rows


def _run_latest_mtime(run: ArtifactRunRef) -> float:
    mtimes = []
    for dataset in run.datasets:
        summary_path = dataset.path / "summary.json"
        if summary_path.is_file():
            mtimes.append(summary_path.stat().st_mtime)
    return max(mtimes, default=run.path.stat().st_mtime if run.path.exists() else 0.0)


def _status_text(value: bool | None) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return "not reported"


def cleanup_failed_text(value: bool | None) -> str:
    if value is True:
        return "failed"
    if value is False:
        return "ok"
    return "not reported"


def _has_any_metric(values: dict[str, Any], names: tuple[str, ...]) -> bool:
    return any(name in values for name in names)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _compact_json(value: Any) -> str:
    if value in (None, [], {}):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


if __name__ == "__main__":
    main()
