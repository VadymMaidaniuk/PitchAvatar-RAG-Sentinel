from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pitchavatar_rag_sentinel.dataset_builder.drafts import (
    ExpectationDraft,
    QueryDraft,
    build_retrieval_dataset,
    dataset_to_pretty_json,
    document_keys_for_source,
    resolve_document_mode,
)
from pitchavatar_rag_sentinel.dataset_builder.parsers import ParserDependencyError, parse_source_bytes
from pitchavatar_rag_sentinel.reporting.artifacts import (
    ArtifactDatasetRef,
    ArtifactLoader,
    ArtifactRunHistoryRow,
    ArtifactRunRef,
    ArtifactRunReport,
    QueryArtifactReport,
    latest_run_by_dataset,
    sort_run_history_latest_first,
)
from pitchavatar_rag_sentinel.reporting.constants import (
    IR_METRIC_NAMES,
    SUMMARY_METRIC_NAMES,
    TIMING_METRIC_NAMES,
    TREND_CHART_METRIC_NAMES,
)
from pitchavatar_rag_sentinel.reporting.formatting import (
    cleanup_failed_text,
    compact_json,
    format_metric_value,
    status_text as _status_text,
)


def sort_runs_latest_first(runs: list[ArtifactRunRef]) -> list[ArtifactRunRef]:
    return sorted(
        runs,
        key=lambda run: (_run_latest_mtime(run), run.run_id),
        reverse=True,
    )


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


def ir_metric_rows(report: ArtifactRunReport) -> list[dict[str, str]]:
    return _named_value_rows(report.ir_metrics, IR_METRIC_NAMES)


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
    st.caption(
        "Read-only local artifact viewer and offline dataset draft builder. "
        "No RAG runs or cleanup actions are available here."
    )

    artifacts_root_text = st.sidebar.text_input("Artifacts root", value="artifacts/runs")
    artifacts_root = Path(artifacts_root_text)

    artifact_tab, trends_tab, dataset_builder_tab = st.tabs(
        ["Artifacts", "Trends", "Dataset Builder"]
    )
    with artifact_tab:
        _render_artifact_console(st, artifacts_root)
    with trends_tab:
        _render_trends_console(st, artifacts_root)
    with dataset_builder_tab:
        _render_dataset_builder(st)


def _render_artifact_console(st: Any, artifacts_root: Path) -> None:
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
    _render_report(st, report, selected_run, selected_dataset, key_prefix="artifact")


def _render_trends_console(st: Any, artifacts_root: Path) -> None:
    if not artifacts_root.exists():
        st.info(f"Artifact root does not exist: {artifacts_root}")
        return

    loader = ArtifactLoader(artifacts_root)
    history = loader.load_run_history()
    if not history:
        st.info(f"No artifact summaries were found under {artifacts_root}.")
        return

    dataset_options = trend_dataset_options(history)
    selected_datasets = st.multiselect(
        "Dataset filter",
        options=dataset_options,
        default=dataset_options,
        key="trends_dataset_filter",
    )
    filtered_history = filter_trend_history(history, selected_datasets)
    if not filtered_history:
        st.info("No runs match the selected dataset filter.")
        return

    st.subheader("Latest Status")
    st.dataframe(
        latest_dataset_status_rows(filtered_history),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Runs")
    st.dataframe(
        trend_table_rows(filtered_history, artifacts_root=artifacts_root),
        hide_index=True,
        use_container_width=True,
    )

    _render_trend_charts(st, filtered_history)

    selected_row = st.selectbox(
        "Run details",
        filtered_history,
        format_func=lambda row: (
            f"{row.created_at.isoformat()} | {row.dataset_id} | {row.run_id}"
        ),
        key="trends_run_details",
    )
    if selected_row is None:
        return

    report = loader.load_report(selected_row.artifact_dir)
    selected_dataset = ArtifactDatasetRef(
        run_id=selected_row.run_id,
        dataset_id=selected_row.dataset_id,
        path=selected_row.artifact_dir,
    )
    selected_run = ArtifactRunRef(
        run_id=selected_row.run_id,
        path=selected_row.artifact_dir.parent,
        datasets=[selected_dataset],
    )
    with st.expander("Selected Run Details"):
        _render_report(st, report, selected_run, selected_dataset, key_prefix="trends")


def _render_dataset_builder(st: Any) -> None:
    uploaded_file = st.file_uploader("Source file", type=["txt", "md", "pdf", "pptx"])
    if uploaded_file is None:
        st.info("Upload a .txt, .md, .pdf, or .pptx file to preview local sections.")
        return

    try:
        parsed_source = parse_source_bytes(
            uploaded_file.getvalue(),
            source_file_name=uploaded_file.name,
        )
    except ParserDependencyError as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    st.metric("Extracted characters", len(parsed_source.extracted_text))
    st.metric("Sections", len(parsed_source.sections))

    document_modes = ["file_as_document", "section_as_document"]
    default_document_mode = resolve_document_mode(parsed_source)
    document_mode = st.selectbox(
        "Document mode",
        options=document_modes,
        index=document_modes.index(default_document_mode),
        help=(
            "file_as_document creates one dataset document per uploaded file. "
            "section_as_document creates one dataset document per parsed section."
        ),
    )
    st.info(
        "For production-like PitchAvatar testing, use file_as_document for PDF/PPTX and "
        "validate retrieval with chunk-level expectations such as "
        "expected_top1_chunk_contains or expected_in_topk_chunk_contains. For controlled "
        "QA/debug tests, use section_as_document."
    )

    document_keys = document_keys_for_source(parsed_source, document_mode=document_mode)
    if document_keys:
        st.caption(f"Generated document keys: {', '.join(document_keys)}")

    section_rows: list[dict[str, Any]] = []
    section_document_keys = document_keys if document_mode == "section_as_document" else []
    for section_index, section in enumerate(parsed_source.sections):
        row = {
            "section_id": section.section_id,
            "title": section.title,
            "characters": section.character_count,
            "preview": _preview_text(section.text, 180),
        }
        if section_document_keys:
            row["document_key"] = section_document_keys[section_index]
        section_rows.append(row)

    if section_rows:
        st.dataframe(section_rows, hide_index=True, use_container_width=True)
    else:
        st.warning(_no_sections_message(uploaded_file.name))

    default_dataset_id = f"{_slugify(Path(uploaded_file.name).stem) or 'dataset'}_draft"
    dataset_id = st.text_input("dataset_id", value=default_dataset_id)
    query_count = st.number_input("Query drafts", min_value=1, max_value=5, value=1, step=1)
    document_options = ["", *document_keys]
    query_drafts: list[QueryDraft] = []

    for index in range(int(query_count)):
        with st.expander(f"Query {index + 1}", expanded=index == 0):
            query_id = st.text_input("query_id", key=f"builder_query_id_{index}")
            query = st.text_area("query", key=f"builder_query_{index}", height=80)
            alpha = st.selectbox(
                "alpha",
                options=[0.5, 0.0, 1.0],
                key=f"builder_alpha_{index}",
            )
            top_k = st.number_input(
                "top_k",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key=f"builder_top_k_{index}",
            )
            threshold = st.number_input(
                "threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key=f"builder_threshold_{index}",
            )
            expected_top1 = st.selectbox(
                "expected_top1",
                options=document_options,
                format_func=lambda value: value or "none",
                key=f"builder_expected_top1_{document_mode}_{index}",
            )
            expected_in_topk = st.multiselect(
                "expected_in_topk",
                options=document_keys,
                default=[expected_top1] if expected_top1 else [],
                key=f"builder_expected_in_topk_{document_mode}_{index}",
            )
            forbidden_docs = st.multiselect(
                "forbidden_docs",
                options=document_keys,
                key=f"builder_forbidden_docs_{document_mode}_{index}",
            )
            document_scope = st.multiselect(
                "document_scope",
                options=document_keys,
                help="Leave empty to use all generated documents.",
                key=f"builder_document_scope_{document_mode}_{index}",
            )
            expected_top1_fragments = st.text_area(
                "expected_top1_chunk_contains",
                help="One manually selected fragment per line.",
                key=f"builder_expected_top1_fragments_{index}",
            )
            expected_in_topk_fragments = st.text_area(
                "expected_in_topk_chunk_contains",
                help="One manually selected fragment per line.",
                key=f"builder_expected_in_topk_fragments_{index}",
            )
            expect_empty = st.checkbox("expect_empty", key=f"builder_expect_empty_{index}")

        if query.strip():
            query_drafts.append(
                QueryDraft(
                    query=query,
                    query_id=query_id.strip() or None,
                    alpha=alpha,
                    top_k=int(top_k),
                    threshold=float(threshold),
                    document_scope=document_scope or "all",
                    expectations=ExpectationDraft(
                        expected_top1=expected_top1 or None,
                        expected_in_topk=expected_in_topk,
                        forbidden_docs=forbidden_docs,
                        expected_top1_chunk_contains=_lines(expected_top1_fragments),
                        expected_in_topk_chunk_contains=_lines(expected_in_topk_fragments),
                        min_results=0 if expect_empty else 1,
                        expect_empty=expect_empty,
                    ),
                )
            )

    if not dataset_id.strip():
        st.warning("dataset_id is required before JSON can be generated.")
        return

    try:
        dataset = build_retrieval_dataset(
            dataset_id=dataset_id.strip(),
            parsed_source=parsed_source,
            query_drafts=query_drafts,
            document_mode=document_mode,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    dataset_json = dataset_to_pretty_json(dataset)
    st.code(dataset_json, language="json")
    st.download_button(
        "Download dataset JSON",
        data=dataset_json,
        file_name=f"{dataset.dataset_id}.json",
        mime="application/json",
    )


def trend_dataset_options(history: list[ArtifactRunHistoryRow]) -> list[str]:
    return sorted({row.dataset_id for row in history})


def filter_trend_history(
    history: list[ArtifactRunHistoryRow],
    selected_datasets: list[str],
) -> list[ArtifactRunHistoryRow]:
    if not selected_datasets:
        return []
    selected = set(selected_datasets)
    return [
        row
        for row in sort_run_history_latest_first(history)
        if row.dataset_id in selected
    ]


def latest_dataset_status_rows(
    history: list[ArtifactRunHistoryRow],
) -> list[dict[str, str | int]]:
    latest_rows = latest_run_by_dataset(history)
    return [
        {
            "dataset_id": row.dataset_id,
            "latest_run_id": row.run_id,
            "created_at": row.created_at.isoformat(),
            "run_passed": _status_text(row.run_passed),
            "all_queries_passed": _status_text(row.all_queries_passed),
            "cleanup_failed": cleanup_failed_text(row.cleanup_failed),
            "query_pass_rate": format_metric_value(row.query_pass_rate),
            "hit_rate_at_1": format_metric_value(row.hit_rate_at_1),
            "hit_rate_at_5": format_metric_value(row.hit_rate_at_5),
            "recall_at_5": format_metric_value(row.recall_at_5),
            "precision_at_5": format_metric_value(row.precision_at_5),
            "mrr": format_metric_value(row.mrr),
            "failed_queries": row.failed_queries,
            "total_queries": row.total_queries,
        }
        for row in latest_rows.values()
    ]


def trend_table_rows(
    history: list[ArtifactRunHistoryRow],
    *,
    artifacts_root: Path,
) -> list[dict[str, str | int]]:
    return [
        {
            "created_at": row.created_at.isoformat(),
            "dataset_id": row.dataset_id,
            "run_id": row.run_id,
            "run_passed": _status_text(row.run_passed),
            "all_queries_passed": _status_text(row.all_queries_passed),
            "cleanup_failed": cleanup_failed_text(row.cleanup_failed),
            "query_pass_rate": format_metric_value(row.query_pass_rate),
            "hit_rate_at_1": format_metric_value(row.hit_rate_at_1),
            "hit_rate_at_5": format_metric_value(row.hit_rate_at_5),
            "recall_at_5": format_metric_value(row.recall_at_5),
            "precision_at_5": format_metric_value(row.precision_at_5),
            "mrr": format_metric_value(row.mrr),
            "ndcg_at_5": format_metric_value(row.ndcg_at_5),
            "ndcg_at_10": format_metric_value(row.ndcg_at_10),
            "top1_document_accuracy": format_metric_value(row.top1_document_accuracy),
            "chunk_hit_rate_at_k": format_metric_value(row.chunk_hit_rate_at_k),
            "total_run_ms": format_metric_value(row.total_run_ms),
            "p95_search_ms": format_metric_value(row.p95_search_ms),
            "failed_queries": row.failed_queries,
            "total_queries": row.total_queries,
            "artifact_dir": _relative_path(row.artifact_dir, artifacts_root),
            "report_html": _relative_path(row.artifact_dir / "report.html", artifacts_root)
            if (row.artifact_dir / "report.html").is_file()
            else "",
        }
        for row in sort_run_history_latest_first(history)
    ]


def trend_chart_rows(
    history: list[ArtifactRunHistoryRow],
    metric_name: str,
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for row in sorted(history, key=lambda item: (item.created_at, item.dataset_id, item.run_id)):
        value = getattr(row, metric_name)
        if value is None:
            continue
        rows.append(
            {
                "created_at": row.created_at.isoformat(),
                "dataset_id": row.dataset_id,
                metric_name: value,
            }
        )
    return rows


def failed_query_chart_rows(
    history: list[ArtifactRunHistoryRow],
) -> list[dict[str, str | int]]:
    return [
        {
            "created_at": row.created_at.isoformat(),
            "dataset_id": row.dataset_id,
            "failed_queries": row.failed_queries,
        }
        for row in sorted(history, key=lambda item: (item.created_at, item.dataset_id, item.run_id))
    ]


def _render_trend_charts(st: Any, history: list[ArtifactRunHistoryRow]) -> None:
    import pandas as pd

    st.subheader("Metric Trends")
    for metric_name in TREND_CHART_METRIC_NAMES:
        rows = trend_chart_rows(history, metric_name)
        if not rows:
            st.info(f"No values available for {metric_name}.")
            continue
        st.write(metric_name)
        st.line_chart(
            pd.DataFrame(rows),
            x="created_at",
            y=metric_name,
            color="dataset_id",
        )

    rows = failed_query_chart_rows(history)
    if rows:
        st.write("failed_queries")
        st.line_chart(
            pd.DataFrame(rows),
            x="created_at",
            y="failed_queries",
            color="dataset_id",
        )


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _render_report(
    st: Any,
    report: ArtifactRunReport,
    selected_run: ArtifactRunRef,
    selected_dataset: ArtifactDatasetRef,
    *,
    key_prefix: str,
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

    if report.ir_metrics:
        st.subheader("IR Metrics")
        st.dataframe(ir_metric_rows(report), hide_index=True, use_container_width=True)

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
            key=f"{key_prefix}_query_detail",
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


def _has_any_metric(values: dict[str, Any], names: tuple[str, ...]) -> bool:
    return any(name in values for name in names)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _preview_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)].rstrip()}..."


def _no_sections_message(source_file_name: str) -> str:
    source_type = Path(source_file_name).suffix.lower()
    if source_type == ".pdf":
        return (
            "No text sections were extracted. PDF support is text-layer only; scanned PDFs "
            "require OCR, which is not part of the offline Dataset Builder."
        )
    if source_type == ".pptx":
        return "No slide text was extracted from this deck."
    return "No text sections were extracted from this file."


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return re.sub(r"_+", "_", slug)[:64].strip("_")


def _compact_json(value: Any) -> str:
    return compact_json(value, blank_empty=True)


if __name__ == "__main__":
    main()
