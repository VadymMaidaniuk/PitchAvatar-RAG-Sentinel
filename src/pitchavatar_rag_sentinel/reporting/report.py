from __future__ import annotations

from pathlib import Path
from typing import Any

from pitchavatar_rag_sentinel.reporting.constants import (
    IR_METRIC_NAMES,
    SUMMARY_METRIC_NAMES,
    TIMING_METRIC_NAMES,
)
from pitchavatar_rag_sentinel.reporting.artifacts import (
    ArtifactRunReport,
    QueryArtifactReport,
)
from pitchavatar_rag_sentinel.reporting.formatting import (
    compact_json,
    format_metric_value,
    html_escape,
    pretty_json,
    status_text,
)


def write_html_report(report: ArtifactRunReport, output_path: Path | str | None = None) -> Path:
    path = Path(output_path) if output_path else report.artifact_dir / "report.html"
    path.write_text(render_html_report(report), encoding="utf-8")
    return path


def render_html_report(report: ArtifactRunReport) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>RAG Sentinel Report - {_html(report.dataset_id)}</title>",
            "<style>",
            _stylesheet(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            _render_header(report),
            _render_metrics(report),
            _render_ir_metrics(report),
            _render_timings(report),
            _render_failed_queries(report.failed_query_results),
            _render_all_queries(report.query_results),
            _render_cleanup(report),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _render_header(report: ArtifactRunReport) -> str:
    if report.cleanup_failed is True:
        cleanup_status = "failed"
        cleanup_status_passed = False
    elif report.cleanup_failed is False:
        cleanup_status = "verified"
        cleanup_status_passed = True
    else:
        cleanup_status = "not reported"
        cleanup_status_passed = None
    return f"""
<section class="panel">
  <p class="eyebrow">RAG Sentinel Artifact Report</p>
  <h1>{_html(report.dataset_id)}</h1>
  <dl class="summary-grid">
    {_status_item("run_id", report.run_id)}
    {_status_item("dataset_id", report.dataset_id)}
    {_status_item("run_passed", _status_text(report.run_passed), report.run_passed)}
    {_status_item(
        "all_queries_passed",
        _status_text(report.all_queries_passed),
        report.all_queries_passed,
    )}
    {_status_item("cleanup_status", cleanup_status, cleanup_status_passed)}
  </dl>
</section>
""".strip()


def _render_metrics(report: ArtifactRunReport) -> str:
    return _render_metric_table(
        title="Metrics Summary",
        metrics=report.metrics,
        metric_names=SUMMARY_METRIC_NAMES,
        empty_text="No retrieval metrics were found in summary.json.",
    )


def _render_ir_metrics(report: ArtifactRunReport) -> str:
    if not report.ir_metrics:
        return ""
    return _render_metric_table(
        title="IR Metrics",
        metrics=report.ir_metrics,
        metric_names=IR_METRIC_NAMES,
        empty_text="No qrels-based IR metrics were found in summary.json.",
    )


def _render_timings(report: ArtifactRunReport) -> str:
    return _render_metric_table(
        title="Timing Summary",
        metrics=report.metrics,
        metric_names=TIMING_METRIC_NAMES,
        empty_text="No timing metrics were found in summary.json.",
        unit="ms",
    )


def _render_metric_table(
    *,
    title: str,
    metrics: dict[str, Any],
    metric_names: tuple[str, ...],
    empty_text: str,
    unit: str | None = None,
) -> str:
    rows = "\n".join(
        "<tr>"
        f"<th><code>{_html(name)}</code></th>"
        f"<td>{_html(_format_metric(metrics.get(name), unit=unit))}</td>"
        "</tr>"
        for name in metric_names
    )
    missing_note = f"<p>{_html(empty_text)}</p>" if not any(name in metrics for name in metric_names) else ""
    body = f"""
{missing_note}
<table>
  <tbody>
    {rows}
  </tbody>
</table>
""".strip()
    return f"""
<section class="panel">
  <h2>{_html(title)}</h2>
  {body}
</section>
""".strip()


def _render_failed_queries(failed_queries: list[QueryArtifactReport]) -> str:
    if not failed_queries:
        body = "<p>No failed queries.</p>"
    else:
        body = "\n".join(_render_failed_query(query) for query in failed_queries)
    return f"""
<section class="panel">
  <h2>Failed Queries</h2>
  {body}
</section>
""".strip()


def _render_failed_query(query: QueryArtifactReport) -> str:
    checks = _render_failed_checks(query.failed_checks)
    return f"""
<article class="query-failure">
  <h3>{_html(query.query_id)}</h3>
  <dl class="detail-grid">
    {_detail_item("query", query.query_text)}
    {_detail_item("actual_top_result", query.top_document_id)}
    {_detail_item("returned_results", query.result_count)}
  </dl>
  {checks}
  <details>
    <summary>Evaluation JSON</summary>
    <pre>{_html(_json(query.evaluation))}</pre>
  </details>
</article>
""".strip()


def _render_failed_checks(checks: list[dict[str, Any]]) -> str:
    if not checks:
        return "<p>No failed checks were recorded.</p>"
    rows = "\n".join(
        "<tr>"
        f"<td><code>{_html(check.get('name'))}</code></td>"
        f"<td>{_html(check.get('level', 'document'))}</td>"
        f"<td>{_html(_compact_json(check.get('expected_fragments')))}</td>"
        f"<td>{_html(check.get('details'))}</td>"
        f"<td>{_html(check.get('matched_document_id'))}</td>"
        f"<td>{_html(check.get('failure_reason'))}</td>"
        "</tr>"
        for check in checks
    )
    return f"""
<table>
  <thead>
    <tr>
      <th>Check</th>
      <th>Level</th>
      <th>Expected Values/Fragments</th>
      <th>Details</th>
      <th>Matched Document</th>
      <th>Failure Reason</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
""".strip()


def _render_all_queries(query_results: list[QueryArtifactReport]) -> str:
    if not query_results:
        body = "<p>No query results were found.</p>"
    else:
        rows = "\n".join(_render_query_row(query) for query in query_results)
        body = f"""
<table>
  <thead>
    <tr>
      <th>Query Id</th>
      <th>Status</th>
      <th>Returned Results</th>
      <th>Failed Checks</th>
      <th>Top Document Id</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
""".strip()
    return f"""
<section class="panel">
  <h2>All Queries</h2>
  {body}
</section>
""".strip()


def _render_query_row(query: QueryArtifactReport) -> str:
    failed_check_names = ", ".join(str(check.get("name")) for check in query.failed_checks)
    return (
        "<tr>"
        f"<td><code>{_html(query.query_id)}</code></td>"
        f"<td>{_badge(query.passed)}</td>"
        f"<td>{_html(query.result_count)}</td>"
        f"<td>{_html(failed_check_names or 'none')}</td>"
        f"<td>{_html(query.top_document_id)}</td>"
        "</tr>"
    )


def _render_cleanup(report: ArtifactRunReport) -> str:
    warning = (
        f"<p class=\"warning\">{_html(report.cleanup_warning)}</p>"
        if report.cleanup_warning
        else "<p>No cleanup warning was recorded.</p>"
    )
    if report.cleanup_details:
        rows = "\n".join(_render_cleanup_row(result) for result in report.cleanup_details)
        details = f"""
<table>
  <thead>
    <tr>
      <th>Runtime Document Id</th>
      <th>Status</th>
      <th>Method</th>
      <th>Verified</th>
      <th>Errors</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
""".strip()
    else:
        details = "<p>No cleanup details were found.</p>"
    return f"""
<section class="panel">
  <h2>Cleanup</h2>
  <dl class="detail-grid">
    {_detail_item("cleanup_failed", _status_text(report.cleanup_failed))}
  </dl>
  {warning}
  {details}
</section>
""".strip()


def _render_cleanup_row(result: dict[str, Any]) -> str:
    errors = result.get("cleanup_errors") or result.get("error")
    return (
        "<tr>"
        f"<td><code>{_html(result.get('runtime_document_id'))}</code></td>"
        f"<td>{_html(result.get('cleanup_status'))}</td>"
        f"<td>{_html(result.get('cleanup_method'))}</td>"
        f"<td>{_html(result.get('cleanup_verified'))}</td>"
        f"<td><pre>{_html(_compact_json(errors))}</pre></td>"
        "</tr>"
    )


def _status_item(label: str, value: object, status: bool | None = None) -> str:
    status_class = ""
    if status is True:
        status_class = " pass"
    elif status is False:
        status_class = " fail"
    return (
        f'<div class="summary-item{status_class}">'
        f"<dt>{_html(label)}</dt>"
        f"<dd>{_html(value)}</dd>"
        "</div>"
    )


def _detail_item(label: str, value: object) -> str:
    return f"<div><dt>{_html(label)}</dt><dd>{_html(value)}</dd></div>"


def _badge(passed: bool) -> str:
    label = "passed" if passed else "failed"
    css_class = "pass" if passed else "fail"
    return f'<span class="badge {css_class}">{label}</span>'


def _status_text(value: bool | None) -> str:
    return status_text(value)


def _format_metric(value: Any, *, unit: str | None = None) -> str:
    text = format_metric_value(value)
    return f"{text} {unit}" if unit and value is not None else text


def _compact_json(value: Any) -> str:
    return compact_json(value)


def _json(value: Any) -> str:
    return pretty_json(value)


def _html(value: object) -> str:
    return html_escape(value)


def _stylesheet() -> str:
    return """
:root {
  color-scheme: light;
  font-family: "Segoe UI", Arial, sans-serif;
  background: #f5f7fa;
  color: #202733;
}
body {
  margin: 0;
}
main {
  width: min(1120px, calc(100% - 32px));
  margin: 0 auto;
  padding: 24px 0 48px;
}
.panel {
  background: #ffffff;
  border: 1px solid #d8dee8;
  border-radius: 8px;
  margin: 0 0 16px;
  padding: 20px;
}
.eyebrow {
  color: #5c6675;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  margin: 0 0 8px;
  text-transform: uppercase;
}
h1,
h2,
h3 {
  letter-spacing: 0;
  margin: 0 0 16px;
}
h1 {
  font-size: 28px;
}
h2 {
  font-size: 20px;
}
h3 {
  font-size: 16px;
}
table {
  border-collapse: collapse;
  width: 100%;
}
th,
td {
  border-bottom: 1px solid #e4e9f1;
  padding: 10px 8px;
  text-align: left;
  vertical-align: top;
}
th {
  color: #435064;
  font-size: 13px;
}
code,
pre {
  font-family: Consolas, "Liberation Mono", monospace;
}
pre {
  margin: 0;
  overflow: auto;
  white-space: pre-wrap;
}
.summary-grid,
.detail-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin: 0;
}
.summary-item,
.detail-grid div {
  border: 1px solid #e1e6ee;
  border-radius: 6px;
  padding: 10px;
}
dt {
  color: #5c6675;
  font-size: 12px;
  font-weight: 700;
  margin: 0 0 6px;
}
dd {
  margin: 0;
  overflow-wrap: anywhere;
}
.pass {
  color: #0d6b3f;
}
.fail {
  color: #a32020;
}
.badge {
  border-radius: 999px;
  display: inline-block;
  font-size: 12px;
  font-weight: 700;
  padding: 3px 8px;
}
.badge.pass {
  background: #e5f6ed;
}
.badge.fail {
  background: #fde9e7;
}
.query-failure {
  border-top: 1px solid #e4e9f1;
  padding-top: 16px;
}
.query-failure + .query-failure {
  margin-top: 16px;
}
.warning {
  color: #8a4b00;
}
details {
  margin-top: 12px;
}
""".strip()
