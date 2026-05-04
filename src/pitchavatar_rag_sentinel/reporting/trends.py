from __future__ import annotations

import csv
import os
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any

from pitchavatar_rag_sentinel.reporting.artifacts import (
    ArtifactRunHistoryRow,
    latest_run_by_dataset,
    load_run_history,
    sort_run_history_latest_first,
)


TREND_CSV_COLUMNS = (
    "created_at",
    "run_id",
    "dataset_id",
    "artifact_dir",
    "run_passed",
    "all_queries_passed",
    "cleanup_failed",
    "query_pass_rate",
    "top1_document_accuracy",
    "document_hit_rate_at_k",
    "top1_chunk_match_rate",
    "chunk_hit_rate_at_k",
    "forbidden_doc_violation_rate",
    "forbidden_chunk_violation_rate",
    "total_run_ms",
    "p95_search_ms",
    "failed_queries",
    "total_queries",
    "report_path",
)


def write_trends_report(
    artifacts_root: Path | str = Path("artifacts/runs"),
    *,
    output_path: Path | str | None = None,
    csv_output_path: Path | str | None = None,
) -> tuple[Path, Path]:
    root = Path(artifacts_root)
    html_path = Path(output_path) if output_path else root / "trends_report.html"
    csv_path = Path(csv_output_path) if csv_output_path else html_path.with_suffix(".csv")
    history = load_run_history(root)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_trends_html(history, output_path=html_path), encoding="utf-8")
    write_trends_csv(history, csv_path)
    return html_path, csv_path


def write_trends_csv(
    history: list[ArtifactRunHistoryRow],
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TREND_CSV_COLUMNS)
        writer.writeheader()
        for row in sort_run_history_latest_first(history):
            writer.writerow(_csv_row(row, output_dir=path.parent))
    return path


def render_trends_html(
    history: list[ArtifactRunHistoryRow],
    *,
    output_path: Path | str | None = None,
) -> str:
    rows = sort_run_history_latest_first(history)
    output_dir = Path(output_path).parent if output_path else Path(".")
    datasets = sorted({row.dataset_id for row in rows})
    latest_rows = list(latest_run_by_dataset(rows).values())
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>RAG Sentinel Trends</title>",
            "<style>",
            _stylesheet(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            _render_header(rows, datasets),
            _render_latest_table(latest_rows, output_dir),
            _render_history_table(rows, output_dir),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _render_header(rows: list[ArtifactRunHistoryRow], datasets: list[str]) -> str:
    return f"""
<section class="panel">
  <p class="eyebrow">RAG Sentinel Trends</p>
  <h1>Artifact Run History</h1>
  <dl class="summary-grid">
    {_status_item("runs", len(rows))}
    {_status_item("datasets", len(datasets))}
    {_status_item("dataset_ids", ", ".join(datasets))}
  </dl>
</section>
""".strip()


def _render_latest_table(rows: list[ArtifactRunHistoryRow], output_dir: Path) -> str:
    if not rows:
        body = "<p>No artifact summaries were found.</p>"
    else:
        table_rows = "\n".join(_render_run_row(row, output_dir, compact=True) for row in rows)
        body = _table(
            headings=(
                "Dataset",
                "Latest Run",
                "Created",
                "Run",
                "Queries",
                "Cleanup",
                "Query Pass Rate",
                "Failed Queries",
                "Report",
            ),
            rows=table_rows,
        )
    return f"""
<section class="panel">
  <h2>Latest Status Per Dataset</h2>
  {body}
</section>
""".strip()


def _render_history_table(rows: list[ArtifactRunHistoryRow], output_dir: Path) -> str:
    if not rows:
        body = "<p>No runs to show.</p>"
    else:
        table_rows = "\n".join(_render_run_row(row, output_dir, compact=False) for row in rows)
        body = _table(
            headings=(
                "Dataset",
                "Run",
                "Created",
                "Run",
                "Queries",
                "Cleanup",
                "Query Pass Rate",
                "Top1 Doc",
                "Chunk Hit@K",
                "Total ms",
                "P95 Search ms",
                "Failed Queries",
                "Report",
            ),
            rows=table_rows,
        )
    return f"""
<section class="panel">
  <h2>Runs Latest First</h2>
  {body}
</section>
""".strip()


def _render_run_row(row: ArtifactRunHistoryRow, output_dir: Path, *, compact: bool) -> str:
    report_link = _report_link(row, output_dir)
    base_cells = [
        f"<td><code>{_html(row.dataset_id)}</code></td>",
        f"<td><code>{_html(row.run_id)}</code></td>",
        f"<td>{_html(_created_at_text(row))}</td>",
        f"<td>{_status_badge(row.run_passed)}</td>",
        f"<td>{_status_badge(row.all_queries_passed)}</td>",
        f"<td>{_cleanup_badge(row.cleanup_failed)}</td>",
        f"<td>{_html(_format_metric(row.query_pass_rate))}</td>",
    ]
    if compact:
        cells = [
            *base_cells,
            f"<td>{row.failed_queries}/{row.total_queries}</td>",
            f"<td>{report_link}</td>",
        ]
    else:
        cells = [
            *base_cells,
            f"<td>{_html(_format_metric(row.top1_document_accuracy))}</td>",
            f"<td>{_html(_format_metric(row.chunk_hit_rate_at_k))}</td>",
            f"<td>{_html(_format_metric(row.total_run_ms))}</td>",
            f"<td>{_html(_format_metric(row.p95_search_ms))}</td>",
            f"<td>{row.failed_queries}/{row.total_queries}</td>",
            f"<td>{report_link}</td>",
        ]
    return f"<tr>{''.join(cells)}</tr>"


def _csv_row(row: ArtifactRunHistoryRow, *, output_dir: Path) -> dict[str, Any]:
    payload = asdict(row)
    payload["artifact_dir"] = str(row.artifact_dir)
    payload["created_at"] = row.created_at.isoformat()
    payload["report_path"] = _relative_report_path(row, output_dir) or ""
    return {column: payload.get(column) for column in TREND_CSV_COLUMNS}


def _table(*, headings: tuple[str, ...], rows: str) -> str:
    heading_html = "".join(f"<th>{_html(heading)}</th>" for heading in headings)
    return f"""
<table>
  <thead><tr>{heading_html}</tr></thead>
  <tbody>
    {rows}
  </tbody>
</table>
""".strip()


def _report_link(row: ArtifactRunHistoryRow, output_dir: Path) -> str:
    relative_path = _relative_report_path(row, output_dir)
    if not relative_path:
        return ""
    return f'<a href="{_html(relative_path)}">report.html</a>'


def _relative_report_path(row: ArtifactRunHistoryRow, output_dir: Path) -> str | None:
    report_path = row.artifact_dir / "report.html"
    if not report_path.is_file():
        return None
    return os.path.relpath(report_path, output_dir)


def _created_at_text(row: ArtifactRunHistoryRow) -> str:
    return f"{row.created_at.isoformat()} ({row.created_at_source})"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _status_item(label: str, value: object) -> str:
    return (
        '<div class="summary-item">'
        f"<dt>{_html(label)}</dt>"
        f"<dd>{_html(value)}</dd>"
        "</div>"
    )


def _status_badge(value: bool | None) -> str:
    if value is True:
        return '<span class="badge pass">passed</span>'
    if value is False:
        return '<span class="badge fail">failed</span>'
    return '<span class="badge muted">not reported</span>'


def _cleanup_badge(value: bool | None) -> str:
    if value is True:
        return '<span class="badge fail">failed</span>'
    if value is False:
        return '<span class="badge pass">ok</span>'
    return '<span class="badge muted">not reported</span>'


def _html(value: object) -> str:
    if value is None:
        return ""
    return escape(str(value))


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
  width: min(1280px, calc(100% - 32px));
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
h2 {
  letter-spacing: 0;
  margin: 0 0 16px;
}
h1 {
  font-size: 28px;
}
h2 {
  font-size: 20px;
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
code {
  font-family: Consolas, "Liberation Mono", monospace;
}
.summary-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin: 0;
}
.summary-item {
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
.badge {
  border-radius: 999px;
  display: inline-block;
  font-size: 12px;
  font-weight: 700;
  padding: 3px 8px;
}
.badge.pass {
  background: #e5f6ed;
  color: #0d6b3f;
}
.badge.fail {
  background: #fde9e7;
  color: #a32020;
}
.badge.muted {
  background: #edf1f7;
  color: #5c6675;
}
a {
  color: #2458a6;
}
""".strip()
