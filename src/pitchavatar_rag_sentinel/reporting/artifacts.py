from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pitchavatar_rag_sentinel.config import SentinelSettings

TIMING_METRIC_NAMES = (
    "total_run_ms",
    "seed_total_ms",
    "search_total_ms",
    "cleanup_total_ms",
    "p50_search_ms",
    "p95_search_ms",
    "max_search_ms",
)
RUN_HISTORY_METRIC_NAMES = (
    "query_pass_rate",
    "top1_document_accuracy",
    "document_hit_rate_at_k",
    "top1_chunk_match_rate",
    "chunk_hit_rate_at_k",
    "forbidden_doc_violation_rate",
    "forbidden_chunk_violation_rate",
    "total_run_ms",
    "p95_search_ms",
)


class ArtifactWriter:
    def __init__(self, settings: SentinelSettings) -> None:
        self._root = Path(settings.artifacts_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def prepare_run_dir(self, run_id: str, dataset_id: str) -> Path:
        run_dir = self._root / run_id / dataset_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path


@dataclass(frozen=True, slots=True)
class ArtifactDatasetRef:
    run_id: str
    dataset_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class ArtifactRunRef:
    run_id: str
    path: Path
    datasets: list[ArtifactDatasetRef] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class QueryArtifactReport:
    query_id: str
    passed: bool
    query_text: str | None
    result_count: int | None
    returned_document_ids: list[str]
    top_document_id: str | None
    failed_checks: list[dict[str, Any]]
    checks: list[dict[str, Any]]
    request: dict[str, Any]
    response: dict[str, Any]
    evaluation: dict[str, Any]
    artifact_path: str | None = None
    raw_artifact: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactRunReport:
    run_id: str
    dataset_id: str
    artifact_dir: Path
    run_passed: bool | None
    all_queries_passed: bool | None
    cleanup_failed: bool | None
    cleanup_warning: str | None
    metrics: dict[str, Any]
    timings: dict[str, Any]
    query_results: list[QueryArtifactReport]
    failed_query_results: list[QueryArtifactReport]
    cleanup_details: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ArtifactRunHistoryRow:
    run_id: str
    dataset_id: str
    artifact_dir: Path
    created_at: datetime
    created_at_source: str
    run_passed: bool | None
    all_queries_passed: bool | None
    cleanup_failed: bool | None
    query_pass_rate: float | None
    top1_document_accuracy: float | None
    document_hit_rate_at_k: float | None
    top1_chunk_match_rate: float | None
    chunk_hit_rate_at_k: float | None
    forbidden_doc_violation_rate: float | None
    forbidden_chunk_violation_rate: float | None
    total_run_ms: float | None
    p95_search_ms: float | None
    failed_queries: int
    total_queries: int


class ArtifactLoader:
    def __init__(self, artifacts_root: Path | str = Path("artifacts/runs")) -> None:
        self.artifacts_root = Path(artifacts_root)

    def list_runs(self) -> list[ArtifactRunRef]:
        return list_artifact_runs(self.artifacts_root)

    def list_datasets(self, run_path: Path | str) -> list[ArtifactDatasetRef]:
        return list_run_datasets(run_path)

    def load_report(self, artifact_dir: Path | str) -> ArtifactRunReport:
        return load_artifact_report(artifact_dir)

    def load_latest_report(self) -> ArtifactRunReport:
        return load_artifact_report(find_latest_artifact_dir(self.artifacts_root))

    def load_run_history(self) -> list[ArtifactRunHistoryRow]:
        return load_run_history(self.artifacts_root)


def list_artifact_runs(artifacts_root: Path | str = Path("artifacts/runs")) -> list[ArtifactRunRef]:
    root = Path(artifacts_root)
    if not root.exists():
        return []

    runs: list[ArtifactRunRef] = []
    for run_path in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        datasets = list_run_datasets(run_path)
        if datasets:
            runs.append(
                ArtifactRunRef(
                    run_id=run_path.name,
                    path=run_path,
                    datasets=datasets,
                )
            )
    return runs


def list_run_datasets(run_path: Path | str) -> list[ArtifactDatasetRef]:
    path = Path(run_path)
    if not path.exists():
        return []

    if (path / "summary.json").is_file():
        summary = load_summary(path)
        return [
            ArtifactDatasetRef(
                run_id=str(summary.get("run_id") or path.parent.name),
                dataset_id=str(summary.get("dataset_id") or path.name),
                path=path,
            )
        ]

    dataset_refs: list[ArtifactDatasetRef] = []
    for dataset_path in sorted(
        (child for child in path.iterdir() if child.is_dir() and (child / "summary.json").is_file()),
        key=lambda child: child.name,
    ):
        summary = load_summary(dataset_path)
        dataset_refs.append(
            ArtifactDatasetRef(
                run_id=str(summary.get("run_id") or path.name),
                dataset_id=str(summary.get("dataset_id") or dataset_path.name),
                path=dataset_path,
            )
        )
    return dataset_refs


def find_latest_artifact_dir(
    artifacts_root: Path | str = Path("artifacts/runs"),
) -> Path:
    dataset_refs = [
        dataset_ref
        for run_ref in list_artifact_runs(artifacts_root)
        for dataset_ref in run_ref.datasets
    ]
    if not dataset_refs:
        raise FileNotFoundError(f"No artifact summaries found under {Path(artifacts_root)}")
    return max(
        dataset_refs,
        key=lambda dataset_ref: (
            (dataset_ref.path / "summary.json").stat().st_mtime,
            str(dataset_ref.path),
        ),
    ).path


def find_summary_paths(artifacts_root: Path | str = Path("artifacts/runs")) -> list[Path]:
    root = Path(artifacts_root)
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("summary.json") if path.is_file()),
        key=lambda path: str(path),
    )


def load_run_history(
    artifacts_root: Path | str = Path("artifacts/runs"),
) -> list[ArtifactRunHistoryRow]:
    rows = []
    for summary_path in find_summary_paths(artifacts_root):
        try:
            rows.append(load_run_history_row(summary_path.parent))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
    return sort_run_history_latest_first(rows)


def sort_run_history_latest_first(
    rows: list[ArtifactRunHistoryRow],
) -> list[ArtifactRunHistoryRow]:
    return sorted(
        rows,
        key=lambda row: (row.created_at, row.run_id, row.dataset_id, str(row.artifact_dir)),
        reverse=True,
    )


def latest_run_by_dataset(
    rows: list[ArtifactRunHistoryRow],
) -> dict[str, ArtifactRunHistoryRow]:
    latest: dict[str, ArtifactRunHistoryRow] = {}
    for row in sort_run_history_latest_first(rows):
        latest.setdefault(row.dataset_id, row)
    return dict(sorted(latest.items(), key=lambda item: item[0]))


def load_run_history_row(artifact_dir: Path | str) -> ArtifactRunHistoryRow:
    path = Path(artifact_dir)
    summary = load_summary(path)
    metrics = _as_dict(summary.get("metrics"))
    query_results = _list_of_dicts(summary.get("query_results"))
    run_id = str(summary.get("run_id") or path.parent.name)
    dataset_id = str(summary.get("dataset_id") or path.name)
    created_at, created_at_source = _infer_created_at(summary, path, run_id)

    return ArtifactRunHistoryRow(
        run_id=run_id,
        dataset_id=dataset_id,
        artifact_dir=path,
        created_at=created_at,
        created_at_source=created_at_source,
        run_passed=_optional_bool(summary.get("run_passed")),
        all_queries_passed=_optional_bool(summary.get("all_queries_passed")),
        cleanup_failed=_optional_bool(summary.get("cleanup_failed")),
        query_pass_rate=_optional_float(metrics.get("query_pass_rate")),
        top1_document_accuracy=_optional_float(metrics.get("top1_document_accuracy")),
        document_hit_rate_at_k=_optional_float(metrics.get("document_hit_rate_at_k")),
        top1_chunk_match_rate=_optional_float(metrics.get("top1_chunk_match_rate")),
        chunk_hit_rate_at_k=_optional_float(metrics.get("chunk_hit_rate_at_k")),
        forbidden_doc_violation_rate=_optional_float(
            metrics.get("forbidden_doc_violation_rate")
        ),
        forbidden_chunk_violation_rate=_optional_float(
            metrics.get("forbidden_chunk_violation_rate")
        ),
        total_run_ms=_optional_float(metrics.get("total_run_ms")),
        p95_search_ms=_optional_float(metrics.get("p95_search_ms")),
        failed_queries=sum(1 for result in query_results if result.get("passed") is False),
        total_queries=len(query_results),
    )


def load_summary(artifact_dir: Path | str) -> dict[str, Any]:
    summary_path = Path(artifact_dir) / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found under {Path(artifact_dir)}")
    return _load_json(summary_path)


def load_query_artifacts(artifact_dir: Path | str) -> list[dict[str, Any]]:
    return [payload for _, payload in _load_query_payloads(Path(artifact_dir))]


def load_artifact_report(artifact_dir: Path | str) -> ArtifactRunReport:
    path = Path(artifact_dir)
    summary = load_summary(path)
    query_payloads = _load_query_payloads(path)
    query_results = _build_query_reports(summary, query_payloads)
    metrics = _as_dict(summary.get("metrics"))
    failed_query_results = [result for result in query_results if not result.passed]

    return ArtifactRunReport(
        run_id=str(summary.get("run_id") or path.parent.name),
        dataset_id=str(summary.get("dataset_id") or path.name),
        artifact_dir=path,
        run_passed=_optional_bool(summary.get("run_passed")),
        all_queries_passed=_optional_bool(summary.get("all_queries_passed")),
        cleanup_failed=_optional_bool(summary.get("cleanup_failed")),
        cleanup_warning=_optional_str(summary.get("cleanup_warning")),
        metrics=metrics,
        timings={name: metrics.get(name) for name in TIMING_METRIC_NAMES if name in metrics},
        query_results=query_results,
        failed_query_results=failed_query_results,
        cleanup_details=_list_of_dicts(summary.get("cleanup_results")),
        summary=summary,
    )


def _build_query_reports(
    summary: dict[str, Any],
    query_payloads: list[tuple[Path, dict[str, Any]]],
) -> list[QueryArtifactReport]:
    summary_results = _list_of_dicts(summary.get("query_results"))
    artifact_by_id = {
        str(payload.get("query_id") or path.stem): (path, payload)
        for path, payload in query_payloads
    }

    ordered_query_ids = [
        str(result.get("query_id"))
        for result in summary_results
        if result.get("query_id") is not None
    ]
    ordered_query_ids.extend(
        sorted(query_id for query_id in artifact_by_id if query_id not in ordered_query_ids)
    )

    reports: list[QueryArtifactReport] = []
    for query_id in ordered_query_ids:
        summary_result = next(
            (result for result in summary_results if str(result.get("query_id")) == query_id),
            {},
        )
        artifact_path, artifact_payload = artifact_by_id.get(query_id, (None, {}))
        reports.append(_build_query_report(query_id, summary_result, artifact_path, artifact_payload))
    return reports


def _build_query_report(
    query_id: str,
    summary_result: dict[str, Any],
    artifact_path: Path | None,
    artifact_payload: dict[str, Any],
) -> QueryArtifactReport:
    request = _as_dict(artifact_payload.get("request") or summary_result.get("request"))
    response = _as_dict(artifact_payload.get("response") or summary_result.get("response"))
    evaluation = _as_dict(artifact_payload.get("evaluation") or summary_result.get("evaluation"))
    response_results = _list_of_dicts(response.get("results"))
    checks = _list_of_dicts(evaluation.get("checks"))
    returned_document_ids = [
        str(document_id) for document_id in _as_list(evaluation.get("returned_document_ids"))
    ]
    top_document_id = _top_document_id(response_results, returned_document_ids)
    failed_checks = [check for check in checks if not bool(check.get("passed"))]

    artifact_path_text = str(artifact_path) if artifact_path else summary_result.get("artifact_path")
    return QueryArtifactReport(
        query_id=query_id,
        passed=bool(summary_result.get("passed", evaluation.get("passed", False))),
        query_text=_optional_str(request.get("query")),
        result_count=_result_count(evaluation, response_results, returned_document_ids),
        returned_document_ids=returned_document_ids,
        top_document_id=top_document_id,
        failed_checks=failed_checks,
        checks=checks,
        request=request,
        response=response,
        evaluation=evaluation,
        artifact_path=_optional_str(artifact_path_text),
        raw_artifact=artifact_payload,
    )


def _load_query_payloads(artifact_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    query_dir = artifact_dir / "queries"
    if not query_dir.is_dir():
        return []
    return [
        (query_path, _load_json(query_path))
        for query_path in sorted(query_dir.glob("*.json"), key=lambda path: path.name)
    ]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def _top_document_id(
    response_results: list[dict[str, Any]],
    returned_document_ids: list[str],
) -> str | None:
    if response_results:
        document_id = response_results[0].get("document_id")
        if document_id is not None:
            return str(document_id)
    if returned_document_ids:
        return returned_document_ids[0]
    return None


def _result_count(
    evaluation: dict[str, Any],
    response_results: list[dict[str, Any]],
    returned_document_ids: list[str],
) -> int | None:
    result_count = evaluation.get("result_count")
    if isinstance(result_count, int):
        return result_count
    if response_results:
        return len(response_results)
    if returned_document_ids:
        return len(returned_document_ids)
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _infer_created_at(
    summary: dict[str, Any],
    artifact_dir: Path,
    run_id: str,
) -> tuple[datetime, str]:
    for field_name in ("created_at", "started_at", "run_started_at", "timestamp"):
        created_at = _parse_timestamp(summary.get(field_name))
        if created_at is not None:
            return created_at, field_name

    created_at = _parse_timestamp_from_run_id(run_id)
    if created_at is not None:
        return created_at, "run_id"

    summary_path = artifact_dir / "summary.json"
    mtime = summary_path.stat().st_mtime if summary_path.exists() else artifact_dir.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc), "summary_mtime"


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return _parse_timestamp(int(text))
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_timestamp_from_run_id(run_id: str) -> datetime | None:
    digit_groups = [group for group in run_id.replace("_", "-").split("-") if group.isdigit()]
    for group in reversed(digit_groups):
        if 12 <= len(group) <= 17:
            return _parse_timestamp(int(group))
    return None
