from __future__ import annotations

from pathlib import Path

from pitchavatar_rag_sentinel.config import SentinelSettings
from pitchavatar_rag_sentinel.datasets.models import RetrievalDataset


def build_dataset_dry_run_plan(
    *,
    settings: SentinelSettings,
    dataset: RetrievalDataset,
    dataset_path: str | Path | None = None,
) -> dict:
    settings.assert_safe_index()
    run_id_format = f"{settings.namespace}-{dataset.dataset_id}-<timestamp-ms>-<uuid8>"

    return {
        "dry_run": True,
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "dataset_id": dataset.dataset_id,
        "document_count": len(dataset.documents),
        "query_count": len(dataset.queries),
        "grpc": {
            "target": settings.grpc_target,
            "secure": settings.grpc_secure,
            "server_name": settings.grpc_server_name,
        },
        "opensearch": {
            "url": settings.opensearch_url,
            "write_alias": settings.opensearch_write_alias,
            "read_alias": settings.opensearch_read_alias,
            "physical_index": settings.opensearch_physical_index,
            "write_target": settings.write_index_name,
            "read_target": settings.read_index_name,
            "primary_target": settings.primary_index_name,
            "allowed_targets": sorted(settings.opensearch_allowed_targets),
            "delete_fallback_enabled": settings.delete_fallback_to_opensearch,
        },
        "cleanup": {
            "fail_on_cleanup_error": settings.fail_on_cleanup_error,
        },
        "destructive_mode_enabled": False,
        "destructive_mode_reason": "dry-run does not call gRPC or mutate OpenSearch",
        "planned_run_id_format": run_id_format,
        "planned_artifact_dir_format": str(
            Path(settings.artifacts_dir) / run_id_format / dataset.dataset_id
        ),
    }
