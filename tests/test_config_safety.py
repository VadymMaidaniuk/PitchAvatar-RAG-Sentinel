from __future__ import annotations

import pytest

from pitchavatar_rag_sentinel.config import SentinelSettings


def make_settings(**overrides: object) -> SentinelSettings:
    values = {
        "grpc_target": "qa-rag-service-dev.pitchavatar.com:443",
        "grpc_secure": True,
        "opensearch_url": "https://opensearch-dev.pitchavatar.com",
        "opensearch_username": "unused",
        "opensearch_password": "unused",
        "opensearch_write_alias": "qa-sentinel-write",
        "opensearch_read_alias": "qa-sentinel-read",
        "opensearch_physical_index": "qa-sentinel-000001",
        "opensearch_allowed_targets": (
            "qa-sentinel-write,qa-sentinel-read,qa-sentinel-000001"
        ),
        "auto_create_index": False,
        "namespace": "qa-test",
        "artifacts_dir": "unused",
    }
    values.update(overrides)
    return SentinelSettings(_env_file=None, **values)


def test_default_opensearch_fallback_cleanup_is_disabled() -> None:
    settings = make_settings()

    assert settings.delete_fallback_to_opensearch is False


def test_non_allowlisted_opensearch_target_fails_fast() -> None:
    with pytest.raises(ValueError, match="non-allowlisted OpenSearch target"):
        make_settings(opensearch_write_alias="unexpected-shared-index")
