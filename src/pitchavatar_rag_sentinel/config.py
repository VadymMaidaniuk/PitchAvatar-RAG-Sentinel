from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class SentinelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_SENTINEL_",
        case_sensitive=False,
        extra="ignore",
    )

    grpc_target: str = "localhost:50051"
    grpc_secure: bool = False
    grpc_server_name: str | None = None
    grpc_timeout_seconds: float = 30.0
    upsert_timeout_seconds: float = 120.0
    delete_timeout_seconds: float = 300.0

    opensearch_url: str = Field(default="")
    opensearch_username: str | None = None
    opensearch_password: str | None = None
    opensearch_verify_certs: bool = False

    # Deprecated single-target setting. Retained for backward compatibility.
    index_name: str = "dev-rag-index-sentinel"
    opensearch_write_alias: str | None = None
    opensearch_read_alias: str | None = None
    opensearch_physical_index: str | None = None
    opensearch_allowed_targets: Annotated[list[str], NoDecode] = Field(default_factory=list)
    auto_create_index: bool = True
    allow_protected_index: bool = False
    protected_index_pattern: str = r"^(dev|stage|prod)-rag-(index|read)(-\d{6})?$"

    namespace: str = "qa-sentinel"
    refresh_wait_seconds: float = 1.0
    cleanup_wait_timeout_seconds: float = 30.0
    search_top_k: int = 10
    search_threshold: float = 0.3
    artifacts_dir: str = "artifacts/runs"
    delete_fallback_to_opensearch: bool = False

    redis_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def apply_legacy_fallbacks(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        merged = dict(data)
        if not merged.get("grpc_target"):
            grpc_port = os.getenv("RAG_SERVICE_GRPC_SERVER_PORT")
            if grpc_port:
                merged["grpc_target"] = f"localhost:{grpc_port}"

        merged.setdefault("opensearch_url", os.getenv("RAG_SERVICE_VECTOR_DS_ADDRESS", ""))
        merged.setdefault("opensearch_username", os.getenv("RAG_SERVICE_OPENSEARCH_USERNAME"))
        merged.setdefault("opensearch_password", os.getenv("RAG_SERVICE_OPENSEARCH_PASSWORD"))
        merged.setdefault(
            "index_name",
            os.getenv("RAG_SERVICE_DEFAULT_INDEX_NAME", "dev-rag-index-sentinel"),
        )
        merged.setdefault(
            "opensearch_write_alias",
            os.getenv("RAG_SERVICE_DEFAULT_INDEX_NAME"),
        )
        merged.setdefault(
            "opensearch_read_alias",
            os.getenv("RAG_SERVICE_DEFAULT_SEARCH_INDEX_NAME"),
        )
        return merged

    @field_validator("opensearch_allowed_targets", mode="before")
    @classmethod
    def parse_opensearch_allowed_targets(cls, value: object) -> object:
        if isinstance(value, str):
            return [target.strip() for target in value.split(",") if target.strip()]
        return value

    @property
    def opensearch_auth(self) -> tuple[str, str] | None:
        if self.opensearch_username and self.opensearch_password:
            return self.opensearch_username, self.opensearch_password
        return None

    @property
    def write_index_name(self) -> str:
        return self.opensearch_write_alias or self.index_name

    @property
    def read_index_name(self) -> str:
        return self.opensearch_read_alias or self.write_index_name

    @property
    def primary_index_name(self) -> str:
        return self.opensearch_physical_index or self.write_index_name

    @model_validator(mode="after")
    def validate_runtime_targets(self) -> "SentinelSettings":
        if not self.grpc_target:
            raise ValueError("RAG_SENTINEL_GRPC_TARGET must be set.")
        if not self.opensearch_url:
            raise ValueError(
                "RAG_SENTINEL_OPENSEARCH_URL must be set, or provide "
                "RAG_SERVICE_VECTOR_DS_ADDRESS as a legacy fallback."
            )
        placeholder_values = {
            "RAG_SENTINEL_OPENSEARCH_USERNAME": self.opensearch_username or "",
            "RAG_SENTINEL_OPENSEARCH_PASSWORD": self.opensearch_password or "",
            "RAG_SENTINEL_OPENSEARCH_URL": self.opensearch_url,
        }
        for key, value in placeholder_values.items():
            normalized = value.strip().strip('"').strip("'").lower()
            if normalized.startswith("your-"):
                raise ValueError(
                    f"{key} still contains a template placeholder. "
                    "Replace it with a real environment-specific value."
                )
        if not self.write_index_name:
            raise ValueError(
                "Configure OpenSearch write target via RAG_SENTINEL_OPENSEARCH_WRITE_ALIAS "
                "or the legacy RAG_SENTINEL_INDEX_NAME."
            )
        if not self.read_index_name:
            raise ValueError(
                "Configure OpenSearch read target via RAG_SENTINEL_OPENSEARCH_READ_ALIAS "
                "or fall back to the write alias."
            )
        self.assert_safe_index()
        return self

    def configured_opensearch_targets(self) -> set[str]:
        return {
            target
            for target in {
                self.write_index_name,
                self.read_index_name,
                self.primary_index_name,
            }
            if target
        }

    def assert_safe_index(self) -> None:
        configured_targets = self.configured_opensearch_targets()
        allowed_targets = set(self.opensearch_allowed_targets)
        if not allowed_targets:
            raise ValueError(
                "SAFETY: refusing to run without RAG_SENTINEL_OPENSEARCH_ALLOWED_TARGETS. "
                "Explicitly allow every configured OpenSearch write/read/physical target."
            )
        non_allowlisted_targets = sorted(configured_targets - allowed_targets)
        if non_allowlisted_targets:
            raise ValueError(
                "SAFETY: refusing to run against non-allowlisted OpenSearch target(s): "
                f"{non_allowlisted_targets}. Configure RAG_SENTINEL_OPENSEARCH_ALLOWED_TARGETS "
                "with the exact QA index or alias names."
            )

        if self.allow_protected_index:
            return
        pattern = re.compile(self.protected_index_pattern)
        for target in configured_targets:
            if target and pattern.match(target):
                raise ValueError(
                    f"SAFETY: refusing to run against protected index {target!r}. "
                    "Override only if you are intentionally targeting a disposable environment."
                )


@lru_cache(maxsize=1)
def get_settings() -> SentinelSettings:
    settings = SentinelSettings()
    settings.assert_safe_index()
    return settings
