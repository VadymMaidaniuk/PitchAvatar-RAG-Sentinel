from __future__ import annotations

import pytest

from pitchavatar_rag_sentinel.clients.rag_client import RagServiceClient
from pitchavatar_rag_sentinel.config import SentinelSettings


@pytest.fixture()
def settings() -> SentinelSettings:
    return SentinelSettings(
        grpc_target="qa-rag-service-dev.pitchavatar.com:443",
        grpc_secure=True,
        grpc_server_name="qa-rag-service-dev.pitchavatar.com",
        opensearch_url="https://opensearch-dev.pitchavatar.com",
        opensearch_username="qa-user",
        opensearch_password="qa-pass",
        opensearch_write_alias="dev-rag-index-qa",
        opensearch_read_alias="dev-rag-read-qa",
        opensearch_physical_index="dev-rag-index-qa-000001",
    )


@pytest.fixture()
def client(settings: SentinelSettings) -> RagServiceClient:
    client = object.__new__(RagServiceClient)
    client._settings = settings
    return client


def test_build_upsert_request_accepts_logical_index_name(client: RagServiceClient) -> None:
    request = client._build_upsert_request(
        document_id="qa-smoke-001",
        content="Paris is the capital of France.",
        metadata={"user_id": "qa-user", "type": "txt"},
        index_name="qa-test",
    )

    assert request.document_id == "qa-smoke-001"
    assert request.index_name == "qa-test"
    assert request.document_content == "Paris is the capital of France."
    assert dict(request.metadata) == {"user_id": "qa-user", "type": "txt"}


@pytest.mark.parametrize(
    "server_side_target",
    ["dev-rag-index-qa", "dev-rag-read-qa", "dev-rag-index-qa-000001"],
)
def test_build_upsert_request_rejects_server_side_index_targets(
    client: RagServiceClient,
    server_side_target: str,
) -> None:
    with pytest.raises(ValueError, match="server-side only"):
        client._build_upsert_request(
            document_id="qa-smoke-001",
            content="Paris is the capital of France.",
            metadata={"user_id": "qa-user", "type": "txt"},
            index_name=server_side_target,
        )


def test_build_search_request_uses_document_ids_and_ignores_deprecated_index_name(
    client: RagServiceClient,
) -> None:
    request = client._build_search_request(
        query="What is the capital of France?",
        document_ids=["qa-smoke-001"],
        alpha=0.5,
        top_k=5,
        threshold=0.4,
        filters=[("user_id", ["qa-user"])],
    )

    assert request.query == "What is the capital of France?"
    assert list(request.document_ids) == ["qa-smoke-001"]
    assert request.top_k == 5
    assert request.alpha == 0.5
    assert request.threshold == pytest.approx(0.4)
    assert request.index_name == ""
    assert len(request.filters) == 1
    assert request.filters[0].field == "user_id"
    assert list(request.filters[0].values) == ["qa-user"]


@pytest.mark.parametrize(
    ("builder_name", "field_name"),
    [
        ("_build_delete_request", "DeleteIndexRequest.index_name"),
        ("_build_index_exists_request", "IndexExistsRequest.index_name"),
    ],
)
def test_legacy_index_name_fields_carry_document_id(
    client: RagServiceClient,
    builder_name: str,
    field_name: str,
) -> None:
    request = getattr(client, builder_name)("qa-smoke-001")
    assert request.index_name == "qa-smoke-001", field_name


@pytest.mark.parametrize(
    ("builder_name", "server_side_target"),
    [
        ("_build_delete_request", "dev-rag-index-qa"),
        ("_build_delete_request", "dev-rag-read-qa"),
        ("_build_delete_request", "dev-rag-index-qa-000001"),
        ("_build_index_exists_request", "dev-rag-index-qa"),
        ("_build_index_exists_request", "dev-rag-read-qa"),
        ("_build_index_exists_request", "dev-rag-index-qa-000001"),
    ],
)
def test_delete_and_index_exists_reject_server_side_targets(
    client: RagServiceClient,
    builder_name: str,
    server_side_target: str,
) -> None:
    with pytest.raises(ValueError, match="server-side only"):
        getattr(client, builder_name)(server_side_target)


def test_build_search_request_rejects_server_side_targets_in_document_scope(
    client: RagServiceClient,
) -> None:
    with pytest.raises(ValueError, match="server-side only"):
        client._build_search_request(
            query="What is the capital of France?",
            document_ids=["dev-rag-read-qa"],
            alpha=0.5,
        )
