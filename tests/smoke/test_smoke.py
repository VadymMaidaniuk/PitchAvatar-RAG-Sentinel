from __future__ import annotations

import time
from collections.abc import Callable, Generator

import pytest

from pitchavatar_rag_sentinel.clients.rag_client import RagServiceClient
from pitchavatar_rag_sentinel.config import SentinelSettings
from pitchavatar_rag_sentinel.utils.assertions import unique_document_ids
from pitchavatar_rag_sentinel.utils.naming import unique_document_id


def _wait_until(
    condition: Callable[[], bool],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
    failure_message: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(poll_interval_seconds)
    raise AssertionError(failure_message)


def _search_document_ids(
    rag_client: RagServiceClient,
    *,
    document_id: str,
    query: str,
) -> list[str]:
    response = rag_client.search(
        query=query,
        document_ids=[document_id],
        alpha=0.0,
        top_k=5,
        threshold=0.3,
    )
    return unique_document_ids(response.results)


def _wait_until_document_exists_via_grpc(
    rag_client: RagServiceClient,
    settings: SentinelSettings,
    document_id: str,
) -> None:
    _wait_until(
        lambda: rag_client.index_exists(document_id).exists,
        timeout_seconds=settings.upsert_timeout_seconds,
        poll_interval_seconds=1.0,
        failure_message=f"Document {document_id!r} did not become visible via IndexExists in time.",
    )


def _wait_until_document_absent_via_grpc(
    rag_client: RagServiceClient,
    settings: SentinelSettings,
    document_id: str,
) -> None:
    _wait_until(
        lambda: not rag_client.index_exists(document_id).exists,
        timeout_seconds=settings.cleanup_wait_timeout_seconds,
        poll_interval_seconds=1.0,
        failure_message=f"Document {document_id!r} was not deleted in time via gRPC.",
    )


def _wait_until_document_searchable_via_grpc(
    rag_client: RagServiceClient,
    settings: SentinelSettings,
    *,
    document_id: str,
    query: str,
) -> None:
    _wait_until(
        lambda: document_id in _search_document_ids(
            rag_client,
            document_id=document_id,
            query=query,
        ),
        timeout_seconds=settings.upsert_timeout_seconds,
        poll_interval_seconds=1.0,
        failure_message=f"Document {document_id!r} did not become searchable via gRPC in time.",
    )


@pytest.fixture(scope="session")
def settings() -> SentinelSettings:
    # Smoke tests are intentionally gRPC-only. We still provide non-placeholder OpenSearch
    # settings so the shared config model validates, but the module never calls OpenSearch.
    return SentinelSettings(
        opensearch_url="https://unused-opensearch.local",
        opensearch_username="unused",
        opensearch_password="unused",
        opensearch_write_alias="unused-opensearch-write",
        opensearch_read_alias="unused-opensearch-read",
        opensearch_physical_index="unused-opensearch-000001",
        opensearch_allowed_targets=[
            "unused-opensearch-write",
            "unused-opensearch-read",
            "unused-opensearch-000001",
        ],
    )


@pytest.fixture()
def tracked_documents(
    rag_client: RagServiceClient,
    settings: SentinelSettings,
) -> Generator[list[str], None, None]:
    document_ids: list[str] = []
    yield document_ids

    cleanup_errors: list[str] = []
    for document_id in document_ids:
        try:
            response = rag_client.delete_document(document_id)
            if not response.success:
                cleanup_errors.append(
                    f"Delete failed for {document_id!r}: {response.message}"
                )
                continue
            _wait_until_document_absent_via_grpc(rag_client, settings, document_id)
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"{document_id!r}: {exc}")

    if cleanup_errors:
        raise AssertionError("Smoke cleanup failed:\n" + "\n".join(cleanup_errors))


@pytest.fixture()
def make_document_id(
    settings: SentinelSettings,
    tracked_documents: list[str],
) -> Callable[[str], str]:
    def factory(slug: str) -> str:
        document_id = unique_document_id(settings.namespace, slug)
        tracked_documents.append(document_id)
        return document_id

    return factory


@pytest.mark.smoke
@pytest.mark.destructive
def test_upsert_search_delete_roundtrip(
    rag_client: RagServiceClient,
    settings: SentinelSettings,
    make_document_id,
) -> None:
    document_id = make_document_id("roundtrip")
    content = (
        "The pangolin is the world's most trafficked mammal. These scaly anteaters curl into "
        "a tight ball when threatened and use long sticky tongues to eat ants and termites."
    )

    upsert = rag_client.upsert_content(
        document_id=document_id,
        content=content,
        metadata={"user_id": "qa-smoke", "type": "txt"},
    )
    assert upsert.success, upsert.message

    _wait_until_document_exists_via_grpc(rag_client, settings, document_id)
    _wait_until_document_searchable_via_grpc(
        rag_client,
        settings,
        document_id=document_id,
        query="pangolin scaly anteater",
    )

    found_ids = _search_document_ids(
        rag_client,
        document_id=document_id,
        query="pangolin scaly anteater",
    )
    assert document_id in found_ids

    delete = rag_client.delete_document(document_id)
    assert delete.success, delete.message

    _wait_until_document_absent_via_grpc(rag_client, settings, document_id)

    exists_after_delete = rag_client.index_exists(document_id)
    assert exists_after_delete.exists is False


@pytest.mark.smoke
@pytest.mark.destructive
def test_index_exists_lifecycle(
    rag_client: RagServiceClient,
    settings: SentinelSettings,
    make_document_id,
) -> None:
    document_id = make_document_id("index-exists")

    before = rag_client.index_exists(document_id)
    assert before.exists is False

    upsert = rag_client.upsert_content(
        document_id=document_id,
        content="Lifecycle probe content for index existence verification.",
        metadata={"user_id": "qa-smoke", "type": "txt"},
    )
    assert upsert.success, upsert.message
    _wait_until_document_exists_via_grpc(rag_client, settings, document_id)

    during = rag_client.index_exists(document_id)
    assert during.exists is True

    delete = rag_client.delete_document(document_id)
    assert delete.success, delete.message
    _wait_until_document_absent_via_grpc(rag_client, settings, document_id)

    after = rag_client.index_exists(document_id)
    assert after.exists is False
