from __future__ import annotations

from collections.abc import Callable, Generator

import pytest

from pitchavatar_rag_sentinel.clients.opensearch_helper import OpenSearchHelper
from pitchavatar_rag_sentinel.clients.rag_client import RagServiceClient
from pitchavatar_rag_sentinel.config import SentinelSettings
from pitchavatar_rag_sentinel.utils.assertions import unique_document_ids
from pitchavatar_rag_sentinel.utils.naming import unique_document_id


@pytest.fixture(scope="session")
def opensearch_helper(settings: SentinelSettings) -> OpenSearchHelper:
    # White-box smoke checks use direct OpenSearch queries only and should not require index
    # bootstrap privileges in QA.
    return OpenSearchHelper(settings)


@pytest.fixture()
def tracked_documents(
    rag_client: RagServiceClient,
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
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"{document_id!r}: {exc}")

    if cleanup_errors:
        raise AssertionError("Smoke OpenSearch cleanup failed:\n" + "\n".join(cleanup_errors))


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
def test_upsert_search_delete_roundtrip_with_opensearch_visibility(
    rag_client: RagServiceClient,
    opensearch_helper: OpenSearchHelper,
    make_document_id,
) -> None:
    document_id = make_document_id("roundtrip-opensearch")
    content = (
        "The pangolin is the world's most trafficked mammal. These scaly anteaters curl into "
        "a tight ball when threatened and use long sticky tongues to eat ants and termites."
    )
    metadata = {"user_id": "qa-smoke", "type": "txt"}

    upsert = rag_client.upsert_content(
        document_id=document_id,
        content=content,
        metadata=metadata,
    )
    assert upsert.success, upsert.message

    chunk_count = opensearch_helper.wait_until_document_present(document_id, min_chunks=1)
    assert chunk_count > 0

    chunks = opensearch_helper.get_chunks_by_document_id(document_id)
    assert chunks
    first_metadata = chunks[0]["metadata"]
    assert first_metadata["document_id"] == document_id
    assert first_metadata["user_id"] == metadata["user_id"]
    assert first_metadata["type"] == metadata["type"]

    search = rag_client.search(
        query="pangolin scaly anteater",
        document_ids=[document_id],
        alpha=0.0,
        top_k=5,
        threshold=0.3,
    )
    found_ids = unique_document_ids(search.results)
    assert document_id in found_ids

    delete = rag_client.delete_document(document_id)
    assert delete.success, delete.message

    opensearch_helper.wait_until_document_absent(document_id)

    exists_after_delete = rag_client.index_exists(document_id)
    assert exists_after_delete.exists is False


@pytest.mark.smoke
@pytest.mark.destructive
def test_index_exists_matches_opensearch_chunk_lifecycle(
    rag_client: RagServiceClient,
    opensearch_helper: OpenSearchHelper,
    make_document_id,
) -> None:
    document_id = make_document_id("index-exists-opensearch")

    before = rag_client.index_exists(document_id)
    assert before.exists is False
    assert opensearch_helper.count_chunks_by_document_id(document_id) == 0

    upsert = rag_client.upsert_content(
        document_id=document_id,
        content="Lifecycle probe content for OpenSearch-backed existence verification.",
        metadata={"user_id": "qa-smoke", "type": "txt"},
    )
    assert upsert.success, upsert.message

    chunk_count = opensearch_helper.wait_until_document_present(document_id, min_chunks=1)
    assert chunk_count > 0

    during = rag_client.index_exists(document_id)
    assert during.exists is True
    assert opensearch_helper.count_chunks_by_document_id(document_id) > 0

    delete = rag_client.delete_document(document_id)
    assert delete.success, delete.message

    opensearch_helper.wait_until_document_absent(document_id)

    after = rag_client.index_exists(document_id)
    assert after.exists is False
    assert opensearch_helper.count_chunks_by_document_id(document_id) == 0
