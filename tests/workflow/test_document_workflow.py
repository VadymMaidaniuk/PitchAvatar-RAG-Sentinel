from __future__ import annotations

import pytest

from pitchavatar_rag_sentinel.clients.opensearch_helper import OpenSearchHelper
from pitchavatar_rag_sentinel.clients.rag_client import RagServiceClient


@pytest.mark.workflow
@pytest.mark.destructive
def test_replace_document_replaces_old_content(
    rag_client: RagServiceClient,
    opensearch_helper: OpenSearchHelper,
    make_document_id,
) -> None:
    document_id = make_document_id("replace")

    v1 = "First version of the document with some initial content about software testing."
    upsert_v1 = rag_client.upsert_content(
        document_id=document_id,
        content=v1,
        metadata={"user_id": "qa-workflow", "type": "txt"},
    )
    assert upsert_v1.success, upsert_v1.message
    opensearch_helper.wait_until_document_present(document_id, min_chunks=1)
    chunks_v1 = opensearch_helper.get_chunks_by_document_id(document_id)
    assert chunks_v1

    v2 = (
        "Completely replaced document about organic gardening. Composting kitchen scraps "
        "creates nutrient-rich soil. Crop rotation prevents soil depletion."
    )
    upsert_v2 = rag_client.upsert_content(
        document_id=document_id,
        content=v2,
        metadata={"user_id": "qa-workflow", "type": "txt"},
    )
    assert upsert_v2.success, upsert_v2.message
    opensearch_helper.wait_until_document_present(document_id, min_chunks=1)
    chunks_v2 = opensearch_helper.get_chunks_by_document_id(document_id)
    assert chunks_v2

    combined_v2 = " ".join(chunk["content"] for chunk in chunks_v2).lower()
    assert "gardening" in combined_v2
    assert "software testing" not in combined_v2


@pytest.mark.workflow
@pytest.mark.destructive
def test_delete_non_existent_document_is_graceful(
    rag_client: RagServiceClient,
    make_document_id,
) -> None:
    document_id = make_document_id("missing-delete")
    response = rag_client.delete_document(document_id)
    assert response.success is True
    message = response.message.lower()
    assert "0" in message or "no chunks" in message


@pytest.mark.workflow
@pytest.mark.destructive
def test_metadata_is_preserved_in_opensearch(
    rag_client: RagServiceClient,
    opensearch_helper: OpenSearchHelper,
    make_document_id,
) -> None:
    document_id = make_document_id("metadata")
    response = rag_client.upsert_content(
        document_id=document_id,
        content="Document with custom metadata for verification in OpenSearch.",
        metadata={
            "user_id": "user-42",
            "type": "docx",
            "custom_key": "custom_value",
        },
    )
    assert response.success, response.message
    opensearch_helper.wait_until_document_present(document_id, min_chunks=1)

    chunks = opensearch_helper.get_chunks_by_document_id(document_id)
    assert chunks
    first_metadata = chunks[0]["metadata"]
    assert first_metadata["document_id"] == document_id
    assert first_metadata["user_id"] == "user-42"
    assert first_metadata["type"] == "docx"
    assert first_metadata["custom_key"] == "custom_value"

