from __future__ import annotations

import grpc
import pytest

from pitchavatar_rag_sentinel.clients.opensearch_helper import OpenSearchHelper
from pitchavatar_rag_sentinel.clients.rag_client import RagServiceClient
from pitchavatar_rag_sentinel.config import SentinelSettings
from pitchavatar_rag_sentinel.utils.assertions import (
    assert_only_expected_document_ids,
    unique_document_ids,
)
from tests.fixtures.search_corpus import SearchCorpus, build_search_corpus

pytestmark = [pytest.mark.integration, pytest.mark.grpc, pytest.mark.opensearch]


@pytest.fixture()
def search_corpus(
    settings: SentinelSettings,
    rag_client: RagServiceClient,
    opensearch_helper: OpenSearchHelper,
    tracked_documents: list[str],
) -> SearchCorpus:
    corpus, documents = build_search_corpus(settings.namespace)
    for document in documents:
        tracked_documents.append(document.document_id)
        response = rag_client.upsert_content(
            document_id=document.document_id,
            content=document.content,
            metadata=document.metadata,
        )
        assert response.success, response.message
        opensearch_helper.wait_until_document_present(document.document_id, min_chunks=1)
    return corpus


@pytest.mark.search
@pytest.mark.destructive
def test_text_search_respects_document_scope(
    rag_client: RagServiceClient,
    search_corpus: SearchCorpus,
) -> None:
    response = rag_client.search(
        query="quantum computing qubits",
        document_ids=[search_corpus.doc_a, search_corpus.doc_b],
        alpha=1.0,
    )
    assert_only_expected_document_ids(
        unique_document_ids(response.results),
        [search_corpus.doc_a],
    )


@pytest.mark.search
@pytest.mark.destructive
def test_text_search_respects_metadata_filters(
    rag_client: RagServiceClient,
    search_corpus: SearchCorpus,
) -> None:
    response = rag_client.search(
        query="quantum",
        document_ids=search_corpus.core_ids(),
        alpha=1.0,
        filters=[("user_id", ["user-beta"])],
    )
    assert_only_expected_document_ids(
        unique_document_ids(response.results),
        [search_corpus.doc_c],
    )


@pytest.mark.search
@pytest.mark.destructive
def test_hybrid_search_pdf_filter_returns_quantum_docs(
    rag_client: RagServiceClient,
    search_corpus: SearchCorpus,
) -> None:
    response = rag_client.search(
        query="quantum",
        document_ids=search_corpus.core_ids(),
        alpha=0.5,
        filters=[("type", ["pdf"])],
    )
    assert_only_expected_document_ids(
        unique_document_ids(response.results),
        [search_corpus.doc_a, search_corpus.doc_c],
    )


@pytest.mark.search
@pytest.mark.destructive
def test_invalid_alpha_returns_invalid_argument(
    rag_client: RagServiceClient,
    search_corpus: SearchCorpus,
) -> None:
    with pytest.raises(grpc.RpcError) as error:
        rag_client.search(
            query="quantum",
            document_ids=[search_corpus.doc_a],
            alpha=0.3,
        )

    message = str(error.value)
    assert "InvalidArgument" in message
    assert "alpha must be" in message
