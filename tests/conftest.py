from __future__ import annotations

from collections.abc import Callable, Generator

import pytest

from pitchavatar_rag_sentinel.clients.opensearch_helper import OpenSearchHelper
from pitchavatar_rag_sentinel.clients.rag_client import RagServiceClient
from pitchavatar_rag_sentinel.config import SentinelSettings, get_settings
from pitchavatar_rag_sentinel.executors.retrieval_flow import RetrievalFlowExecutor
from pitchavatar_rag_sentinel.reporting.artifacts import ArtifactWriter
from pitchavatar_rag_sentinel.utils.naming import unique_document_id


@pytest.fixture(scope="session")
def settings() -> SentinelSettings:
    return get_settings()


@pytest.fixture(scope="session")
def opensearch_helper(settings: SentinelSettings) -> OpenSearchHelper:
    helper = OpenSearchHelper(settings)
    helper.ensure_test_index()
    return helper


@pytest.fixture(scope="session")
def rag_client(settings: SentinelSettings) -> Generator[RagServiceClient, None, None]:
    client = RagServiceClient(settings)
    yield client
    client.close()


@pytest.fixture(scope="session")
def artifact_writer(settings: SentinelSettings) -> ArtifactWriter:
    return ArtifactWriter(settings)


@pytest.fixture(scope="session")
def retrieval_executor(
    settings: SentinelSettings,
    rag_client: RagServiceClient,
    opensearch_helper: OpenSearchHelper,
    artifact_writer: ArtifactWriter,
) -> RetrievalFlowExecutor:
    return RetrievalFlowExecutor(
        settings=settings,
        rag_client=rag_client,
        opensearch_helper=opensearch_helper,
        artifact_writer=artifact_writer,
    )


@pytest.fixture()
def tracked_documents(opensearch_helper: OpenSearchHelper) -> Generator[list[str], None, None]:
    document_ids: list[str] = []
    yield document_ids
    for document_id in document_ids:
        opensearch_helper.cleanup_document(document_id)
    opensearch_helper.refresh_index()


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
