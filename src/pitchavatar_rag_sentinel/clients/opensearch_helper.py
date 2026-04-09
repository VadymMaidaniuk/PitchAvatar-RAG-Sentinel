from __future__ import annotations

import logging
import time
from collections.abc import Generator
from urllib.parse import urlparse

from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import AuthorizationException

from pitchavatar_rag_sentinel.config import SentinelSettings


logger = logging.getLogger(__name__)


TEST_INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
        },
    },
    "mappings": {
        "dynamic": False,
        "properties": {
            "content": {"type": "text"},
            "contentVector": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "engine": "lucene",
                    "space_type": "cosinesimil",
                    "name": "hnsw",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16,
                    },
                },
            },
            "metadata": {
                "properties": {
                    "content_hash": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "indexed_at": {"type": "date"},
                    "type": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                },
            },
        },
    },
}


class OpenSearchHelper:
    def __init__(self, settings: SentinelSettings) -> None:
        self._settings = settings
        parsed = urlparse(settings.opensearch_url)
        self._client = OpenSearch(
            hosts=[settings.opensearch_url],
            http_auth=settings.opensearch_auth,
            use_ssl=parsed.scheme == "https",
            verify_certs=settings.opensearch_verify_certs,
            ssl_assert_hostname=settings.opensearch_verify_certs,
            ssl_show_warn=False,
        )

    @property
    def client(self) -> OpenSearch:
        return self._client

    @property
    def index_name(self) -> str:
        return self._settings.write_index_name

    @property
    def write_index_name(self) -> str:
        return self._settings.write_index_name

    @property
    def read_index_name(self) -> str:
        return self._settings.read_index_name

    @property
    def primary_index_name(self) -> str:
        return self._settings.primary_index_name

    def ensure_test_index(self) -> None:
        self._settings.assert_safe_index()
        try:
            if self._resources_ready():
                return
        except AuthorizationException:
            logger.warning(
                "Skipping OpenSearch target bootstrap because the current credentials cannot "
                "inspect index or alias existence. Direct query-based verification may still work."
            )
            return
        if not self._settings.auto_create_index:
            raise RuntimeError(
                "OpenSearch targets are not ready and auto creation is disabled. "
                f"write={self.write_index_name!r}, read={self.read_index_name!r}, "
                f"physical={self.primary_index_name!r}"
            )
        self._create_or_configure_targets()

    def refresh_index(self) -> None:
        try:
            self._client.indices.refresh(index=self.primary_index_name)
        except AuthorizationException:
            logger.warning(
                "Skipping OpenSearch refresh for %r because the current credentials do not have "
                "refresh permissions. Visibility checks will rely on eventual consistency.",
                self.primary_index_name,
            )
        time.sleep(self._settings.refresh_wait_seconds)

    def count_chunks_by_document_id(self, document_id: str) -> int:
        response = self._client.count(
            index=self.read_index_name,
            body={"query": {"term": {"metadata.document_id": document_id}}},
        )
        return int(response["count"])

    def get_chunks_by_document_id(self, document_id: str) -> list[dict]:
        response = self._client.search(
            index=self.read_index_name,
            body={
                "size": 10000,
                "query": {"term": {"metadata.document_id": document_id}},
            },
        )
        hits = [hit["_source"] for hit in response["hits"]["hits"]]
        hits.sort(key=lambda item: item.get("metadata", {}).get("chunk_index", -1))
        return hits

    def cleanup_document(self, document_id: str) -> None:
        self._client.delete_by_query(
            index=self.write_index_name,
            body={"query": {"term": {"metadata.document_id": document_id}}},
            params={"refresh": "true", "conflicts": "proceed"},
        )

    def wait_until_document_present(
        self,
        document_id: str,
        *,
        min_chunks: int = 1,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 1.0,
    ) -> int:
        deadline = time.monotonic() + timeout_seconds
        last_count = 0
        while time.monotonic() < deadline:
            self.refresh_index()
            last_count = self.count_chunks_by_document_id(document_id)
            if last_count >= min_chunks:
                return last_count
            time.sleep(poll_interval_seconds)
        raise TimeoutError(
            f"Document {document_id!r} was not indexed in time. Last chunk count: {last_count}."
        )

    def wait_until_document_absent(
        self,
        document_id: str,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self.refresh_index()
            if self.count_chunks_by_document_id(document_id) == 0:
                return
            time.sleep(poll_interval_seconds)
        raise TimeoutError(f"Document {document_id!r} was not deleted in time.")

    def bulk_insert_test_chunks(self, document_id: str, num_chunks: int) -> None:
        helpers.bulk(
            self._client,
            self._chunk_actions(document_id, num_chunks),
            index=self.write_index_name,
            refresh=True,
        )

    def _chunk_actions(self, document_id: str, num_chunks: int) -> Generator[dict, None, None]:
        for chunk_index in range(num_chunks):
            yield {
                "_index": self.write_index_name,
                "_id": f"{document_id}-chunk-{chunk_index:06d}",
                "_source": {
                    "content": (
                        f"Test chunk {chunk_index} of document {document_id}. "
                        "Sample content for delete workflow verification."
                    ),
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "test": "true",
                    },
                    "contentVector": [0.01] * 1536,
                },
            }

    def _resources_ready(self) -> bool:
        return (
            self._resource_exists(self.primary_index_name)
            and self._resource_exists(self.write_index_name)
            and self._resource_exists(self.read_index_name)
        )

    def _resource_exists(self, name: str) -> bool:
        return bool(
            self._client.indices.exists(index=name)
            or self._client.indices.exists_alias(name=name)
        )

    def _create_or_configure_targets(self) -> None:
        if not self._client.indices.exists(index=self.primary_index_name):
            self._client.indices.create(index=self.primary_index_name, body=TEST_INDEX_MAPPING)

        if self.write_index_name == self.primary_index_name and self.read_index_name == self.primary_index_name:
            return

        actions: list[dict] = []
        if not self._client.indices.exists_alias(name=self.write_index_name):
            actions.append(
                {
                    "add": {
                        "index": self.primary_index_name,
                        "alias": self.write_index_name,
                        "is_write_index": True,
                    }
                }
            )

        if not self._client.indices.exists_alias(name=self.read_index_name):
            actions.append(
                {
                    "add": {
                        "index": self.primary_index_name,
                        "alias": self.read_index_name,
                    }
                }
            )

        if actions:
            self._client.indices.update_aliases(body={"actions": actions})
