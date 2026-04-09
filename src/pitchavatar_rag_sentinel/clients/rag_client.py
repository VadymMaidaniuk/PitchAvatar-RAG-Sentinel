from __future__ import annotations

from collections.abc import Iterable, Mapping

import grpc

from pitchavatar_rag_sentinel.config import SentinelSettings
from pitchavatar_rag_sentinel.generated import rag_pb2, rag_pb2_grpc


class RagServiceClient:
    """Thin gRPC wrapper around the external RAG service contract."""

    def __init__(self, settings: SentinelSettings) -> None:
        self._settings = settings
        self._channel = self._build_channel(settings)
        self._stub = rag_pb2_grpc.RagServiceStub(self._channel)

    @staticmethod
    def _build_channel(settings: SentinelSettings) -> grpc.Channel:
        if settings.grpc_secure:
            credentials = grpc.ssl_channel_credentials()
            options: list[tuple[str, str]] = []
            if settings.grpc_server_name:
                options.append(("grpc.ssl_target_name_override", settings.grpc_server_name))
            return grpc.secure_channel(settings.grpc_target, credentials, options=options)
        return grpc.insecure_channel(settings.grpc_target)

    def close(self) -> None:
        self._channel.close()

    def upsert_content(
        self,
        document_id: str,
        content: str,
        metadata: Mapping[str, str] | None = None,
        *,
        index_name: str = "",
        timeout_seconds: float | None = None,
    ) -> rag_pb2.UploadDocumentResponse:
        request = self._build_upsert_request(
            document_id=document_id,
            content=content,
            metadata=metadata,
            index_name=index_name,
        )
        return self._stub.Upsert(
            request,
            timeout=timeout_seconds or self._settings.upsert_timeout_seconds,
        )

    def search(
        self,
        query: str,
        document_ids: Iterable[str],
        *,
        alpha: float,
        top_k: int | None = None,
        threshold: float | None = None,
        filters: Iterable[tuple[str, list[str]]] | None = None,
        timeout_seconds: float | None = None,
    ) -> rag_pb2.SearchResponse:
        request = self._build_search_request(
            query=query,
            alpha=alpha,
            document_ids=list(document_ids),
            top_k=top_k,
            threshold=threshold,
            filters=filters,
        )
        return self._stub.SearchWithThreshold(
            request,
            timeout=timeout_seconds or self._settings.grpc_timeout_seconds,
        )

    def delete_document(
        self,
        document_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> rag_pb2.DeleteIndexResponse:
        request = self._build_delete_request(document_id)
        return self._stub.DeleteIndex(
            request,
            timeout=timeout_seconds or self._settings.delete_timeout_seconds,
        )

    def index_exists(
        self,
        document_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> rag_pb2.IndexExistsResponse:
        request = self._build_index_exists_request(document_id)
        return self._stub.IndexExists(
            request,
            timeout=timeout_seconds or self._settings.grpc_timeout_seconds,
        )

    def _build_upsert_request(
        self,
        *,
        document_id: str,
        content: str,
        metadata: Mapping[str, str] | None = None,
        index_name: str = "",
    ) -> rag_pb2.UploadDocumentRequest:
        if index_name:
            # QA OpenSearch aliases and physical index names are server-side only and must not
            # be passed in gRPC payloads from the black-box client.
            self._assert_not_server_side_index_target(
                index_name,
                request_name="UploadDocumentRequest.index_name",
            )
        return rag_pb2.UploadDocumentRequest(
            document_id=document_id,
            index_name=index_name,
            document_content=content,
            metadata=dict(metadata or {}),
        )

    def _build_search_request(
        self,
        *,
        query: str,
        document_ids: Iterable[str],
        alpha: float,
        top_k: int | None = None,
        threshold: float | None = None,
        filters: Iterable[tuple[str, list[str]]] | None = None,
    ) -> rag_pb2.SearchWithThresholdRequest:
        scoped_document_ids = list(document_ids)
        for document_id in scoped_document_ids:
            self._assert_not_server_side_index_target(
                document_id,
                request_name="SearchWithThresholdRequest.document_ids[]",
            )

        request_filters = [
            rag_pb2.MetadataFilter(field=field, values=values)
            for field, values in (filters or [])
        ]
        # SearchWithThreshold.index_name is deprecated/unused in the current service contract,
        # so the black-box client always scopes search via document_ids only.
        return rag_pb2.SearchWithThresholdRequest(
            query=query,
            top_k=top_k or self._settings.search_top_k,
            alpha=alpha,
            threshold=threshold if threshold is not None else self._settings.search_threshold,
            document_ids=scoped_document_ids,
            filters=request_filters,
        )

    def _build_delete_request(self, document_id: str) -> rag_pb2.DeleteIndexRequest:
        self._assert_not_server_side_index_target(
            document_id,
            request_name="DeleteIndexRequest.index_name",
        )
        # For DeleteIndex, index_name actually carries document_id according to service docs.
        return rag_pb2.DeleteIndexRequest(index_name=document_id)

    def _build_index_exists_request(self, document_id: str) -> rag_pb2.IndexExistsRequest:
        self._assert_not_server_side_index_target(
            document_id,
            request_name="IndexExistsRequest.index_name",
        )
        # For IndexExists, index_name actually carries document_id according to service docs.
        return rag_pb2.IndexExistsRequest(index_name=document_id)

    def _assert_not_server_side_index_target(self, value: str, *, request_name: str) -> None:
        if value and value in self._server_side_index_targets:
            raise ValueError(
                f"{request_name} must not use configured OpenSearch target {value!r}. "
                "QA OpenSearch aliases and physical index names are server-side only."
            )

    @property
    def _server_side_index_targets(self) -> set[str]:
        return {
            target
            for target in {
                self._settings.index_name,
                self._settings.write_index_name,
                self._settings.read_index_name,
                self._settings.primary_index_name,
            }
            if target
        }
