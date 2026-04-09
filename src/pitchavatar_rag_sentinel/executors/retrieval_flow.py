from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import grpc

from pitchavatar_rag_sentinel.clients.opensearch_helper import OpenSearchHelper
from pitchavatar_rag_sentinel.clients.rag_client import RagServiceClient
from pitchavatar_rag_sentinel.config import SentinelSettings
from pitchavatar_rag_sentinel.datasets.models import QueryCaseSpec, RetrievalDataset
from pitchavatar_rag_sentinel.evaluators.retrieval import (
    RetrievalEvaluationResult,
    evaluate_retrieval_query,
)
from pitchavatar_rag_sentinel.reporting.artifacts import ArtifactWriter
from pitchavatar_rag_sentinel.utils.assertions import unique_document_ids
from pitchavatar_rag_sentinel.utils.naming import unique_document_id


@dataclass(slots=True)
class SeededDocumentState:
    key: str
    runtime_document_id: str
    indexed_chunk_count: int
    metadata: dict[str, str]


@dataclass(slots=True)
class QueryExecutionResult:
    query_id: str
    passed: bool
    latency_ms: float
    evaluation: RetrievalEvaluationResult
    request: dict
    response: dict
    artifact_path: str


@dataclass(slots=True)
class CleanupResult:
    runtime_document_id: str
    grpc_delete_attempted: bool
    grpc_delete_success: bool
    grpc_message: str | None
    fallback_cleanup_used: bool
    cleanup_verified: bool
    error: str | None = None


@dataclass(slots=True)
class DatasetRunSummary:
    dataset_id: str
    run_id: str
    all_queries_passed: bool
    seeded_documents: dict[str, SeededDocumentState]
    query_results: list[QueryExecutionResult]
    cleanup_results: list[CleanupResult]
    run_dir: str
    run_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "all_queries_passed": self.all_queries_passed,
            "seeded_documents": {
                key: asdict(state) for key, state in self.seeded_documents.items()
            },
            "query_results": [
                {
                    "query_id": result.query_id,
                    "passed": result.passed,
                    "latency_ms": result.latency_ms,
                    "evaluation": result.evaluation.to_dict(),
                    "request": result.request,
                    "response": result.response,
                    "artifact_path": result.artifact_path,
                }
                for result in self.query_results
            ],
            "cleanup_results": [asdict(result) for result in self.cleanup_results],
            "run_dir": self.run_dir,
            "run_error": self.run_error,
        }


class RetrievalFlowExecutor:
    def __init__(
        self,
        settings: SentinelSettings,
        rag_client: RagServiceClient,
        opensearch_helper: OpenSearchHelper,
        artifact_writer: ArtifactWriter,
    ) -> None:
        self._settings = settings
        self._rag_client = rag_client
        self._opensearch = opensearch_helper
        self._artifacts = artifact_writer

    def run_dataset(self, dataset: RetrievalDataset) -> DatasetRunSummary:
        run_id = unique_document_id(self._settings.namespace, dataset.dataset_id)
        run_dir = self._artifacts.prepare_run_dir(run_id, dataset.dataset_id)
        seeded_documents: dict[str, SeededDocumentState] = {}
        query_results: list[QueryExecutionResult] = []
        cleanup_results: list[CleanupResult] = []
        run_error: str | None = None
        original_error: Exception | None = None

        try:
            self._seed_documents(dataset, run_dir, seeded_documents)
            for query_case in dataset.queries:
                query_results.append(
                    self._execute_query(
                        dataset=dataset,
                        query_case=query_case,
                        seeded_documents=seeded_documents,
                        run_dir=run_dir,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            run_error = str(exc)
            original_error = exc
        finally:
            if seeded_documents:
                cleanup_results = self._cleanup_seeded_documents(list(seeded_documents.values()))

        summary = DatasetRunSummary(
            dataset_id=dataset.dataset_id,
            run_id=run_id,
            all_queries_passed=(run_error is None and all(result.passed for result in query_results)),
            seeded_documents=seeded_documents,
            query_results=query_results,
            cleanup_results=cleanup_results,
            run_dir=str(run_dir),
            run_error=run_error,
        )
        self._artifacts.write_json(run_dir / "summary.json", summary.to_dict())
        if original_error is not None:
            raise RuntimeError(
                f"Dataset run failed for {dataset.dataset_id!r}. Artifacts: {run_dir}"
            ) from original_error
        return summary

    def _seed_documents(
        self,
        dataset: RetrievalDataset,
        run_dir: Path,
        seeded_documents: dict[str, SeededDocumentState],
    ) -> None:
        for document in dataset.documents:
            runtime_document_id = unique_document_id(
                self._settings.namespace,
                f"{dataset.dataset_id}-{document.key}",
            )
            response = self._rag_client.upsert_content(
                document_id=runtime_document_id,
                content=document.content,
                metadata=document.metadata,
            )
            if not response.success:
                raise RuntimeError(
                    f"Failed to upsert document {document.key!r}: {response.message}"
                )

            indexed_chunk_count = self._opensearch.wait_until_document_present(
                runtime_document_id,
                min_chunks=document.min_expected_chunks,
                timeout_seconds=self._settings.upsert_timeout_seconds,
            )
            seeded_documents[document.key] = SeededDocumentState(
                key=document.key,
                runtime_document_id=runtime_document_id,
                indexed_chunk_count=indexed_chunk_count,
                metadata=document.metadata,
            )

        manifest = {
            "dataset_id": dataset.dataset_id,
            "documents": {
                key: asdict(value) for key, value in seeded_documents.items()
            },
        }
        self._artifacts.write_json(run_dir / "seed_manifest.json", manifest)

    def _execute_query(
        self,
        dataset: RetrievalDataset,
        query_case: QueryCaseSpec,
        seeded_documents: dict[str, SeededDocumentState],
        run_dir: Path,
    ) -> QueryExecutionResult:
        key_to_runtime_id = {
            key: value.runtime_document_id for key, value in seeded_documents.items()
        }

        if query_case.document_scope == "all":
            document_scope = list(key_to_runtime_id.values())
        else:
            document_scope = [key_to_runtime_id[key] for key in query_case.document_scope]

        request_payload = {
            "query": query_case.query,
            "alpha": query_case.alpha,
            "top_k": query_case.top_k,
            "threshold": query_case.threshold,
            "document_ids": document_scope,
            "filters": [filter_spec.model_dump() for filter_spec in query_case.filters],
        }

        start = time.perf_counter()
        response = self._rag_client.search(
            query=query_case.query,
            document_ids=document_scope,
            alpha=query_case.alpha,
            top_k=query_case.top_k,
            threshold=query_case.threshold,
            filters=[
                (filter_spec.field, filter_spec.values) for filter_spec in query_case.filters
            ],
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        response_payload = {
            "results": [
                {
                    "document_id": result.document_id,
                    "page_content": result.page_content,
                    "metadata": dict(result.metadata),
                    "score": result.score,
                }
                for result in response.results
            ]
        }
        returned_document_ids = unique_document_ids(response.results)
        evaluation = evaluate_retrieval_query(
            query_case=query_case,
            returned_document_ids=returned_document_ids,
            key_to_runtime_id=key_to_runtime_id,
        )

        artifact_path = self._artifacts.write_json(
            run_dir / "queries" / f"{query_case.query_id}.json",
            {
                "dataset_id": dataset.dataset_id,
                "query_id": query_case.query_id,
                "latency_ms": latency_ms,
                "request": request_payload,
                "response": response_payload,
                "evaluation": evaluation.to_dict(),
            },
        )

        return QueryExecutionResult(
            query_id=query_case.query_id,
            passed=evaluation.passed,
            latency_ms=latency_ms,
            evaluation=evaluation,
            request=request_payload,
            response=response_payload,
            artifact_path=str(artifact_path),
        )

    def _cleanup_seeded_documents(
        self,
        documents: list[SeededDocumentState],
    ) -> list[CleanupResult]:
        cleanup_results: list[CleanupResult] = []

        for document in documents:
            grpc_delete_attempted = True
            grpc_delete_success = False
            grpc_message: str | None = None
            fallback_cleanup_used = False
            cleanup_verified = False
            error: str | None = None

            try:
                response = self._rag_client.delete_document(document.runtime_document_id)
                grpc_delete_success = response.success
                grpc_message = response.message
                self._opensearch.wait_until_document_absent(
                    document.runtime_document_id,
                    timeout_seconds=self._settings.cleanup_wait_timeout_seconds,
                )
                cleanup_verified = True
            except (grpc.RpcError, RuntimeError, TimeoutError) as exc:
                error = str(exc)
                if self._settings.delete_fallback_to_opensearch:
                    try:
                        fallback_cleanup_used = True
                        self._opensearch.cleanup_document(document.runtime_document_id)
                        self._opensearch.wait_until_document_absent(
                            document.runtime_document_id,
                            timeout_seconds=self._settings.cleanup_wait_timeout_seconds,
                        )
                        cleanup_verified = True
                    except Exception as fallback_exc:  # noqa: BLE001
                        error = f"{error}; fallback_cleanup_error={fallback_exc}"

            cleanup_results.append(
                CleanupResult(
                    runtime_document_id=document.runtime_document_id,
                    grpc_delete_attempted=grpc_delete_attempted,
                    grpc_delete_success=grpc_delete_success,
                    grpc_message=grpc_message,
                    fallback_cleanup_used=fallback_cleanup_used,
                    cleanup_verified=cleanup_verified,
                    error=error,
                )
            )

        return cleanup_results
