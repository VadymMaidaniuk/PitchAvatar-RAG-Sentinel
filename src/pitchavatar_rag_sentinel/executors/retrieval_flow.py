from __future__ import annotations

import logging
import time
import traceback as traceback_module
from dataclasses import asdict, dataclass, field
from pathlib import Path

import grpc
from opensearchpy.exceptions import OpenSearchException

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


logger = logging.getLogger(__name__)


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
    cleanup_status: str
    cleanup_method: str
    cleanup_errors: list[dict[str, str | None]] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class DatasetRunSummary:
    dataset_id: str
    run_id: str
    run_passed: bool
    all_queries_passed: bool
    cleanup_failed: bool
    cleanup_warning: str | None
    seeded_documents: dict[str, SeededDocumentState]
    query_results: list[QueryExecutionResult]
    cleanup_results: list[CleanupResult]
    run_dir: str
    run_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "run_passed": self.run_passed,
            "all_queries_passed": self.all_queries_passed,
            "cleanup_failed": self.cleanup_failed,
            "cleanup_warning": self.cleanup_warning,
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
                try:
                    cleanup_results = self._cleanup_seeded_documents(list(seeded_documents.values()))
                except Exception as exc:  # noqa: BLE001
                    cleanup_error = self._cleanup_error_details("cleanup_loop", exc)
                    logger.exception("Cleanup loop failed before all results could be recorded.")
                    cleanup_results = [
                        CleanupResult(
                            runtime_document_id="__cleanup_loop__",
                            grpc_delete_attempted=False,
                            grpc_delete_success=False,
                            grpc_message=None,
                            fallback_cleanup_used=False,
                            cleanup_verified=False,
                            cleanup_status="failed",
                            cleanup_method="cleanup_loop",
                            cleanup_errors=[cleanup_error],
                            error=cleanup_error["error_repr"],
                        )
                    ]

        all_queries_passed = run_error is None and all(result.passed for result in query_results)
        cleanup_failed = any(not result.cleanup_verified for result in cleanup_results)
        cleanup_warning = self._cleanup_warning(cleanup_failed)
        run_passed = all_queries_passed and (
            not cleanup_failed or not self._settings.fail_on_cleanup_error
        )

        summary = DatasetRunSummary(
            dataset_id=dataset.dataset_id,
            run_id=run_id,
            run_passed=run_passed,
            all_queries_passed=all_queries_passed,
            cleanup_failed=cleanup_failed,
            cleanup_warning=cleanup_warning,
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

    def _cleanup_warning(self, cleanup_failed: bool) -> str | None:
        if not cleanup_failed:
            return None
        if self._settings.fail_on_cleanup_error:
            return (
                "Cleanup failed and RAG_SENTINEL_FAIL_ON_CLEANUP_ERROR=true, "
                "so the overall run is failed."
            )
        return (
            "Cleanup failed but RAG_SENTINEL_FAIL_ON_CLEANUP_ERROR=false, "
            "so retrieval results are preserved and the cleanup failure is reported as a warning."
        )

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
            logger.info(
                "Upserting dataset document %r as %r.",
                document.key,
                runtime_document_id,
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

            # Register the runtime document immediately after a successful upsert so cleanup still
            # covers it if visibility waits time out or the run is interrupted mid-seed.
            seeded_documents[document.key] = SeededDocumentState(
                key=document.key,
                runtime_document_id=runtime_document_id,
                indexed_chunk_count=0,
                metadata=document.metadata,
            )

            logger.info(
                "Waiting for document %r to become visible in OpenSearch.",
                runtime_document_id,
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
            logger.info(
                "Document %r is visible in OpenSearch with %s chunk(s).",
                runtime_document_id,
                indexed_chunk_count,
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
        logger.info("Executing query %r with scope %s.", query_case.query_id, document_scope)
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
            cleanup_method = "grpc_delete"
            cleanup_errors: list[dict[str, str | None]] = []

            try:
                logger.info("Cleaning up runtime document %r via gRPC delete.", document.runtime_document_id)
                response = self._rag_client.delete_document(document.runtime_document_id)
                grpc_delete_success = response.success
                grpc_message = response.message
                self._opensearch.wait_until_document_absent(
                    document.runtime_document_id,
                    timeout_seconds=self._settings.cleanup_wait_timeout_seconds,
                )
                cleanup_verified = True
                logger.info("Cleanup verified for runtime document %r.", document.runtime_document_id)
            except (grpc.RpcError, RuntimeError, TimeoutError, OpenSearchException) as exc:
                cleanup_errors.append(self._cleanup_error_details(cleanup_method, exc))
                logger.warning(
                    "Cleanup via gRPC failed for runtime document %r.",
                    document.runtime_document_id,
                    exc_info=True,
                )
                if self._settings.delete_fallback_to_opensearch:
                    cleanup_method = "opensearch_delete_by_query"
                    try:
                        fallback_cleanup_used = True
                        logger.info(
                            "Falling back to direct OpenSearch cleanup for runtime document %r.",
                            document.runtime_document_id,
                        )
                        self._opensearch.cleanup_document(document.runtime_document_id)
                        self._opensearch.wait_until_document_absent(
                            document.runtime_document_id,
                            timeout_seconds=self._settings.cleanup_wait_timeout_seconds,
                        )
                        cleanup_verified = True
                        logger.info(
                            "Fallback cleanup verified for runtime document %r.",
                            document.runtime_document_id,
                        )
                    except (RuntimeError, TimeoutError, OpenSearchException) as fallback_exc:
                        cleanup_errors.append(
                            self._cleanup_error_details(cleanup_method, fallback_exc)
                        )
                        logger.warning(
                            "Fallback OpenSearch cleanup failed for runtime document %r.",
                            document.runtime_document_id,
                            exc_info=True,
                        )
                    except Exception as fallback_exc:  # noqa: BLE001
                        cleanup_errors.append(
                            self._cleanup_error_details(cleanup_method, fallback_exc)
                        )
                        logger.exception(
                            "Unexpected fallback cleanup failure for runtime document %r.",
                            document.runtime_document_id,
                        )
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(self._cleanup_error_details(cleanup_method, exc))
                logger.exception(
                    "Unexpected cleanup failure for runtime document %r.",
                    document.runtime_document_id,
                )
                if self._settings.delete_fallback_to_opensearch:
                    cleanup_method = "opensearch_delete_by_query"
                    try:
                        fallback_cleanup_used = True
                        logger.info(
                            "Falling back to direct OpenSearch cleanup for runtime document %r.",
                            document.runtime_document_id,
                        )
                        self._opensearch.cleanup_document(document.runtime_document_id)
                        self._opensearch.wait_until_document_absent(
                            document.runtime_document_id,
                            timeout_seconds=self._settings.cleanup_wait_timeout_seconds,
                        )
                        cleanup_verified = True
                        logger.info(
                            "Fallback cleanup verified for runtime document %r.",
                            document.runtime_document_id,
                        )
                    except (RuntimeError, TimeoutError, OpenSearchException) as fallback_exc:
                        cleanup_errors.append(
                            self._cleanup_error_details(cleanup_method, fallback_exc)
                        )
                        logger.warning(
                            "Fallback OpenSearch cleanup failed for runtime document %r.",
                            document.runtime_document_id,
                            exc_info=True,
                        )
                    except Exception as fallback_exc:  # noqa: BLE001
                        cleanup_errors.append(
                            self._cleanup_error_details(cleanup_method, fallback_exc)
                        )
                        logger.exception(
                            "Unexpected fallback cleanup failure for runtime document %r.",
                            document.runtime_document_id,
                        )

            cleanup_status = "verified" if cleanup_verified else "failed"
            error = "; ".join(error_detail["error_repr"] or "" for error_detail in cleanup_errors)
            if not error:
                error = None

            cleanup_results.append(
                CleanupResult(
                    runtime_document_id=document.runtime_document_id,
                    grpc_delete_attempted=grpc_delete_attempted,
                    grpc_delete_success=grpc_delete_success,
                    grpc_message=grpc_message,
                    fallback_cleanup_used=fallback_cleanup_used,
                    cleanup_verified=cleanup_verified,
                    cleanup_status=cleanup_status,
                    cleanup_method=cleanup_method,
                    cleanup_errors=cleanup_errors,
                    error=error,
                )
            )

        return cleanup_results

    @staticmethod
    def _cleanup_error_details(method: str, exc: BaseException) -> dict[str, str | None]:
        traceback = None
        if exc.__traceback__ is not None:
            traceback = "".join(
                traceback_module.format_exception(type(exc), exc, exc.__traceback__)
            )
        return {
            "method": method,
            "error_type": f"{type(exc).__module__}.{type(exc).__name__}",
            "error_repr": repr(exc),
            "traceback": traceback,
        }
