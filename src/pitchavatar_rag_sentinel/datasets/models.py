from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SeedDocumentSpec(BaseModel):
    key: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    min_expected_chunks: int = 1


class MetadataFilterSpec(BaseModel):
    field: str
    values: list[str]


class QueryQrelSpec(BaseModel):
    document_key: str
    relevance: int = Field(ge=0, strict=True)
    chunk_id: str | None = None
    chunk_index: int | None = Field(default=None, ge=0)
    chunk_contains: list[str] = Field(default_factory=list)

    @field_validator("relevance", mode="before")
    @classmethod
    def validate_relevance_is_integer(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("relevance must be an integer greater than or equal to 0")
        return value

    @model_validator(mode="after")
    def validate_qrel(self) -> "QueryQrelSpec":
        if not self.document_key.strip():
            raise ValueError("qrels document_key must not be empty")
        if any(not fragment.strip() for fragment in self.chunk_contains):
            raise ValueError("qrels chunk_contains fragments must not be empty")
        return self


class QueryExpectations(BaseModel):
    expected_top1: str | None = None
    expected_in_topk: list[str] = Field(default_factory=list)
    forbidden_docs: list[str] = Field(default_factory=list)
    expected_top1_chunk_contains: list[str] = Field(default_factory=list)
    expected_in_topk_chunk_contains: list[str] = Field(default_factory=list)
    forbidden_chunk_contains: list[str] = Field(default_factory=list)
    min_results: int = 1
    expect_empty: bool = False

    @model_validator(mode="after")
    def validate_chunk_fragments(self) -> "QueryExpectations":
        for field_name in (
            "expected_top1_chunk_contains",
            "expected_in_topk_chunk_contains",
            "forbidden_chunk_contains",
        ):
            fragments = getattr(self, field_name)
            if any(not fragment.strip() for fragment in fragments):
                raise ValueError(f"{field_name} fragments must not be empty")
        return self


class QueryCaseSpec(BaseModel):
    query_id: str
    query: str
    alpha: Literal[0.0, 0.5, 1.0]
    top_k: int = 10
    threshold: float = 0.3
    document_scope: Literal["all"] | list[str] = "all"
    filters: list[MetadataFilterSpec] = Field(default_factory=list)
    expectations: QueryExpectations = Field(default_factory=QueryExpectations)
    qrels: list[QueryQrelSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_query_case(self) -> "QueryCaseSpec":
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        return self


class RetrievalDataset(BaseModel):
    dataset_id: str
    description: str
    documents: list[SeedDocumentSpec]
    queries: list[QueryCaseSpec]

    @model_validator(mode="after")
    def validate_dataset(self) -> "RetrievalDataset":
        document_keys = [document.key for document in self.documents]
        if len(document_keys) != len(set(document_keys)):
            raise ValueError("document keys must be unique within a dataset")

        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique within a dataset")

        document_key_set = set(document_keys)
        for query in self.queries:
            if query.document_scope != "all":
                missing_scope = sorted(set(query.document_scope) - document_key_set)
                if missing_scope:
                    raise ValueError(
                        f"query {query.query_id!r} references unknown document keys in scope: "
                        f"{missing_scope}"
                    )

            all_expectation_keys = (
                ([query.expectations.expected_top1] if query.expectations.expected_top1 else [])
                + query.expectations.expected_in_topk
                + query.expectations.forbidden_docs
            )
            missing_expectations = sorted(
                {key for key in all_expectation_keys if key not in document_key_set}
            )
            if missing_expectations:
                raise ValueError(
                    f"query {query.query_id!r} references unknown document keys in expectations: "
                    f"{missing_expectations}"
                )

            missing_qrels = sorted(
                {
                    qrel.document_key
                    for qrel in query.qrels
                    if qrel.document_key not in document_key_set
                }
            )
            if missing_qrels:
                raise ValueError(
                    f"query {query.query_id!r} references unknown document keys in qrels: "
                    f"{missing_qrels}"
                )
        return self
