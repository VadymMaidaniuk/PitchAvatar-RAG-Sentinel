from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SeedDocumentSpec(BaseModel):
    key: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    min_expected_chunks: int = 1


class MetadataFilterSpec(BaseModel):
    field: str
    values: list[str]


class QueryExpectations(BaseModel):
    expected_top1: str | None = None
    expected_in_topk: list[str] = Field(default_factory=list)
    forbidden_docs: list[str] = Field(default_factory=list)
    min_results: int = 1
    expect_empty: bool = False


class QueryCaseSpec(BaseModel):
    query_id: str
    query: str
    alpha: Literal[0.0, 0.5, 1.0]
    top_k: int = 10
    threshold: float = 0.3
    document_scope: Literal["all"] | list[str] = "all"
    filters: list[MetadataFilterSpec] = Field(default_factory=list)
    expectations: QueryExpectations = Field(default_factory=QueryExpectations)

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
        return self

