# MVP Technical Specification

## Goal

Build Sentinel as a dataset-driven QA harness for PitchAvatar RAG testing on a deployed environment.

The first-stage product validates retrieval quality and data lifecycle with this flow:

`upsert dataset -> search -> evaluate retrieval -> cleanup`

## Context

The framework targets:

- gRPC endpoint of the deployed RAG service
- isolated OpenSearch environment dedicated to QA verification

Direct OpenSearch queries are valid for white-box QA verification when needed, but the OpenSearch
target names remain server-side configuration and must not be passed as gRPC payload values.

The first stage does not require UI automation and does not depend on internal service code.

## Primary use case

QA prepares a retrieval dataset, Sentinel seeds it through gRPC, executes search queries through gRPC,
verifies indexed state through OpenSearch, evaluates retrieval outcomes, stores artifacts, and removes
the seeded documents after the run.

## In scope

- external Python repo
- dataset schema and loader
- black-box gRPC client
- OpenSearch verify helper
- dataset execution flow
- retrieval evaluation
- artifact capture
- automatic cleanup

## Out of scope for stage 1

- browser testing
- Playwright
- answer-level LLM evaluation
- RAGAS as a mandatory dependency
- performance/load testing
- Redis-driven orchestration

## Functional requirements

### Config

Sentinel must support:

- gRPC target configuration
- OpenSearch target configuration
- QA namespace for runtime document ids
- artifact output directory
- cleanup fallback toggle

### Dataset layer

Sentinel must support retrieval datasets containing:

- documents to seed
- search queries
- query expectations

### Flow executor

The executor must:

1. load dataset
2. generate runtime document ids
3. upsert dataset documents
4. wait for indexing visibility
5. execute search queries
6. evaluate results
7. persist artifacts
8. delete seeded documents
9. verify cleanup

### Evaluation

The first stage must support:

- expected top1
- expected in top-k
- forbidden documents absent
- minimum result count
- explicit empty expectation

### Cleanup

Primary cleanup path:

- gRPC delete

Fallback cleanup path:

- direct OpenSearch delete-by-query when enabled

## Deliverables

- dataset schema
- baseline dataset
- retrieval flow executor
- artifact writer
- pytest retrieval suite
- CLI runner for a single dataset

## Acceptance criteria

- Sentinel can run at least one dataset end-to-end
- each run produces a persisted summary artifact
- seeded documents are cleaned up after execution
- retrieval pass/fail signal is derived from dataset expectations

## Phase 2 extension path

The same dataset flow will later expand to support:

- combined response endpoint that returns chunks plus LLM answer
- answer-level evaluators
- optional RAGAS integration
- larger regression corpora
- performance-oriented datasets
