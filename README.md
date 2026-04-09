# PitchAvatar RAG Sentinel

QA-owned dataset-driven harness for PitchAvatar RAG testing.

The current primary flow is:

`seed dataset -> search via gRPC -> evaluate retrieval -> cleanup seeded documents`

The framework talks to the RAG service as a black box over gRPC and uses OpenSearch only for:

- index visibility checks
- chunk verification
- cleanup verification
- emergency cleanup fallback
- optional direct white-box verification when QA needs to inspect indexed documents outside gRPC

OpenSearch target semantics:

- `write alias` is used for direct cleanup fallback and synthetic bulk seeding helpers
- `read alias` is used for verification queries and chunk inspection
- `physical index` is used for refresh and optional auto-create flows
- QA OpenSearch aliases and physical index names are server-side only and must not be sent in gRPC payloads
- manual QA verification confirmed that direct OpenSearch queries can find seeded documents in the isolated QA environment when white-box checks are needed

## Current concept

Sentinel is no longer centered on isolated endpoint tests.
The main unit of work is a retrieval dataset:

- documents to upsert
- queries to execute
- expectations for retrieved chunks or document ids

This lets QA build stable regression corpora first, and later extend the same flow with:

- LLM answer evaluation
- RAGAS or other eval layers
- performance datasets

## Quick start

```powershell
cd C:\Projects\PitchAvatar-RAG-Sentinel
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,report]"
Copy-Item .env.example .env
.venv\Scripts\python scripts\generate_proto.py
```

Important on Windows:

- do not run bare `pytest` unless the virtual environment is activated
- the safe form is `.venv\Scripts\python -m pytest ...`
- otherwise PowerShell may pick the global `pytest.exe`

## Preflight

Before the first real run:

- replace template values like `your-username` and `your-password` in `.env`
- set `RAG_SENTINEL_GRPC_TARGET` to the real QA/dev gRPC endpoint
- point `RAG_SENTINEL_OPENSEARCH_URL` to the dedicated QA/OpenSearch environment
- configure OpenSearch targets explicitly:
  - `RAG_SENTINEL_OPENSEARCH_WRITE_ALIAS`
  - `RAG_SENTINEL_OPENSEARCH_READ_ALIAS`
  - `RAG_SENTINEL_OPENSEARCH_PHYSICAL_INDEX` when known
- if your QA role only has direct `_search` access in OpenSearch, keep `RAG_SENTINEL_DELETE_FALLBACK_TO_OPENSEARCH=false` and rely on gRPC delete plus OpenSearch search-based verification
- if you run tiny corpora, confirm the RAG service itself is started with a BM25 threshold that will not filter out everything; QA verification confirmed `RAG_SERVICE_SEARCH_BM25_MIN_SCORE=0.1` works for white-box BM25 checks

Recommended QA values:

```env
RAG_SENTINEL_GRPC_TARGET=qa-rag-service-dev.pitchavatar.com:443
RAG_SENTINEL_GRPC_SECURE=true
RAG_SENTINEL_GRPC_SERVER_NAME=qa-rag-service-dev.pitchavatar.com

RAG_SENTINEL_OPENSEARCH_WRITE_ALIAS=dev-rag-index-qa
RAG_SENTINEL_OPENSEARCH_READ_ALIAS=dev-rag-read-qa
RAG_SENTINEL_OPENSEARCH_PHYSICAL_INDEX=dev-rag-index-qa-000001
```

## Layout

```text
datasets/
  retrieval/                    # JSON datasets for retrieval regression
proto/
  rag.proto
scripts/
  generate_proto.py
  run_dataset.py                # manual seed/search/evaluate/cleanup runner
src/pitchavatar_rag_sentinel/
  clients/                      # gRPC and OpenSearch clients
  datasets/                     # dataset schema and loader
  evaluators/                   # retrieval evaluation logic
  executors/                    # orchestration flows
  generated/                    # protobuf stubs
  reporting/                    # artifact writer
  utils/
tests/
  retrieval/                    # primary dataset-driven regression suite
  smoke/
  workflow/
  search/
```

## Main retrieval flow

The canonical MVP path is:

1. load a retrieval dataset
2. generate unique runtime document ids under the QA namespace
3. upsert dataset documents through gRPC
4. verify chunk indexing in OpenSearch
5. execute search queries through gRPC
6. evaluate returned document ids/chunks against dataset expectations
7. store request/response/evaluation artifacts
8. delete seeded documents through gRPC
9. verify cleanup in OpenSearch, with fallback cleanup if enabled

## Dataset shape

Each retrieval dataset defines:

- `dataset_id`
- `description`
- `documents`
- `queries`

Each query can declare:

- `query`
- `alpha`
- `top_k`
- `threshold`
- `document_scope`
- `filters`
- `expectations`

Expectations currently support:

- `expected_top1`
- `expected_in_topk`
- `forbidden_docs`
- `min_results`
- `expect_empty`

Example dataset:

- [quantum_baseline.json](C:/Projects/PitchAvatar-RAG-Sentinel/datasets/retrieval/quantum_baseline.json)

## Commands

Run the dataset suite:

```powershell
.venv\Scripts\python -m pytest tests\retrieval -vv
```

Run all retrieval-marked tests:

```powershell
.venv\Scripts\python -m pytest -m retrieval -vv
```

Run one dataset manually and print summary:

```powershell
.venv\Scripts\python scripts\run_dataset.py datasets\retrieval\quantum_baseline.json --summary
```

Run the gRPC-only smoke suite:

```powershell
.venv\Scripts\python -m pytest tests\smoke\test_smoke.py -vv
```

Run the currently accessible QA checks only:

```powershell
.\scripts\test.ps1 -Profile grpc
.\scripts\test.ps1 -Profile available
```

## Artifacts

Run artifacts are written under:

- `artifacts/runs/<run-id>/<dataset-id>/seed_manifest.json`
- `artifacts/runs/<run-id>/<dataset-id>/queries/<query-id>.json`
- `artifacts/runs/<run-id>/<dataset-id>/summary.json`

Artifacts contain:

- effective request payload
- gRPC response payload
- evaluation results
- seeded runtime document ids
- cleanup status

## Current scope

Primary:

- dataset-driven retrieval regression
- gRPC seed/search/delete lifecycle
- OpenSearch verify and cleanup
- artifact capture for QA analysis

Secondary legacy suites:

- smoke
- workflow
- endpoint-focused search checks

Recommended while QA access is limited to gRPC plus direct OpenSearch `_search`:

- `grpc` profile: local contract tests plus gRPC-only smoke
- `available` profile: `grpc` plus white-box OpenSearch search checks in `smoke`, `search`, and `workflow`
- full dataset retrieval runs stay available, but they are no longer the default recommendation for partial-access QA roles

## Notes

- `DeleteIndex` and `IndexExists` still use `index_name` as `document_id` in the current service contract.
- `SearchWithThreshold.index_name` is deprecated/unused and Sentinel scopes retrieval through `document_ids`.
- `UploadDocumentRequest.index_name` must never carry `dev-rag-index-qa`, `dev-rag-read-qa`, or the physical QA index.
- Direct OpenSearch verification is valid for QA diagnostics and retrieval checks, but those aliases/index names remain OpenSearch-side only and must not leak into gRPC payloads.
- The framework protects against accidental use of production-like indices by default.
- Sentinel supports separate OpenSearch `write/read` aliases for the QA setup.
- `Playwright` is not needed for this stage.
- `RAGAS` is not the core of the framework, but the structure now leaves room to add answer-level evaluation later.
