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

Install optional UI dependencies when you want the local artifact console:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,report,ui]"
```

## Preflight

Before the first real run:

- replace template values like `your-username` and `your-password` in `.env`
- set `RAG_SENTINEL_GRPC_TARGET` to the real QA/dev gRPC endpoint
- point `RAG_SENTINEL_OPENSEARCH_URL` to the dedicated QA/OpenSearch environment
- configure OpenSearch targets explicitly:
  - `RAG_SENTINEL_OPENSEARCH_WRITE_ALIAS`
  - `RAG_SENTINEL_OPENSEARCH_READ_ALIAS`
  - `RAG_SENTINEL_OPENSEARCH_PHYSICAL_INDEX` when known
- allow every configured OpenSearch target explicitly with `RAG_SENTINEL_OPENSEARCH_ALLOWED_TARGETS`; Sentinel refuses to run when any write/read/physical target is missing from that allowlist
- if your QA role only has direct `_search` access in OpenSearch, keep `RAG_SENTINEL_DELETE_FALLBACK_TO_OPENSEARCH=false` and rely on gRPC delete plus OpenSearch search-based verification
- keep `RAG_SENTINEL_FAIL_ON_CLEANUP_ERROR=true` unless you explicitly want retrieval evaluation to pass while reporting cleanup failures as warnings
- if you run tiny corpora, confirm the RAG service itself is started with a BM25 threshold that will not filter out everything; QA verification confirmed `RAG_SERVICE_SEARCH_BM25_MIN_SCORE=0.1` works for white-box BM25 checks

Recommended QA values:

```env
RAG_SENTINEL_GRPC_TARGET=qa-rag-service-dev.pitchavatar.com:443
RAG_SENTINEL_GRPC_SECURE=true
RAG_SENTINEL_GRPC_SERVER_NAME=qa-rag-service-dev.pitchavatar.com

RAG_SENTINEL_OPENSEARCH_WRITE_ALIAS=dev-rag-index-qa
RAG_SENTINEL_OPENSEARCH_READ_ALIAS=dev-rag-read-qa
RAG_SENTINEL_OPENSEARCH_PHYSICAL_INDEX=dev-rag-index-qa-000001
RAG_SENTINEL_OPENSEARCH_ALLOWED_TARGETS=dev-rag-index-qa,dev-rag-read-qa,dev-rag-index-qa-000001
RAG_SENTINEL_DELETE_FALLBACK_TO_OPENSEARCH=false
RAG_SENTINEL_FAIL_ON_CLEANUP_ERROR=true
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

Document-level checks:

- `expected_top1`
- `expected_in_topk`
- `forbidden_docs` (reported in artifacts as `forbidden_docs_absent`)
- `min_results`
- `expect_empty`

Chunk-level checks:

- `expected_top1_chunk_contains`: the top returned chunk must contain every listed fragment
- `expected_in_topk_chunk_contains`: at least one returned chunk must contain every listed fragment
- `forbidden_chunk_contains`: no returned chunk may contain any listed fragment

Chunk checks are deterministic text checks, not LLM evaluation. Matching is case-insensitive
and normalizes whitespace before comparing fragments.

Example chunk-level expectation:

```json
{
  "query_id": "q_chunk_scope",
  "query": "What does the runbook say about rollback?",
  "alpha": 0.5,
  "expectations": {
    "expected_top1": "doc_runbook",
    "expected_top1_chunk_contains": ["rollback window", "approval"],
    "forbidden_chunk_contains": ["deprecated procedure"]
  }
}
```

Example dataset:

- [quantum_baseline.json](C:/Projects/PitchAvatar-RAG-Sentinel/datasets/retrieval/quantum_baseline.json)

Dataset strategy and categories:

- [docs/datasets.md](C:/Projects/PitchAvatar-RAG-Sentinel/docs/datasets.md)
- [docs/dataset_builder.md](C:/Projects/PitchAvatar-RAG-Sentinel/docs/dataset_builder.md)
- [docs/smoke_calibration_report.md](C:/Projects/PitchAvatar-RAG-Sentinel/docs/smoke_calibration_report.md)

## Commands

Run the offline suite used by CI. This does not require real gRPC or OpenSearch:

```powershell
.venv\Scripts\python -m pytest -m "not integration and not destructive" -vv
```

Equivalent local profile:

```powershell
.\scripts\test.ps1 -Profile offline
```

Run the full dataset suite against configured services. These tests mutate the QA environment and are marked `integration`, `grpc`, `opensearch`, and `destructive`:

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

Validate a dataset run plan without gRPC/OpenSearch calls:

```powershell
.venv\Scripts\python scripts\run_dataset.py datasets\retrieval\quantum_baseline.json --dry-run
```

Preview local `.txt` or `.md` source sections for a draft dataset:

```powershell
.venv\Scripts\python scripts\preview_dataset_source.py path\to\source.md
```

Generate a read-only HTML report from existing artifacts:

```powershell
.venv\Scripts\python scripts\generate_report.py --run-dir artifacts\runs\<run-id>\<dataset-id>
Invoke-Item artifacts\runs\<run-id>\<dataset-id>\report.html
```

Generate a report for the newest artifact directory under `artifacts/runs`:

```powershell
.venv\Scripts\python scripts\generate_report.py --latest
```

Launch the read-only Streamlit console for local artifacts:

```powershell
streamlit run apps/sentinel_console.py
```

or without activating the virtual environment:

```powershell
.venv\Scripts\streamlit.exe run apps\sentinel_console.py
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
- cleanup warnings and structured cleanup errors when cleanup fails
- deterministic aggregate metrics in `summary.json`

By default `RAG_SENTINEL_FAIL_ON_CLEANUP_ERROR=true`, so a final cleanup failure makes the
overall run fail after `summary.json` is written. Set it to `false` only when you want retrieval
results to pass while cleanup failures are preserved as warnings in the summary artifact.

## Artifact reports

`scripts/generate_report.py` builds a static `report.html` inside an existing dataset artifact
directory. It is read-only: it loads `summary.json` and `queries/*.json`, then renders run status,
metrics, timing metrics, failed query details, all query results, and cleanup warnings/errors.

`apps/sentinel_console.py` provides a read-only Streamlit console over the same local artifacts.
It lets QA select an artifact root, run, and dataset, then browse summary fields, metrics, timing
metrics, failed queries, all queries, query detail JSON, and cleanup details.

The report viewer does not start real RAG runs, does not call gRPC or OpenSearch, and does not
perform cleanup or any destructive action. It is intended for sharing and inspecting already
captured QA artifacts.

Future reporting work can build on the same loader for a trend dashboard, dataset builder, and
real run launcher with explicit safety controls.

## Current metrics

`summary.json` includes a `metrics` object with deterministic QA regression metrics calculated
from explicit retrieval expectations and their check results. These metrics cover query pass rate,
document top-1 accuracy, document hit rate at k, forbidden document violations, empty-query checks,
chunk top-1 match rate, chunk hit rate at k, forbidden chunk violations, and basic run/search
timings when a real run executes.

These are not `pytrec_eval` metrics yet, not Ragas metrics yet, and not LLM-as-judge scores. They
are intended for regression reporting now and future dashboard visualization later. Rates are
reported as floats from `0.0` to `1.0`; when no explicit checks apply, the corresponding rate is
`null`.

Status fields differ intentionally:

- `all_queries_passed`: retrieval expectations only.
- `run_passed`: final run status, including cleanup policy.
- `metrics.query_pass_rate`: aggregate ratio of passed query evaluations to total query evaluations.

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
- CI and local offline checks should use `pytest -m "not integration and not destructive"` so real service tests do not run accidentally.
- retrieval smoke datasets are stable health checks; diagnostic and precision datasets are for investigation or stricter retrieval behavior checks.
- `alpha=1.0` is not used as the stable smoke default until backend `SearchWithThreshold` semantics are confirmed.

## Notes

- `DeleteIndex` and `IndexExists` still use `index_name` as `document_id` in the current service contract.
- `SearchWithThreshold.index_name` is deprecated/unused and Sentinel scopes retrieval through `document_ids`.
- `UploadDocumentRequest.index_name` must never carry `dev-rag-index-qa`, `dev-rag-read-qa`, or the physical QA index.
- Direct OpenSearch verification is valid for QA diagnostics and retrieval checks, but those aliases/index names remain OpenSearch-side only and must not leak into gRPC payloads.
- The framework protects against accidental use of production-like indices by default.
- OpenSearch write/read/physical targets must be exact-name allowlisted before Sentinel will run.
- Sentinel supports separate OpenSearch `write/read` aliases for the QA setup.
- `Playwright` is not needed for this stage.
- `RAGAS` is not the core of the framework, but the structure now leaves room to add answer-level evaluation later.
