# PitchAvatar RAG Sentinel

RAG Sentinel is a QA-owned Python 3.11+ harness for black-box retrieval regression testing of the
PitchAvatar RAG service.

The current stable lifecycle is:

```text
seed dataset over gRPC
  -> wait for OpenSearch visibility
  -> search via gRPC
  -> evaluate retrieval expectations and optional qrels IR metrics
  -> cleanup seeded documents
  -> write local artifacts
```

OpenSearch is used for QA-side resource readiness, chunk visibility checks, chunk inspection,
cleanup verification, and explicit cleanup fallback. Search and document lifecycle calls use the RAG
service gRPC API.

The project writes local JSON artifacts, per-run HTML reports, trend HTML/CSV reports, and exposes a
read-only Streamlit console for local artifact review.

## Prerequisites

- Python 3.11 or newer.
- PowerShell examples below assume Windows and the repository root.
- Real runs require access to the configured QA/dev gRPC service and QA OpenSearch targets.

## Quick Install

```powershell
cd C:\Projects\PitchAvatar-RAG-Sentinel
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,report]"
Copy-Item .env.example .env
.venv\Scripts\python scripts\generate_proto.py
```

Equivalent bootstrap helper:

```powershell
.\scripts\bootstrap.ps1 -WithReport
```

Use `.venv\Scripts\python -m ...` on Windows unless the virtual environment is activated.

Optional extras:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,report,ui]"
.venv\Scripts\python -m pip install -e ".[dev,report,ui,parsers]"
```

Install `ui` for the Streamlit console. Install `parsers` when the Dataset Builder must preview
PDF or PPTX files.

## Environment

Copy `.env.example` to `.env` and replace the template values before any real run. The important
settings are:

- `RAG_SENTINEL_GRPC_TARGET`, `RAG_SENTINEL_GRPC_SECURE`, and optional
  `RAG_SENTINEL_GRPC_SERVER_NAME`.
- `RAG_SENTINEL_OPENSEARCH_URL`, username, password, and certificate verification mode.
- `RAG_SENTINEL_OPENSEARCH_WRITE_ALIAS`, `RAG_SENTINEL_OPENSEARCH_READ_ALIAS`, and optional
  `RAG_SENTINEL_OPENSEARCH_PHYSICAL_INDEX`.
- `RAG_SENTINEL_OPENSEARCH_ALLOWED_TARGETS`, which must list every configured OpenSearch write,
  read, and physical target.
- `RAG_SENTINEL_NAMESPACE`, search defaults, timeout values, artifact root, and cleanup policy.

The harness refuses to run with placeholder credentials, without an OpenSearch allowlist, or against
protected index names unless explicitly configured for a disposable QA environment. OpenSearch
aliases and index names are QA verification/cleanup settings only; they are not sent in gRPC search
payloads.

## Dry Run

Validate a dataset and planned environment without gRPC or OpenSearch calls:

```powershell
.venv\Scripts\python scripts\run_dataset.py datasets\retrieval\smoke\retrieval_smoke_v1.json --dry-run
```

## Stable Smoke Run

Run the stable smoke dataset against configured QA/dev services:

```powershell
.venv\Scripts\python scripts\run_dataset.py datasets\retrieval\smoke\retrieval_smoke_v1.json --summary
```

This performs real seed, search, evaluation, and cleanup. It requires safe `.env` settings and
explicitly allowlisted QA OpenSearch targets.

## Dataset Catalog

Current retrieval datasets live under `datasets/retrieval/`:

- `smoke/retrieval_smoke_v1.json`: stable minimal real-environment sanity check.
- `filters/metadata_filtering_v1.json`: metadata filter and document-scope checks.
- `negative/negative_queries_v1.json`: empty-result and forbidden-result checks.
- `chunking/chunk_boundary_v1.json`: chunk-fragment checks.
- `multilingual/uk_en_mixed_v1.json`: Ukrainian/English mixed corpus checks.
- `precision/retrieval_precision_v1.json`: stricter precision checks outside the smoke gate.
- `diagnostics/alpha_matrix_v1.json`: investigation dataset for alpha-mode behavior.
- `regression/qrels_smoke_v1.json`: small qrels corpus for document-level IR metrics.
- `sandbox/builder_smoke_v1.json`: Dataset Builder sample output.

The original root-level baselines also remain supported:

- `datasets/retrieval/retrieval_baseline_v1.json`
- `datasets/retrieval/quantum_baseline.json`

## Reports

Generate a per-run static report:

```powershell
.venv\Scripts\python scripts\generate_report.py --run-dir artifacts\runs\<run-id>\<dataset-id>
```

Generate a per-run report for the newest local artifact:

```powershell
.venv\Scripts\python scripts\generate_report.py --latest
```

Generate a trends report across local artifact summaries:

```powershell
.venv\Scripts\python scripts\generate_trends_report.py --artifacts-root artifacts\runs
```

By default the trends command writes:

- `artifacts\runs\trends_report.html`
- `artifacts\runs\trends_report.csv`

Use `--output` and `--csv-output` to choose different paths.

Launch the read-only Streamlit console:

```powershell
.venv\Scripts\streamlit.exe run apps\sentinel_console.py
```

The console reads local artifacts and includes a Dataset Builder tab. It does not start real RAG
runs, call cleanup APIs, or mutate services.

## Dataset Builder

The Dataset Builder is an offline drafting helper for `.txt`, `.md`, `.pdf`, and `.pptx` sources.
It previews parsed sections and can generate retrieval dataset JSON after QA manually defines
queries, expected documents or sections, and expected chunk fragments.

Preview a source file:

```powershell
.venv\Scripts\python scripts\preview_dataset_source.py path\to\source.md
```

PDF support is text-layer only; scanned PDFs require OCR, which is not implemented. PPTX support
extracts visible slide text and table cells, not charts, images, animations, layouts, or speaker
notes.

Always dry-run generated JSON before any real retrieval run.

## Common Checks

```powershell
.venv\Scripts\ruff.exe check .
.venv\Scripts\python -m pytest -m "not integration and not destructive" -vv
.\scripts\test.ps1 -Profile offline
```

Offline tests must not call gRPC or OpenSearch. Real service tests are marked `integration` and
destructive service-mutating tests are marked `destructive`.

Useful PowerShell profiles:

```powershell
.\scripts\test.ps1 -Profile offline
.\scripts\test.ps1 -Profile grpc
.\scripts\test.ps1 -Profile available
.\scripts\test.ps1 -Profile full
```

## Detailed Docs

- [Datasets](docs/datasets.md)
- [Dataset Builder](docs/dataset_builder.md)
- [Artifact Contract](docs/artifact_contract.md)
- [Reporting](docs/reporting.md)
- [IR Metrics](docs/ir_metrics.md)
- [Test Organization](tests/README.md)
- [Future Answer Evaluation Plan](docs/answer_evaluation_plan.md)
- [Smoke Calibration Report](docs/smoke_calibration_report.md)

## Current Scope

Implemented today:

- dataset-driven retrieval regression
- gRPC seed/search/delete lifecycle
- OpenSearch resource readiness, verification, and cleanup safety
- document-level and chunk-level expectation checks
- optional qrels-based IR metrics
- local artifacts, per-run reports, trend reports, and read-only console
- offline Dataset Builder for `.txt`, `.md`, `.pdf`, and `.pptx`

Not implemented in this phase:

- answer evaluation
- Ragas or LLM judges
- `pytrec_eval`
- database-backed artifact storage
- real run execution from UI
- gRPC proto changes
- OCR for scanned PDFs
- DOCX parsing
- automatic query or expectation generation
