# PitchAvatar RAG Sentinel

RAG Sentinel is a QA-owned Python harness for testing the PitchAvatar RAG service as a black box.
The current stable flow is:

```text
seed dataset -> search via gRPC -> evaluate retrieval -> cleanup seeded documents
```

It writes local JSON artifacts, static reports, trend reports, and a read-only Streamlit console.
OpenSearch is used only for QA visibility checks, chunk inspection, cleanup verification, and
explicit cleanup fallback.

## Quick Install

```powershell
cd C:\Projects\PitchAvatar-RAG-Sentinel
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,report]"
Copy-Item .env.example .env
.venv\Scripts\python scripts\generate_proto.py
```

Use `.venv\Scripts\python -m pytest ...` on Windows unless the virtual environment is activated.

Optional extras:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,report,ui]"
.venv\Scripts\python -m pip install -e ".[dev,report,ui,parsers]"
```

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

## Reports

Generate a per-run static report:

```powershell
.venv\Scripts\python scripts\generate_report.py --run-dir artifacts\runs\<run-id>\<dataset-id>
```

Generate a trends report across local artifact summaries:

```powershell
.venv\Scripts\python scripts\generate_trends_report.py --artifacts-root artifacts\runs
```

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

Always dry-run generated JSON before any real retrieval run.

## Common Checks

```powershell
.venv\Scripts\ruff.exe check .
.venv\Scripts\python -m pytest -m "not integration and not destructive" -vv
.\scripts\test.ps1 -Profile offline
```

Offline tests must not call gRPC or OpenSearch. Real service tests are marked `integration` and
destructive service-mutating tests are marked `destructive`.

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
- OpenSearch verification and cleanup safety
- document-level and chunk-level expectation checks
- optional qrels-based IR metrics
- local artifacts, reports, trends, and read-only console
- offline Dataset Builder for `.txt`, `.md`, `.pdf`, and `.pptx`

Not implemented in this phase:

- answer evaluation
- Ragas
- `pytrec_eval`
- database-backed artifact storage
- real run execution from UI
- gRPC proto changes
