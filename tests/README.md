# Test Organization

RAG Sentinel tests are split by safety boundary. The default local and CI profile is offline.

## Offline Tests

Offline tests use `pytest.mark.offline`. They validate dataset models, dry-run behavior,
evaluators, artifact readers, reports, trends, Dataset Builder parsing/drafting, and client payload
contracts without calling external services.

Rule: offline tests must not call gRPC or OpenSearch. They should not require `.env` service
credentials, running infrastructure, or network access.

## Integration Tests

Integration tests use `pytest.mark.integration` and usually also one or more capability markers
such as `grpc`, `opensearch`, `smoke`, `workflow`, `search`, `retrieval`, or `dataset`.

These tests require configured QA/dev services and should be run intentionally.

## Destructive Tests

Destructive tests use `pytest.mark.destructive`. They mutate QA/OpenSearch state by upserting,
searching, deleting, or cleanup-verifying documents. They must clean up after themselves and should
only target explicitly allowlisted QA indexes.

## Markers

Markers are declared in `pyproject.toml`:

- `offline`: no real gRPC or OpenSearch services.
- `integration`: requires configured external services.
- `grpc`: calls the RAG service over gRPC.
- `opensearch`: calls OpenSearch directly.
- `smoke`: minimal environment-readiness checks.
- `workflow`: document lifecycle and mutation behavior checks.
- `search`: retrieval, filtering, threshold, and ranking checks.
- `retrieval`: dataset-driven seed-search-evaluate-cleanup flows.
- `dataset`: dataset-backed scenarios and regression corpora.
- `destructive`: mutates OpenSearch data and requires cleanup.
- `slow`: long-running or large-volume scenarios.

## Commands

Lint:

```powershell
.venv\Scripts\ruff.exe check .
```

Offline pytest:

```powershell
.venv\Scripts\python -m pytest -m "not integration and not destructive" -vv
```

PowerShell offline profile:

```powershell
.\scripts\test.ps1 -Profile offline
```

Dry-run stable smoke dataset without gRPC/OpenSearch calls:

```powershell
.venv\Scripts\python scripts\run_dataset.py datasets\retrieval\smoke\retrieval_smoke_v1.json --dry-run
```

Dry-run qrels smoke dataset:

```powershell
.venv\Scripts\python scripts\run_dataset.py datasets\retrieval\regression\qrels_smoke_v1.json --dry-run
```
