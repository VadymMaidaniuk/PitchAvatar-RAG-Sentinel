# Retrieval Dataset Strategy

RAG Sentinel datasets are small JSON corpora for black-box retrieval checks. They seed documents,
run scoped or filtered searches, evaluate returned document ids and chunk text, write artifacts,
and clean up seeded data.

## Layout

Recommended retrieval dataset layout:

```text
datasets/retrieval/
  smoke/          # fastest real-environment sanity checks
  regression/     # stable broader baselines promoted after repeated success
  filters/        # metadata filter and document-scope checks
  negative/       # unrelated-query and forbidden-result checks
  chunking/       # chunk-level precision checks
  multilingual/   # Ukrainian/English and other mixed-language checks
  precision/      # stricter precision checks, not the stable smoke gate
  diagnostics/    # investigation/reproducer datasets for known or unclear behavior
```

The original root-level datasets remain supported:

- `datasets/retrieval/quantum_baseline.json`
- `datasets/retrieval/retrieval_baseline_v1.json`

## Current Baselines

- `smoke/retrieval_smoke_v1.json`: minimal document and chunk sanity check.
- `filters/metadata_filtering_v1.json`: similar documents separated by metadata filters and scope.
- `negative/negative_queries_v1.json`: empty-result and forbidden-result checks.
- `chunking/chunk_boundary_v1.json`: correct-document checks plus required chunk fragments.
- `multilingual/uk_en_mixed_v1.json`: Ukrainian and English mixed corpus checks.
- `precision/retrieval_precision_v1.json`: strict lower-rank forbidden checks preserved outside
  stable smoke.
- `diagnostics/alpha_matrix_v1.json`: reproduces current alpha-mode behavior while backend
  `SearchWithThreshold` semantics are clarified.

Smoke datasets are stable health checks. They should verify that seed, search, deterministic
evaluation, artifact writing, and cleanup work without depending on unresolved backend semantics.
Diagnostic datasets are for investigation and may encode current known behavior. Precision and
negative datasets can be stricter than smoke and should be interpreted with that purpose in mind.

Current calibration note:

- stable smoke uses `alpha=0.5`
- `alpha=1.0` is not used as a stable default until backend semantics are confirmed
- see [smoke_calibration_report.md](smoke_calibration_report.md)

## Expectation Types

Document-level checks:

- `expect_empty`
- `min_results`
- `expected_top1`
- `expected_in_topk`
- `forbidden_docs` (reported in artifacts as `forbidden_docs_absent`)

Chunk-level checks:

- `expected_top1_chunk_contains`: top result chunk must contain every listed fragment.
- `expected_in_topk_chunk_contains`: at least one returned chunk must contain every listed fragment.
- `forbidden_chunk_contains`: no returned chunk may contain any listed fragment.

Chunk checks are deterministic substring checks, not semantic or LLM evaluation. Matching is
case-insensitive and normalizes whitespace. Choose fragments that are short, unique, and stable.

Example:

```json
{
  "query_id": "q_boundary_beta_rollback",
  "query": "rollback guard token BETA-742 approval",
  "alpha": 0.5,
  "top_k": 4,
  "threshold": 0.0,
  "document_scope": ["doc_ops_multi_section", "doc_ops_decoy"],
  "expectations": {
    "expected_top1": "doc_ops_multi_section",
    "expected_top1_chunk_contains": [
      "Section Beta rollback",
      "rollback guard token BETA-742"
    ],
    "forbidden_docs": ["doc_ops_decoy"],
    "min_results": 1
  }
}
```

## Commands

Dry-run a dataset without gRPC or OpenSearch calls:

```powershell
.venv\Scripts\python scripts\run_dataset.py datasets\retrieval\smoke\retrieval_smoke_v1.json --dry-run
```

Run one dataset against configured services:

```powershell
.venv\Scripts\python scripts\run_dataset.py datasets\retrieval\smoke\retrieval_smoke_v1.json --summary
```

Validate all retrieval dataset JSON offline:

```powershell
.venv\Scripts\python -m pytest tests\datasets -vv
```
