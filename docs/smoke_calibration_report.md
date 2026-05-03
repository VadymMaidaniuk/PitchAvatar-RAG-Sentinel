# Smoke Calibration Report

## Context

`retrieval_smoke_v1` originally mixed stable health checks with stricter lower-rank
forbidden checks and one `alpha=1.0` billing query. Real diagnostics showed that the
harness, indexing, artifact writing, and cleanup are working, but current dev RAG service
retrieval behavior differs by `alpha`.

This report documents the calibration decision: stable smoke should verify the basic
seed-search-evaluate-cleanup path with deterministic top-result/chunk expectations, while
stricter precision and alpha behavior checks live in separate diagnostic datasets.

## Diagnostic Matrix

Billing query:

`invoice dispute adjustment reason`

The diagnostic run seeded the same `doc_release` and `doc_billing` documents used by smoke.
Both documents became visible in OpenSearch. Direct OpenSearch `match` on `content` found
`doc_billing`.

| Scope | alpha | gRPC SearchWithThreshold result |
| --- | ---: | --- |
| all docs | 1.0 | 0 results |
| all docs | 0.5 | 2 results, top-1 `doc_billing` |
| all docs | 0.0 | 2 results, top-1 `doc_billing` |
| only `doc_billing` | 1.0 | 0 results |
| only `doc_billing` | 0.5 | 1 result, `doc_billing` |
| only `doc_billing` | 0.0 | 1 result, `doc_billing` |
| direct OpenSearch `match` | n/a | 1 hit, `doc_billing` |

## Interpretation

The `alpha=1.0` path has unclear behavior in the current dev RAG service. It returns no
results even when `document_ids` scope contains only `doc_billing`, although OpenSearch can
match the same document directly and gRPC returns the document for `alpha=0.5` and `alpha=0.0`.

Stable smoke should not block framework development on this backend semantic question. Until
backend behavior is confirmed, stable smoke uses `alpha=0.5`.

The original lower-rank forbidden checks are useful, but they are stricter than a basic health
check. In the tiny smoke corpus, `top_k=3`, `threshold=0.0`, and `document_scope=all` can return
low-scored lower-rank noise even when top-1 is correct. These checks now live in
`precision/retrieval_precision_v1.json`.

## Dataset Split

- `smoke/retrieval_smoke_v1.json`: stable health check. Uses `alpha=0.5`, `min_results`,
  `expected_top1`, and chunk expectations.
- `precision/retrieval_precision_v1.json`: strict lower-rank forbidden checks preserved from
  the original smoke corpus. This is diagnostic/precision coverage, not the stable smoke gate.
- `diagnostics/alpha_matrix_v1.json`: reproduces current alpha behavior, including the current
  `alpha=1.0` empty-result behavior for the billing query.

## Backend Questions

- What is the exact `alpha` semantics in `SearchWithThreshold`?
- Does `alpha=1.0` mean text/BM25-only or vector-only?
- Is there a server-side `min_score` or filter for `alpha=1.0`?
- Which OpenSearch field is queried by `SearchWithThreshold`?
- Is `document_ids` scope a strict filter in every alpha mode?

## Recommendation

Do not loosen assertions just to make a failing run pass. Keep stable smoke focused on health
and regression signal, and use diagnostic datasets to track unresolved backend behavior until
the service semantics are clarified.
