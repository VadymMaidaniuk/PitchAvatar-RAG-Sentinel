# Reporting

RAG Sentinel reporting is read-only and local-artifact based. It reads files under
`artifacts/runs` and does not call gRPC, OpenSearch, cleanup APIs, LLM judges, Ragas, or
`pytrec_eval`.

## Per-Run Report

Generate a static HTML report for one dataset artifact directory:

```powershell
.venv\Scripts\python scripts\generate_report.py --run-dir artifacts\runs\<run-id>\<dataset-id>
```

Generate a per-run report for the newest dataset artifact:

```powershell
.venv\Scripts\python scripts\generate_report.py --latest
```

The output is `report.html` inside the selected dataset artifact directory.

## Trends Report

Generate a static trends report across all dataset summaries under the artifact root:

```powershell
.venv\Scripts\python scripts\generate_trends_report.py --artifacts-root artifacts\runs
```

By default this writes:

- `artifacts\runs\trends_report.html`
- `artifacts\runs\trends_report.csv`

Use `--output` and `--csv-output` to choose different paths.

The trends report includes overall run count, datasets covered, latest status per dataset, all runs
sorted latest-first, pass/fail status, failed query counts, retrieval metrics, timing metrics, and
links to per-run `report.html` when that file exists.

## Streamlit Console

Launch the local read-only console:

```powershell
.venv\Scripts\streamlit.exe run apps\sentinel_console.py
```

The `Artifacts` tab shows one selected dataset run. The `Trends` tab scans all local summaries from
the selected artifact root, supports dataset filtering, shows latest status per dataset, lists runs
latest-first, draws simple metric trend charts, and lets you open the same per-run detail view used
by the Artifacts tab.

The console does not start real RAG runs and does not perform cleanup.

## Limitations

- Reports read local artifacts only; there is no database or remote sync.
- Timestamp inference uses explicit timestamp-like summary fields when present, then the run ID
  timestamp pattern, then `summary.json` modified time as fallback.
- Trends depend on consistent `dataset_id` values across runs.
- Older artifacts without `metrics` still load, but metric cells are shown as `n/a`.
- Links to per-run `report.html` appear only after those reports have been generated.
