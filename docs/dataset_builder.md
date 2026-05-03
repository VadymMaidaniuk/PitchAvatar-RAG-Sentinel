# Dataset Builder

The Dataset Builder is a local offline helper for drafting retrieval dataset JSON from plain-text
source material. A parsed file is only a source corpus. It is not automatically a complete dataset:
QA still chooses the queries, expected documents or sections, and chunk fragments manually.

The builder does not call gRPC, OpenSearch, LLM judges, Ragas, or `pytrec_eval`. It only parses local
text and can export a retrieval dataset JSON when explicitly downloaded or written by a caller.

## Supported Files

Current parser support is intentionally small:

- `.txt`
- `.md`

Markdown files are split by headings. Text files are split by blank lines, with a simple long-text
fallback that breaks very large paragraphs into smaller sections. Whitespace is normalized in
extracted section text. Empty files parse successfully with zero sections.

PDF, DOCX, and PPTX are future parser adapters. They should be added behind the same parser output
shape without making the current offline path depend on heavy document libraries.

## Preview Sections

Use the read-only preview command before creating a dataset:

```powershell
.venv\Scripts\python scripts\preview_dataset_source.py path\to\source.md
```

The command prints the source file name, total extracted characters, section count, section IDs,
titles, character counts, and short previews. It does not create or mutate any files.

## Create A Draft

Programmatic draft creation uses the parser output plus manual query drafts:

```python
from pitchavatar_rag_sentinel.dataset_builder import (
    ExpectationDraft,
    QueryDraft,
    build_retrieval_dataset,
    make_document_key,
    parse_source_file,
)

parsed = parse_source_file("path/to/source.md")
expected_key = make_document_key(parsed.source_file_name, parsed.sections[0])

dataset = build_retrieval_dataset(
    dataset_id="source_draft_v1",
    parsed_source=parsed,
    query_drafts=[
        QueryDraft(
            query_id="q_manual_lookup",
            query="manual search phrase",
            expectations=ExpectationDraft(
                expected_top1=expected_key,
                expected_in_topk=[expected_key],
                expected_top1_chunk_contains=["short exact fragment"],
            ),
        )
    ],
)
```

`build_retrieval_dataset` returns the existing `RetrievalDataset` Pydantic model, so generated JSON
uses the same validation rules as the rest of RAG Sentinel.

The Streamlit console also includes an optional `Dataset Builder` tab. It lets a user upload `.txt`
or `.md`, inspect parsed sections, enter a dataset ID, manually enter query drafts, manually select
expected document keys and chunk fragments, preview generated JSON, and download it. It does not
run datasets or perform cleanup.

## Limitations

- No automatic question generation.
- No automatic expectation inference.
- No semantic or LLM-based judging.
- No PDF, DOCX, or PPTX parsing yet.
- Generated document content is based on parsed sections, so QA should review previews before using
  a draft as a real regression dataset.

Future work can add PDF/DOCX/PPTX adapters and semi-automatic query suggestions, but expectations
should remain explicit and reviewable before export.
