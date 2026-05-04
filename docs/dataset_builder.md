# Dataset Builder

The Dataset Builder is a local offline helper for drafting retrieval dataset JSON from source
material. A parsed file is only a source corpus. It is not automatically a complete dataset: QA
still chooses the queries, expected documents or sections, and chunk fragments manually.

The builder does not call gRPC, OpenSearch, LLM judges, Ragas, or `pytrec_eval`. It only parses local
text and can export a retrieval dataset JSON when explicitly downloaded or written by a caller.

## Supported Files

Current parser support:

- `.txt`
- `.md`
- `.pdf`
- `.pptx`

Markdown files are split by headings. Text files are split by blank lines, with a simple long-text
fallback that breaks very large paragraphs into smaller sections. Whitespace is normalized in
extracted section text. Empty files parse successfully with zero sections.

PDF and PPTX support is optional. Install parser extras before previewing those formats:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,report,ui,parsers]"
```

PDF support uses the text layer only. Scanned PDFs are not supported yet because OCR is not part of
the offline Dataset Builder. Completely empty PDF pages are skipped, which keeps generated draft
documents from being seeded with empty page text.

PPTX support extracts visible text from common slide text shapes and table cells. It does not parse
animations, layouts, images, charts, or speaker notes in this first adapter version.

In the real PitchAvatar product path, PHP extracts text from uploaded files and passes text into the
Go/RAG chunking pipeline. Slide and page metadata may not be preserved in RAG chunks. Parsed pages
and slides are useful for QA preview and manual expectation authoring, but production-like datasets
for PDF/PPTX should usually use one dataset document per source file.

## Document Modes

Dataset generation supports two document modes:

- `file_as_document`: one dataset document per source file. This is the default for `.pdf` and
  `.pptx`, and is the recommended production-like mode for PitchAvatar PDF/PPTX tests. The document
  content concatenates extracted sections in order and does not add artificial page/slide markers.
- `section_as_document`: one dataset document per parsed section. For PPTX, each slide becomes a
  document. For PDF, each non-empty parsed page becomes a document. For TXT/MD, each parsed section
  becomes a document. This is useful for controlled QA/debug retrieval experiments.

For `file_as_document`, document-level expectations are coarse. Prefer chunk-level expectations such
as `expected_top1_chunk_contains` or `expected_in_topk_chunk_contains` to verify that the correct
part of the file was retrieved.

## Preview Sections

Use the read-only preview command before creating a dataset:

```powershell
.venv\Scripts\python scripts\preview_dataset_source.py path\to\source.md
.venv\Scripts\python scripts\preview_dataset_source.py path\to\file.pdf
.venv\Scripts\python scripts\preview_dataset_source.py path\to\deck.pptx
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

Pass `document_mode="file_as_document"` or `document_mode="section_as_document"` to override the
format default. If omitted, TXT/MD keep section-based documents while PDF/PPTX use file-based
documents.

The Streamlit console also includes an optional `Dataset Builder` tab. It lets a user upload `.txt`,
`.md`, `.pdf`, or `.pptx`, inspect parsed sections, choose the document mode, enter a dataset ID,
manually enter query drafts, manually select expected document keys and chunk fragments, preview
generated JSON, and download it. It does not run datasets or perform cleanup.

Before any real retrieval run, validate downloaded/generated datasets with `--dry-run`:

```powershell
.venv\Scripts\python scripts\run_dataset.py path\to\generated_dataset.json --dry-run
```

Dry-run validation loads the same retrieval dataset schema and prints the planned document/query
counts without constructing gRPC or OpenSearch clients.

## Limitations

- No automatic question generation.
- No automatic expectation inference.
- No semantic or LLM-based judging.
- No OCR for scanned PDFs.
- No visual layout, chart, image, animation, or speaker-note extraction from PPTX.
- No DOCX parsing yet.
- Generated document content is based on parsed sections, so QA should review previews before using
  a draft as a real regression dataset.

Future work can add DOCX adapters and semi-automatic query suggestions, but expectations should
remain explicit and reviewable before export.
