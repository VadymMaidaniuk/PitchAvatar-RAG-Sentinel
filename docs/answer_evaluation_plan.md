# Future Answer Evaluation Plan

Answer evaluation is planned but not implemented yet. The current framework should stay focused on
retrieval until the RAG service exposes a stable endpoint that returns retrieved chunks and the LLM
answer together.

## Planned Endpoint Shape

Future support assumes a new endpoint that returns:

- retrieved chunks with document IDs, chunk text, metadata, and scores
- final LLM answer text
- enough request metadata to reproduce the query scope

The existing gRPC proto is not changed in this stabilization pass.

## Dataset Schema Idea

Future datasets may add an optional `answer_expectations` object per query. It should be additive
and backward compatible with existing retrieval datasets.

Possible shape:

```json
{
  "answer_expectations": {
    "answer_not_empty": true,
    "answer_must_contain": ["upload limit"],
    "answer_must_not_contain": ["deprecated"],
    "answer_language": "en",
    "answer_should_refuse_when_context_missing": false
  }
}
```

## Deterministic Checks

Initial answer checks should be deterministic:

- `answer_not_empty`: answer text must be present after trimming whitespace.
- `answer_must_contain`: answer must contain all configured literal fragments after normalized,
  case-insensitive comparison.
- `answer_must_not_contain`: answer must not contain configured forbidden fragments.
- `answer_language`: answer should match the expected language using a simple local detector or a
  constrained rule set.
- `answer_should_refuse_when_context_missing`: answer should use an expected refusal pattern when
  retrieval context is empty or insufficient.

These checks should remain separate from retrieval expectation checks.

## Future Ragas Integration

Ragas may be added later after deterministic answer checks and artifact contracts are stable.
Ragas should be optional, dependency-gated, and reported separately from deterministic metrics. It
should not be required for offline tests.

## Reporting Impact

Reports will likely need:

- an answer evaluation summary section
- per-query answer text and answer checks
- separate deterministic answer metrics
- optional Ragas metrics when configured
- clear distinction between retrieval failures, answer failures, and cleanup failures

The Streamlit console should remain read-only unless explicit run execution safety controls are
designed later.

## Artifact Impact

Future query artifacts may add:

- `answer`
- `answer_evaluation`
- `answer_request` or endpoint-specific request fields if they differ from retrieval search
- optional `ragas_evaluation`

Future summaries may add:

- `answer_metrics`
- optional `ragas_metrics`
- answer-specific pass/fail status fields

All additions should be optional so old retrieval-only artifacts continue to load.
