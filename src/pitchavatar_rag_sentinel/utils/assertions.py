from __future__ import annotations

from collections.abc import Iterable


def unique_document_ids(results: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    document_ids: list[str] = []
    for result in results:
        document_id = getattr(result, "document_id")
        if document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)
    return document_ids


def assert_only_expected_document_ids(actual: Iterable[str], expected: Iterable[str]) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    assert actual_set == expected_set, (
        "Unexpected document ids.\n"
        f"Actual:   {sorted(actual_set)}\n"
        f"Expected: {sorted(expected_set)}"
    )

