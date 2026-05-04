from __future__ import annotations

import json
from html import escape
from typing import Any


def format_metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def status_text(value: bool | None) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return "not reported"


def cleanup_failed_text(value: bool | None) -> str:
    if value is True:
        return "failed"
    if value is False:
        return "ok"
    return "not reported"


def compact_json(value: Any, *, blank_empty: bool = False) -> str:
    if value is None or (blank_empty and value in ([], {})):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)


def html_escape(value: object) -> str:
    if value is None:
        return ""
    return escape(str(value))
