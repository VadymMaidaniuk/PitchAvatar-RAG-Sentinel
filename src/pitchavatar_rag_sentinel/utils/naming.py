from __future__ import annotations

import time
from uuid import uuid4


def unique_document_id(namespace: str, slug: str) -> str:
    timestamp = int(time.time() * 1000)
    suffix = uuid4().hex[:8]
    return f"{namespace}-{slug}-{timestamp}-{suffix}"

