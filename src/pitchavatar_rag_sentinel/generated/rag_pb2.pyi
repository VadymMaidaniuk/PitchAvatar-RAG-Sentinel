from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class UploadDocumentRequest(_message.Message):
    __slots__ = ("document_id", "index_name", "document_content", "document_url", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_CONTENT_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_URL_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    index_name: str
    document_content: str
    document_url: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, document_id: _Optional[str] = ..., index_name: _Optional[str] = ..., document_content: _Optional[str] = ..., document_url: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class UploadDocumentResponse(_message.Message):
    __slots__ = ("document_id", "success", "message")
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    success: bool
    message: str
    def __init__(self, document_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ...) -> None: ...

class MetadataFilter(_message.Message):
    __slots__ = ("field", "values")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    field: str
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, field: _Optional[str] = ..., values: _Optional[_Iterable[str]] = ...) -> None: ...

class SearchWithThresholdRequest(_message.Message):
    __slots__ = ("query", "top_k", "alpha", "index_name", "threshold", "document_ids", "filters")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    query: str
    top_k: int
    alpha: float
    index_name: str
    threshold: float
    document_ids: _containers.RepeatedScalarFieldContainer[str]
    filters: _containers.RepeatedCompositeFieldContainer[MetadataFilter]
    def __init__(self, query: _Optional[str] = ..., top_k: _Optional[int] = ..., alpha: _Optional[float] = ..., index_name: _Optional[str] = ..., threshold: _Optional[float] = ..., document_ids: _Optional[_Iterable[str]] = ..., filters: _Optional[_Iterable[_Union[MetadataFilter, _Mapping]]] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[SearchResult]
    def __init__(self, results: _Optional[_Iterable[_Union[SearchResult, _Mapping]]] = ...) -> None: ...

class SearchResult(_message.Message):
    __slots__ = ("document_id", "page_content", "metadata", "score")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    PAGE_CONTENT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    page_content: str
    metadata: _containers.ScalarMap[str, str]
    score: float
    def __init__(self, document_id: _Optional[str] = ..., page_content: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ..., score: _Optional[float] = ...) -> None: ...

class DeleteIndexRequest(_message.Message):
    __slots__ = ("index_name",)
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    def __init__(self, index_name: _Optional[str] = ...) -> None: ...

class DeleteIndexResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class IndexExistsRequest(_message.Message):
    __slots__ = ("index_name",)
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    def __init__(self, index_name: _Optional[str] = ...) -> None: ...

class IndexExistsResponse(_message.Message):
    __slots__ = ("exists", "message")
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    exists: bool
    message: str
    def __init__(self, exists: bool = ..., message: _Optional[str] = ...) -> None: ...
