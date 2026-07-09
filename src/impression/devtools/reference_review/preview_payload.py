"""Preview payload request and result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .async_core.messages import ReviewTaskKind, ReviewWorkbenchMessage
from .async_core.qt_handoff import sanitize_error_text
from .source_registry import EntrypointParameterRecord, ReviewSourceModelRecord


@dataclass(frozen=True)
class PreviewPayloadRequest:
    """Immutable request identity for building a preview payload."""

    owner: str
    request_id: int
    fixture_id: str
    generation: int
    source_path: Path
    entrypoint: str
    parameters: tuple[EntrypointParameterRecord, ...] = ()
    artifact_paths: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.owner:
            raise ValueError("owner must not be empty")
        if self.request_id < 1:
            raise ValueError("request_id must be positive")
        if not self.fixture_id:
            raise ValueError("fixture_id must not be empty")
        if self.generation < 0:
            raise ValueError("generation must not be negative")
        if not self.entrypoint:
            raise ValueError("entrypoint must not be empty")
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(
            self, "artifact_paths", tuple(Path(path) for path in self.artifact_paths)
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.owner,
                self.request_id,
                self.fixture_id,
                self.generation,
                self.source_path,
                self.entrypoint,
                self.parameters,
                self.artifact_paths,
                dict(self.metadata),
            ),
        )

    @classmethod
    def from_source_record(
        cls,
        record: ReviewSourceModelRecord,
        *,
        owner: str,
        request_id: int,
        generation: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PreviewPayloadRequest":
        return cls(
            owner=owner,
            request_id=request_id,
            fixture_id=record.fixture_id,
            generation=generation,
            source_path=record.source_path,
            entrypoint=record.entrypoint,
            parameters=record.parameters,
            artifact_paths=record.artifact_paths,
            metadata=metadata or {},
        )

    @classmethod
    def from_workbench_message(
        cls,
        message: ReviewWorkbenchMessage,
        record: ReviewSourceModelRecord,
        *,
        generation: int,
    ) -> "PreviewPayloadRequest":
        if message.kind is not ReviewTaskKind.PREVIEW_BUILD:
            raise ValueError("preview payload requests require PREVIEW_BUILD messages")
        if message.fixture_id not in (None, record.fixture_id):
            raise ValueError("message fixture_id does not match source record")
        return cls.from_source_record(
            record,
            owner=message.owner,
            request_id=message.request_id,
            generation=generation,
            metadata=message.payload,
        )

    @property
    def identity(self) -> tuple[str, int, str, int]:
        return (self.owner, self.request_id, self.fixture_id, self.generation)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "request_id": self.request_id,
            "fixture_id": self.fixture_id,
            "generation": self.generation,
            "source_path": self.source_path.as_posix(),
            "entrypoint": self.entrypoint,
            "parameters": [
                {"name": item.name, "value": item.value} for item in self.parameters
            ],
            "artifact_paths": [path.as_posix() for path in self.artifact_paths],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PreviewPayloadDiagnostic:
    """Sanitized preview payload failure diagnostic."""

    code: str
    message: str
    fixture_id: str
    owner: str
    request_id: int
    generation: int
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("code must not be empty")
        if not self.message:
            raise ValueError("message must not be empty")
        if not self.fixture_id:
            raise ValueError("fixture_id must not be empty")
        if not self.owner:
            raise ValueError("owner must not be empty")
        if self.request_id < 1:
            raise ValueError("request_id must be positive")
        if self.generation < 0:
            raise ValueError("generation must not be negative")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @classmethod
    def from_exception(
        cls,
        request: PreviewPayloadRequest,
        exc: BaseException,
        *,
        code: str = "preview-payload-error",
        cwd: Path | None = None,
        home: Path | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> "PreviewPayloadDiagnostic":
        text = sanitize_error_text(str(exc) or exc.__class__.__name__, cwd=cwd, home=home)
        return cls(
            code=code,
            message=text,
            fixture_id=request.fixture_id,
            owner=request.owner,
            request_id=request.request_id,
            generation=request.generation,
            details=details or {},
        )

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.code,
                self.message,
                self.fixture_id,
                self.owner,
                self.request_id,
                self.generation,
                dict(self.details),
            ),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "fixture_id": self.fixture_id,
            "owner": self.owner,
            "request_id": self.request_id,
            "generation": self.generation,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PreviewPayloadFileMetadata:
    """Metadata for a file-backed preview payload."""

    path: Path
    payload_format: str
    byte_count: int
    dataset_count: int

    def __post_init__(self) -> None:
        if not self.payload_format:
            raise ValueError("payload_format must not be empty")
        if self.byte_count < 0:
            raise ValueError("byte_count must not be negative")
        if self.dataset_count < 0:
            raise ValueError("dataset_count must not be negative")
        object.__setattr__(self, "path", Path(self.path))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "payload_format": self.payload_format,
            "byte_count": self.byte_count,
            "dataset_count": self.dataset_count,
        }


@dataclass(frozen=True)
class PreviewPayload:
    """Immutable preview payload result record."""

    request: PreviewPayloadRequest
    payload_path: Path | None = None
    payload_kind: str = "impress"
    file_metadata: PreviewPayloadFileMetadata | None = None
    diagnostic: PreviewPayloadDiagnostic | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.payload_path is not None:
            object.__setattr__(self, "payload_path", Path(self.payload_path))
        if not self.payload_kind:
            raise ValueError("payload_kind must not be empty")
        if self.payload_path is not None and self.diagnostic is not None:
            raise ValueError("payload results cannot include both payload_path and diagnostic")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.request,
                self.payload_path,
                self.payload_kind,
                self.file_metadata,
                self.diagnostic,
                dict(self.metadata),
            ),
        )

    @classmethod
    def success(
        cls,
        request: PreviewPayloadRequest,
        *,
        payload_path: Path | None = None,
        payload_kind: str = "impress",
        file_metadata: PreviewPayloadFileMetadata | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PreviewPayload":
        return cls(
            request=request,
            payload_path=payload_path,
            payload_kind=payload_kind,
            file_metadata=file_metadata,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        request: PreviewPayloadRequest,
        diagnostic: PreviewPayloadDiagnostic,
        *,
        payload_kind: str = "impress",
        metadata: Mapping[str, Any] | None = None,
    ) -> "PreviewPayload":
        if diagnostic.fixture_id != request.fixture_id:
            raise ValueError("diagnostic fixture_id does not match request")
        return cls(
            request=request,
            payload_kind=payload_kind,
            diagnostic=diagnostic,
            metadata=metadata or {},
        )

    @property
    def ok(self) -> bool:
        return self.diagnostic is None

    @property
    def identity(self) -> tuple[str, int, str, int]:
        return self.request.identity

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request": self.request.to_json_dict(),
            "payload_path": None if self.payload_path is None else self.payload_path.as_posix(),
            "payload_kind": self.payload_kind,
            "file_metadata": None
            if self.file_metadata is None
            else self.file_metadata.to_json_dict(),
            "diagnostic": None
            if self.diagnostic is None
            else self.diagnostic.to_json_dict(),
            "metadata": dict(self.metadata),
        }
