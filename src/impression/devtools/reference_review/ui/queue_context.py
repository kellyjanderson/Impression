"""Queue and selected-fixture context view models."""

from __future__ import annotations

from dataclasses import dataclass

from ..source_registry import ReviewContextPayload, ReviewSourceModelRecord, build_review_context_payload


@dataclass(frozen=True)
class FixtureQueueItem:
    fixture_id: str
    feature_name: str
    source_display_path: str
    expected_output: str | None
    status: str = "dirty"

    @classmethod
    def from_record(cls, record: ReviewSourceModelRecord, *, status: str = "dirty") -> "FixtureQueueItem":
        return cls(
            fixture_id=record.fixture_id,
            feature_name=record.feature_name,
            source_display_path=record.identity.display_path,
            expected_output=record.expected_output,
            status=status,
        )


@dataclass(frozen=True)
class SelectedFixtureContext:
    fixture_id: str | None
    feature_name: str | None
    source_display_path: str | None
    expected_output: str | None
    empty: bool = False

    @classmethod
    def empty_state(cls) -> "SelectedFixtureContext":
        return cls(None, None, None, None, empty=True)

    @classmethod
    def from_payload(cls, payload: ReviewContextPayload) -> "SelectedFixtureContext":
        return cls(
            payload.fixture_id,
            payload.feature_name,
            payload.source_display_path,
            payload.expected_output,
        )


class FixtureQueueViewModel:
    """Owns visible selection state while source registry owns fixture data."""

    def __init__(
        self,
        records: tuple[ReviewSourceModelRecord, ...],
        *,
        statuses: dict[str, str] | None = None,
    ) -> None:
        self.records = records
        self.statuses = statuses or {}
        self.selected_index = self._first_dirty_index()

    def _first_dirty_index(self) -> int | None:
        for index, record in enumerate(self.records):
            if self.statuses.get(record.fixture_id, "dirty") == "dirty":
                return index
        return 0 if self.records else None

    @property
    def items(self) -> tuple[FixtureQueueItem, ...]:
        return tuple(
            FixtureQueueItem.from_record(
                record,
                status=self.statuses.get(record.fixture_id, "dirty"),
            )
            for record in self.records
        )

    @property
    def selected_record(self) -> ReviewSourceModelRecord | None:
        if self.selected_index is None:
            return None
        return self.records[self.selected_index]

    @property
    def selected_context(self) -> SelectedFixtureContext:
        record = self.selected_record
        if record is None:
            return SelectedFixtureContext.empty_state()
        return SelectedFixtureContext.from_payload(build_review_context_payload(record))

    def select_fixture(self, fixture_id: str) -> bool:
        for index, record in enumerate(self.records):
            if record.fixture_id == fixture_id:
                self.selected_index = index
                return True
        return False

    def next(self) -> SelectedFixtureContext:
        if self.selected_index is None:
            return self.selected_context
        self.selected_index = min(self.selected_index + 1, len(self.records) - 1)
        return self.selected_context

    def previous(self) -> SelectedFixtureContext:
        if self.selected_index is None:
            return self.selected_context
        self.selected_index = max(self.selected_index - 1, 0)
        return self.selected_context

