# Concurrency Patterns Reference

## Qt / QML Worker Example

Preferred shape:

```python
class QueryWorker(QObject):
    result = Signal(QueryEnvelope)
    failed = Signal(QueryEnvelope)

class IdentityController(QObject):
    owner = "identity"

    def refresh(self):
        rid = worker.submit(owner=self.owner, kind="identity-list", fn=fetch_identities)
        self.latest_request = rid

    def on_result(self, envelope):
        if envelope.owner != self.owner:
            return
        if envelope.request_id != self.latest_request:
            return
        self.model.replace_rows(envelope.payload)
```

Avoid:

```python
# Too weak: every controller sees every payload shaped like rows.
if hasattr(payload, "rows"):
    self.model.replace_rows(payload.rows)
```

## Timer Refresh Pattern

- Keep one in-flight refresh per screen/kind.
- If the timer fires while a refresh is in flight, set `refresh_pending=True` instead of starting another task.
- When the task finishes, run one pending refresh if needed.
- Slow down or surface status if the task duration exceeds the timer interval.

## Pause/Live List Pattern

- Live mode may append/replace rows.
- Paused mode must not mutate the visible row set.
- Background refresh in paused mode can update a badge/counter but not reorder or push selected rows.
- Resume has an explicit rule: jump to latest, preserve selected key if still present, or ask/indicate.

## SQLite Pattern

- UI reads use read-only task-local connections.
- Collector/writer uses one managed writer path.
- Long maintenance operations check collector status first.
- Busy/locked is a first-class result, not an empty list.

## Validation Examples

- Start identity refresh and watch refresh at the same time; assert each model receives only its own rows.
- Trigger request A, then request B; force B to finish first; assert A is discarded.
- Pause a live list, insert new records, refresh; assert visible rows and selected key stay stable.
- Destroy/close a screen while a worker is running; assert no signal-source-deleted crash and no late state mutation.
- Run UI reads while collector writes; assert the UI reports busy/slow status rather than freezing or clearing data.
