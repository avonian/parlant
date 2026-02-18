"""
In-memory SessionStore for the stateless /v2/process endpoint.

Provides an ephemeral session store for a single request. The engine writes
events during processing (tool events, status events) and reads them back.
This handles the within-request lifecycle without touching any persistent storage.
"""

from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence, Set, cast

from typing_extensions import override

from parlant.core.agents import AgentId
from parlant.core.common import ItemNotFoundError, JSONSerializable, UniqueId, generate_id
from parlant.core.customers import CustomerId
from parlant.core.persistence.common import Cursor, SortDirection
from parlant.core.sessions import (
    Event,
    EventId,
    EventKind,
    EventSource,
    EventUpdateParams,
    Session,
    SessionId,
    SessionListing,
    SessionMode,
    SessionStore,
    SessionUpdateParams,
)


class InMemorySessionStore(SessionStore):
    """Ephemeral session store backed by in-memory lists."""

    def __init__(self, session: Session, initial_events: Sequence[Event]) -> None:
        self._sessions: dict[SessionId, Session] = {session.id: session}
        self._events: dict[SessionId, list[Event]] = {session.id: list(initial_events)}

    @override
    async def create_session(
        self,
        customer_id: CustomerId,
        agent_id: AgentId,
        creation_utc: datetime | None = None,
        title: str | None = None,
        mode: SessionMode | None = None,
        metadata: Mapping[str, JSONSerializable] = {},
        labels: Optional[Set[str]] = None,
    ) -> Session:
        session = Session(
            id=SessionId(generate_id()),
            creation_utc=creation_utc or datetime.now(timezone.utc),
            customer_id=customer_id,
            agent_id=agent_id,
            mode=mode or "auto",
            title=title,
            consumption_offsets={"client": 0},
            agent_states=[],
            metadata=metadata,
            labels=labels or set(),
        )
        self._sessions[session.id] = session
        self._events[session.id] = []
        return session

    @override
    async def read_session(self, session_id: SessionId) -> Session:
        if session_id not in self._sessions:
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")
        return self._sessions[session_id]

    @override
    async def delete_session(self, session_id: SessionId) -> None:
        self._sessions.pop(session_id, None)
        self._events.pop(session_id, None)

    @override
    async def update_session(
        self,
        session_id: SessionId,
        params: SessionUpdateParams,
    ) -> Session:
        if session_id not in self._sessions:
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        old = self._sessions[session_id]
        self._sessions[session_id] = Session(
            id=old.id,
            creation_utc=old.creation_utc,
            customer_id=params.get("customer_id", old.customer_id),
            agent_id=params.get("agent_id", old.agent_id),
            mode=params.get("mode", old.mode),
            title=params.get("title", old.title),
            consumption_offsets=params.get("consumption_offsets", old.consumption_offsets),
            agent_states=params.get("agent_states", old.agent_states),
            metadata=params.get("metadata", old.metadata),
            labels=old.labels,
        )
        return self._sessions[session_id]

    @override
    async def list_sessions(
        self,
        agent_id: AgentId | None = None,
        customer_id: CustomerId | None = None,
        limit: int | None = None,
        cursor: Cursor | None = None,
        sort_direction: SortDirection | None = None,
        labels: Optional[Set[str]] = None,
    ) -> SessionListing:
        items = list(self._sessions.values())
        return SessionListing(items=items, total_count=len(items), has_more=False)

    @override
    async def set_metadata(
        self,
        session_id: SessionId,
        key: str,
        value: JSONSerializable,
    ) -> Session:
        session = await self.read_session(session_id)
        new_metadata = {**session.metadata, key: value}
        return await self.update_session(session_id, SessionUpdateParams(metadata=new_metadata))

    @override
    async def unset_metadata(
        self,
        session_id: SessionId,
        key: str,
    ) -> Session:
        session = await self.read_session(session_id)
        new_metadata = {k: v for k, v in session.metadata.items() if k != key}
        return await self.update_session(session_id, SessionUpdateParams(metadata=new_metadata))

    @override
    async def upsert_labels(
        self,
        session_id: SessionId,
        labels: Set[str],
    ) -> Session:
        session = await self.read_session(session_id)
        updated = Session(
            id=session.id,
            creation_utc=session.creation_utc,
            customer_id=session.customer_id,
            agent_id=session.agent_id,
            mode=session.mode,
            title=session.title,
            consumption_offsets=session.consumption_offsets,
            agent_states=session.agent_states,
            metadata=session.metadata,
            labels=session.labels | labels,
        )
        self._sessions[session_id] = updated
        return updated

    @override
    async def remove_labels(
        self,
        session_id: SessionId,
        labels: Set[str],
    ) -> Session:
        session = await self.read_session(session_id)
        updated = Session(
            id=session.id,
            creation_utc=session.creation_utc,
            customer_id=session.customer_id,
            agent_id=session.agent_id,
            mode=session.mode,
            title=session.title,
            consumption_offsets=session.consumption_offsets,
            agent_states=session.agent_states,
            metadata=session.metadata,
            labels=session.labels - labels,
        )
        self._sessions[session_id] = updated
        return updated

    @override
    async def create_event(
        self,
        session_id: SessionId,
        source: EventSource,
        kind: EventKind,
        trace_id: str,
        data: JSONSerializable,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: datetime | None = None,
    ) -> Event:
        if session_id not in self._sessions:
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        events = self._events[session_id]
        event = Event(
            id=EventId(generate_id()),
            source=source,
            kind=kind,
            offset=len(events),
            creation_utc=creation_utc or datetime.now(timezone.utc),
            trace_id=trace_id,
            data=data,
            metadata=metadata,
            deleted=False,
        )
        events.append(event)
        return event

    @override
    async def read_event(
        self,
        session_id: SessionId,
        event_id: EventId,
    ) -> Event:
        if session_id not in self._events:
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        for event in self._events[session_id]:
            if event.id == event_id:
                return event

        raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")

    @override
    async def delete_event(self, event_id: EventId) -> None:
        for events in self._events.values():
            for i, event in enumerate(events):
                if event.id == event_id:
                    events[i] = Event(
                        id=event.id,
                        source=event.source,
                        kind=event.kind,
                        offset=event.offset,
                        creation_utc=event.creation_utc,
                        trace_id=event.trace_id,
                        data=event.data,
                        metadata=event.metadata,
                        deleted=True,
                    )
                    return
        raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")

    @override
    async def list_events(
        self,
        session_id: SessionId,
        source: EventSource | None = None,
        trace_id: str | None = None,
        kinds: Sequence[EventKind] = [],
        min_offset: int | None = None,
        exclude_deleted: bool = True,
    ) -> Sequence[Event]:
        if session_id not in self._events:
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        result = []
        for event in self._events[session_id]:
            if exclude_deleted and event.deleted:
                continue
            if source and event.source != source:
                continue
            if trace_id and event.trace_id != trace_id:
                continue
            if kinds and event.kind not in kinds:
                continue
            if min_offset is not None and event.offset < min_offset:
                continue
            result.append(event)
        return result

    @override
    async def update_event(
        self,
        session_id: SessionId,
        event_id: EventId,
        params: EventUpdateParams,
    ) -> Event:
        if session_id not in self._events:
            raise ItemNotFoundError(item_id=UniqueId(session_id), message="Session not found")

        for i, event in enumerate(self._events[session_id]):
            if event.id == event_id:
                updated = Event(
                    id=event.id,
                    source=event.source,
                    kind=event.kind,
                    offset=event.offset,
                    creation_utc=event.creation_utc,
                    trace_id=event.trace_id,
                    data=params.get("data", event.data),
                    metadata=cast(
                        Mapping[str, JSONSerializable],
                        params.get("metadata", event.metadata),
                    ),
                    deleted=event.deleted,
                )
                self._events[session_id][i] = updated
                return updated

        raise ItemNotFoundError(item_id=UniqueId(event_id), message="Event not found")
