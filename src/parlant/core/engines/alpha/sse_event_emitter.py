"""
SSE EventEmitter for the stateless /v2/process endpoint.

Collects engine events (status, message, tool, custom) into an async queue,
then yields them as SSE-formatted strings for a StreamingResponse.
Also persists events to the in-memory session store so the engine can
read them back (e.g., to check staged tool events).
"""

import asyncio
import json
from dataclasses import asdict
from typing import Any, AsyncIterator, Mapping, cast

from typing_extensions import override

from parlant.core.agents import Agent
from parlant.core.common import JSONSerializable
from parlant.core.emissions import (
    EmittedEvent,
    EventEmitter,
    MessageEventHandle,
    ensure_new_usage_params_and_get_trace_id,
)
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    SessionId,
    SessionStore,
    StatusEventData,
    ToolEventData,
)


_SENTINEL = object()


class SSEEventEmitter(EventEmitter):
    """
    EventEmitter that queues events for SSE streaming AND persists them
    to the session store so the engine can read back its own events.
    """

    def __init__(
        self,
        emitting_agent: Agent,
        session_store: SessionStore,
        session_id: SessionId,
    ) -> None:
        self._agent = emitting_agent
        self._session_store = session_store
        self._session_id = session_id
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._closed = False

    @override
    async def emit_status_event(
        self,
        trace_id: str | None = None,
        data: StatusEventData | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> EmittedEvent:
        trace_id = ensure_new_usage_params_and_get_trace_id(trace_id, data, **kwargs)

        event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.STATUS,
            trace_id=trace_id,
            data=cast(JSONSerializable, data),
            metadata=metadata,
        )

        await self._persist_and_queue(event)
        return event

    @override
    async def emit_message_event(
        self,
        trace_id: str | None = None,
        data: str | MessageEventData | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> MessageEventHandle:
        trace_id = ensure_new_usage_params_and_get_trace_id(trace_id, data, **kwargs)

        if isinstance(data, str):
            message_data = cast(
                JSONSerializable,
                MessageEventData(
                    message=data,
                    participant={
                        "id": self._agent.id,
                        "display_name": self._agent.name,
                    },
                ),
            )
        else:
            message_data = cast(JSONSerializable, data)

        event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.MESSAGE,
            trace_id=trace_id,
            data=message_data,
            metadata=metadata,
        )

        persisted = await self._persist_and_queue(event)

        async def update_message(new_data: MessageEventData) -> MessageEventHandle:
            nonlocal event
            event = EmittedEvent(
                source=event.source,
                kind=event.kind,
                trace_id=event.trace_id,
                data=cast(JSONSerializable, new_data),
                metadata=event.metadata,
            )
            # Update in session store
            await self._session_store.update_event(
                session_id=self._session_id,
                event_id=persisted.id,
                params={"data": cast(JSONSerializable, new_data)},
            )
            # Queue updated message event for SSE
            await self._queue.put(event)
            return MessageEventHandle(event=event, update=update_message)

        return MessageEventHandle(event=event, update=update_message)

    @override
    async def emit_tool_event(
        self,
        trace_id: str | None = None,
        data: ToolEventData | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> EmittedEvent:
        trace_id = ensure_new_usage_params_and_get_trace_id(trace_id, data, **kwargs)

        event = EmittedEvent(
            source=EventSource.SYSTEM,
            kind=EventKind.TOOL,
            trace_id=trace_id,
            data=cast(JSONSerializable, data),
            metadata=metadata,
        )

        await self._persist_and_queue(event)
        return event

    @override
    async def emit_custom_event(
        self,
        trace_id: str | None = None,
        data: JSONSerializable | None = None,
        metadata: Mapping[str, JSONSerializable] | None = None,
        **kwargs: Any,
    ) -> EmittedEvent:
        trace_id = ensure_new_usage_params_and_get_trace_id(trace_id, data, **kwargs)

        event = EmittedEvent(
            source=EventSource.AI_AGENT,
            kind=EventKind.CUSTOM,
            trace_id=trace_id,
            data=data,
            metadata=metadata,
        )

        await self._persist_and_queue(event)
        return event

    async def _persist_and_queue(self, event: EmittedEvent) -> Any:
        """Persist to session store and queue for SSE output."""
        persisted = await self._session_store.create_event(
            session_id=self._session_id,
            source=event.source,
            kind=event.kind,
            trace_id=event.trace_id,
            data=event.data,
            metadata=event.metadata or {},
        )
        await self._queue.put(event)
        return persisted

    def close(self) -> None:
        """Signal that no more events will be emitted."""
        self._closed = True
        self._queue.put_nowait(_SENTINEL)

    async def events(self) -> AsyncIterator[str]:
        """Yield SSE-formatted strings from the queue."""
        while True:
            item = await self._queue.get()

            if item is _SENTINEL:
                break

            # Support raw SSE frame strings (e.g., engine_state event)
            if isinstance(item, str):
                yield item
                continue

            if not isinstance(item, EmittedEvent):
                continue

            event_type = item.kind.value  # "status", "message", "tool", "custom"
            event_data = _serialize_emitted_event(item)

            yield f"event: {event_type}\ndata: {json.dumps(event_data, default=_json_safe)}\n\n"


def _json_safe(obj: object) -> object:
    """Fallback serializer for json.dumps — handles Pydantic models, dataclasses, etc."""
    # Pydantic models (e.g., Langflow Data objects)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")  # type: ignore[union-attr]
        except Exception:
            pass
    # dataclasses
    try:
        return asdict(obj)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        pass
    return str(obj)


def _serialize_emitted_event(event: EmittedEvent) -> dict[str, Any]:
    """Convert an EmittedEvent to a JSON-serializable dict for SSE."""
    result: dict[str, Any] = {
        "source": event.source.value,
        "trace_id": event.trace_id,
    }

    if event.data is not None:
        result["data"] = event.data

    if event.metadata:
        result["metadata"] = dict(event.metadata)

    return result
