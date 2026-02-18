"""
Stateless /v2/process endpoint.

Single POST endpoint that receives everything inline (agent, guidelines, tools,
history, etc.) and streams back SSE events. No pre-synced state required.
"""

import asyncio
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from parlant.core.agents import Agent, AgentId, CompositionMode, MessageOutputMode
from parlant.core.canned_responses import CannedResponse, CannedResponseField, CannedResponseId
from parlant.core.common import Criticality, JSONSerializable, generate_id
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableValue,
    ContextVariableValueId,
)
from parlant.core.customers import Customer, CustomerId
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.engines.alpha.callback_tool_service import CallbackToolService
from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.engines.alpha.canned_response_generator import CannedResponseGenerator
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import GuidelineMatcher
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.alpha.in_memory_entity_queries import InMemoryEntityQueries
from parlant.core.engines.alpha.in_memory_session_store import InMemorySessionStore
from parlant.core.engines.alpha.message_generator import MessageGenerator
from parlant.core.engines.alpha.perceived_performance_policy import PerceivedPerformancePolicyProvider
from parlant.core.engines.alpha.relational_resolver import RelationalResolver
from parlant.core.engines.alpha.sse_event_emitter import SSEEventEmitter, _SENTINEL, _serialize_emitted_event
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.engines.types import Context
from parlant.core.entity_cq import EntityCommands
from parlant.core.glossary import Term, TermId
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationId,
)
from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.playbooks import PlaybookId
from parlant.core.sessions import (
    AgentState,
    Event,
    EventId,
    EventKind,
    EventSource,
    Session,
    SessionId,
)
from parlant.core.tags import TagId
from parlant.core.tools import (
    Tool,
    ToolId,
    ToolOverlap,
    ToolParameterDescriptor,
    ToolParameterOptions,
)
from parlant.core.tracer import Tracer


# ── Request DTOs ──────────────────────────────────────────────────────────────


class AgentDTO(BaseModel):
    id: str
    name: str
    description: str | None = None
    model_name: str | None = None
    composition_mode: str = "fluid"
    message_output_mode: str = "block"
    max_engine_iterations: int = 5


class CustomerDTO(BaseModel):
    id: str
    name: str
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, str] = Field(default_factory=dict)


class HistoryEventDTO(BaseModel):
    source: str  # "customer", "ai_agent", "system", etc.
    kind: str  # "message", "tool", "status", "custom"
    trace_id: str = ""
    data: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    creation_utc: str | None = None


class InlineToolParameterDTO(BaseModel):
    type: str = "string"
    item_type: str | None = None
    enum: list[str] | None = None
    description: str = ""
    examples: list[str] = Field(default_factory=list)


class InlineToolDefinitionDTO(BaseModel):
    service_name: str
    tool_name: str
    description: str = ""
    parameters: dict[str, InlineToolParameterDTO] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    consequential: bool = False


class InlineGuidelineDTO(BaseModel):
    id: str
    condition: str
    action: str | None = None
    description: str | None = None
    criticality: str = "medium"
    composition_mode: str | None = None
    track: bool = True
    labels: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_associations: list[str] = Field(default_factory=list)  # List of "service:tool" IDs


class InlineTermDTO(BaseModel):
    id: str
    name: str
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)


class InlineCannedResponseDTO(BaseModel):
    id: str
    value: str
    fields: list[dict[str, Any]] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    field_dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InlineContextVariableDTO(BaseModel):
    id: str
    name: str
    description: str | None = None
    value: Any = None  # Pre-resolved value
    key: str = ""  # The customer key used for lookup


class EngineStateDTO(BaseModel):
    applied_guideline_ids: list[str] = Field(default_factory=list)
    journey_paths: dict[str, list[str | None]] = Field(default_factory=dict)


class ProcessRequestDTO(BaseModel):
    agent: AgentDTO
    customer: CustomerDTO
    history: list[HistoryEventDTO] = Field(default_factory=list)
    guidelines: list[InlineGuidelineDTO] = Field(default_factory=list)
    terms: list[InlineTermDTO] = Field(default_factory=list)
    canned_responses: list[InlineCannedResponseDTO] = Field(default_factory=list)
    context_variables: list[InlineContextVariableDTO] = Field(default_factory=list)
    tools: list[InlineToolDefinitionDTO] = Field(default_factory=list)
    tool_callback_url: str = ""
    engine_state: EngineStateDTO | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Converters ────────────────────────────────────────────────────────────────


def _to_agent(dto: AgentDTO) -> Agent:
    try:
        composition_mode = CompositionMode(dto.composition_mode)
    except ValueError:
        composition_mode = CompositionMode.FLUID

    try:
        message_output_mode = MessageOutputMode(dto.message_output_mode)
    except ValueError:
        message_output_mode = MessageOutputMode.BLOCK

    return Agent(
        id=AgentId(dto.id),
        name=dto.name,
        description=dto.description,
        creation_utc=datetime.now(timezone.utc),
        max_engine_iterations=dto.max_engine_iterations,
        tags=[],
        composition_mode=composition_mode,
        message_output_mode=message_output_mode,
        model_name=dto.model_name,
    )


def _to_customer(dto: CustomerDTO) -> Customer:
    return Customer(
        id=CustomerId(dto.id),
        creation_utc=datetime.now(timezone.utc),
        name=dto.name,
        extra=dto.extra,
        tags=[TagId(t) for t in dto.tags],
    )


def _to_events(history: list[HistoryEventDTO]) -> list[Event]:
    events = []
    for i, h in enumerate(history):
        try:
            source = EventSource(h.source)
        except ValueError:
            source = EventSource.CUSTOMER

        try:
            kind = EventKind(h.kind)
        except ValueError:
            kind = EventKind.MESSAGE

        creation_utc = (
            datetime.fromisoformat(h.creation_utc)
            if h.creation_utc
            else datetime.now(timezone.utc)
        )

        data = h.data

        # Ensure message events always have a 'participant' field.
        # The engine accesses event.data["participant"]["display_name"]
        # and will crash with KeyError if it's missing.
        if kind == EventKind.MESSAGE and isinstance(data, dict):
            if "participant" not in data:
                display_name = "Customer" if source == EventSource.CUSTOMER else "Agent"
                data = {
                    **data,
                    "participant": {"id": None, "display_name": display_name},
                }

        events.append(
            Event(
                id=EventId(generate_id()),
                source=source,
                kind=kind,
                offset=i,
                creation_utc=creation_utc,
                trace_id=h.trace_id or generate_id(),
                data=data,
                metadata=h.metadata,
                deleted=False,
            )
        )
    return events


def _to_guidelines(dtos: list[InlineGuidelineDTO]) -> list[Guideline]:
    guidelines = []
    for g in dtos:
        try:
            criticality = Criticality(g.criticality)
        except ValueError:
            criticality = Criticality.MEDIUM

        composition_mode: CompositionMode | None = None
        if g.composition_mode:
            try:
                composition_mode = CompositionMode(g.composition_mode)
            except ValueError:
                pass

        guidelines.append(
            Guideline(
                id=GuidelineId(g.id),
                creation_utc=datetime.now(timezone.utc),
                content=GuidelineContent(
                    condition=g.condition,
                    action=g.action,
                    description=g.description,
                ),
                enabled=True,
                tags=[],
                metadata=g.metadata,
                criticality=criticality,
                labels=set(g.labels),
                composition_mode=composition_mode,
                track=g.track,
            )
        )
    return guidelines


def _to_terms(dtos: list[InlineTermDTO]) -> list[Term]:
    return [
        Term(
            id=TermId(t.id),
            creation_utc=datetime.now(timezone.utc),
            name=t.name,
            description=t.description,
            synonyms=t.synonyms,
            tags=[],
        )
        for t in dtos
    ]


def _to_canned_responses(dtos: list[InlineCannedResponseDTO]) -> list[CannedResponse]:
    return [
        CannedResponse(
            id=CannedResponseId(cr.id),
            creation_utc=datetime.now(timezone.utc),
            value=cr.value,
            fields=[
                CannedResponseField(
                    name=f.get("name", ""),
                    description=f.get("description", ""),
                    examples=f.get("examples", []),
                )
                for f in cr.fields
            ],
            signals=cr.signals,
            metadata=cr.metadata,
            tags=[],
            field_dependencies=cr.field_dependencies,
        )
        for cr in dtos
    ]


def _to_context_variables(
    dtos: list[InlineContextVariableDTO],
) -> tuple[list[ContextVariable], dict[ContextVariableId, ContextVariableValue]]:
    variables = []
    values: dict[ContextVariableId, ContextVariableValue] = {}

    for cv in dtos:
        var_id = ContextVariableId(cv.id)
        variables.append(
            ContextVariable(
                id=var_id,
                name=cv.name,
                description=cv.description,
                creation_utc=datetime.now(timezone.utc),
                tool_id=None,  # No tool-based refresh in stateless mode
                freshness_rules=None,
                tags=[],
            )
        )
        if cv.value is not None:
            values[var_id] = ContextVariableValue(
                id=ContextVariableValueId(generate_id()),
                last_modified=datetime.now(timezone.utc),
                data=cv.value,
            )

    return variables, values


def _to_tools(
    dtos: list[InlineToolDefinitionDTO],
) -> dict[str, dict[str, Tool]]:
    """Group tools by service_name → {tool_name: Tool}."""
    services: dict[str, dict[str, Tool]] = {}
    for t in dtos:
        params: dict[str, tuple[ToolParameterDescriptor, ToolParameterOptions]] = {}
        for pname, pdto in t.parameters.items():
            descriptor = ToolParameterDescriptor(
                type=pdto.type,
                description=pdto.description,
            )
            if pdto.item_type:
                descriptor["item_type"] = pdto.item_type
            if pdto.enum:
                descriptor["enum"] = pdto.enum
            if pdto.examples:
                descriptor["examples"] = pdto.examples

            params[pname] = (descriptor, ToolParameterOptions())

        tool = Tool(
            name=t.tool_name,
            creation_utc=datetime.now(timezone.utc),
            description=t.description,
            metadata={},
            parameters=params,
            required=t.required,
            consequential=t.consequential,
            overlap=ToolOverlap.NONE,
        )

        services.setdefault(t.service_name, {})[t.tool_name] = tool

    return services


def _to_tool_associations(
    guidelines: list[InlineGuidelineDTO],
) -> list[GuidelineToolAssociation]:
    associations = []
    for g in guidelines:
        for tool_id_str in g.tool_associations:
            try:
                tool_id = ToolId.from_string(tool_id_str)
            except ValueError:
                continue

            associations.append(
                GuidelineToolAssociation(
                    id=GuidelineToolAssociationId(generate_id()),
                    creation_utc=datetime.now(timezone.utc),
                    guideline_id=GuidelineId(g.id),
                    tool_id=tool_id,
                )
            )
    return associations


# ── Router factory ────────────────────────────────────────────────────────────


def create_router(
    logger: Logger,
    tracer: Tracer,
    meter: Meter,
    websocket_logger: WebSocketLogger,
    guideline_matcher: GuidelineMatcher,
    relational_resolver: RelationalResolver,
    tool_event_generator: ToolEventGenerator,
    message_generator: MessageGenerator,
    canned_response_generator: CannedResponseGenerator,
    perceived_performance_policy_provider: PerceivedPerformancePolicyProvider,
    hooks: EngineHooks,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v2/process",
        operation_id="v2_process",
        response_class=StreamingResponse,
    )
    async def process(request: ProcessRequestDTO) -> StreamingResponse:
        # 1. Convert DTOs → domain objects
        agent = _to_agent(request.agent)
        customer = _to_customer(request.customer)
        history_events = _to_events(request.history)
        guidelines = _to_guidelines(request.guidelines)
        terms = _to_terms(request.terms)
        canned_responses = _to_canned_responses(request.canned_responses)
        context_variables, cv_values = _to_context_variables(request.context_variables)
        tool_services = _to_tools(request.tools)
        tool_associations = _to_tool_associations(request.guidelines)

        # 2. Build ephemeral session
        session_id = SessionId(generate_id())

        agent_states: list[AgentState] = []
        if request.engine_state:
            agent_states.append(
                AgentState(
                    trace_id="previous",
                    applied_guideline_ids=[
                        GuidelineId(gid) for gid in request.engine_state.applied_guideline_ids
                    ],
                    journey_paths=request.engine_state.journey_paths,
                )
            )

        session = Session(
            id=session_id,
            creation_utc=datetime.now(timezone.utc),
            customer_id=customer.id,
            agent_id=agent.id,
            mode="auto",
            title=None,
            consumption_offsets={"client": 0},
            agent_states=agent_states,
            metadata=request.metadata,
        )

        # 3. Build in-memory session store
        session_store = InMemorySessionStore(session, history_events)

        # 4. Build callback tool service (one per service_name)
        #    The InMemoryEntityQueries.read_tool_service() returns the same service
        #    for any name since all tools route to the same callback URL.
        all_tools: dict[str, Tool] = {}
        for svc_tools in tool_services.values():
            all_tools.update(svc_tools)

        callback_service = CallbackToolService(
            service_name="nforce-tools",
            callback_url=request.tool_callback_url,
            tools=all_tools,
            metadata=request.metadata,
        )

        # 5. Build in-memory entity queries
        entity_queries = InMemoryEntityQueries(
            agent=agent,
            customer=customer,
            session=session,
            session_store=session_store,
            guidelines=guidelines,
            terms=terms,
            context_variables=context_variables,
            context_variable_values=cv_values,
            tool_associations=tool_associations,
            tool_service=callback_service,
            canned_responses=canned_responses,
        )

        # 6. Build entity commands (backed by in-memory session store)
        entity_commands = EntityCommands(
            session_store=session_store,
            context_variable_store=_NoOpContextVariableStore(),
        )

        # 7. Create SSE event emitter
        sse_emitter = SSEEventEmitter(
            emitting_agent=agent,
            session_store=session_store,
            session_id=session_id,
        )

        # Register SSE sink so engine logs (from all sub-components) are
        # piped into the SSE stream as 'log' events.
        sink_id = f"v2_{session_id}"
        websocket_logger.register_sse_sink(sink_id, sse_emitter._queue)

        # 8. Create per-request engine with in-memory backing
        engine = AlphaEngine(
            logger=logger,
            tracer=tracer,
            meter=meter,
            entity_queries=entity_queries,
            entity_commands=entity_commands,
            guideline_matcher=guideline_matcher,
            relational_resolver=relational_resolver,
            tool_event_generator=tool_event_generator,
            fluid_message_generator=message_generator,
            canned_response_generator=canned_response_generator,
            perceived_performance_policy_provider=perceived_performance_policy_provider,
            hooks=hooks,
        )

        # 9. Run engine in background task
        context = Context(session_id=session_id, agent_id=agent.id)

        logger.info(
            f"v2/process: starting engine — agent={agent.id}, "
            f"guidelines={len(guidelines)}, tools={sum(len(v) for v in tool_services.values())}, "
            f"history={len(history_events)}, cv={len(context_variables)}"
        )

        async def run_engine() -> None:
            try:
                await engine.process(context, sse_emitter)
                logger.info(f"v2/process: engine completed for session {session_id}")
            except Exception as exc:
                logger.error(f"v2/process engine error: {traceback.format_exception(exc)}")
                # Emit error as SSE event
                try:
                    await sse_emitter.emit_status_event(
                        trace_id=generate_id(),
                        data={"status": "error", "data": {"message": str(exc)}},
                    )
                except Exception:
                    pass
            finally:
                # Emit engine_state from final session state
                try:
                    final_session = await session_store.read_session(session_id)
                    if final_session.agent_states:
                        last_state = final_session.agent_states[-1]
                        state_data = {
                            "applied_guideline_ids": list(last_state.applied_guideline_ids),
                            "journey_paths": dict(last_state.journey_paths),
                        }
                        # Queue a special 'state' event before closing
                        import json as _json

                        sse_emitter._queue.put_nowait(
                            _make_raw_sse("state", state_data)
                        )
                except Exception:
                    pass

                sse_emitter.close()

        async def event_stream():
            # Send initial SSE comment immediately to confirm the stream is live.
            yield ": stream-start\n\n"

            task = asyncio.create_task(run_engine())
            try:
                # Use a timeout on queue reads so we can send SSE keepalive comments.
                # Without this, the HTTP connection may idle-timeout during long engine runs.
                while True:
                    try:
                        item = await asyncio.wait_for(sse_emitter._queue.get(), timeout=2.0)
                    except asyncio.TimeoutError:
                        # Send SSE comment as keepalive
                        yield ": keepalive\n\n"
                        continue

                    if item is _SENTINEL:
                        break

                    # Raw SSE frame strings (e.g., engine_state event)
                    if isinstance(item, str):
                        yield item
                        continue

                    if not isinstance(item, EmittedEvent):
                        continue

                    event_type = item.kind.value
                    event_data = _serialize_emitted_event(item)
                    yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"

            except asyncio.CancelledError:
                logger.warning("v2/process: event_stream cancelled by client disconnect")
                task.cancel()
                raise
            finally:
                websocket_logger.unregister_sse_sink(sink_id)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_raw_sse(event_type: str, data: Any) -> str:
    """Create a raw SSE frame string (not an EmittedEvent)."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class _NoOpContextVariableStore:
    """Minimal stub that satisfies EntityCommands' context_variable_store interface.

    In stateless mode, context variable values are pre-resolved in the request
    payload, so we don't need a real store. The engine's EntityCommands may
    call update_value() when a tool updates agent_state — we accept it silently.
    """

    async def update_value(
        self,
        variable_id: Any,
        key: str,
        data: Any,
    ) -> ContextVariableValue:
        return ContextVariableValue(
            id=ContextVariableValueId(generate_id()),
            last_modified=datetime.now(timezone.utc),
            data=data,
        )
