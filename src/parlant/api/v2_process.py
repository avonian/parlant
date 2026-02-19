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
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolCallBatcher, ToolCaller
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
    ToolService,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
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
    tags: list[str] = Field(default_factory=list)


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
    tags: list[str] = Field(default_factory=list)  # Tag IDs for relationship resolution


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


class InlineRelationshipDTO(BaseModel):
    id: str
    source_id: str
    source_kind: str  # "guideline", "tag", "tool"
    target_id: str
    target_kind: str  # "guideline", "tag", "tool"
    kind: str  # "entailment", "priority", "dependency", etc.


class EngineStateDTO(BaseModel):
    applied_guideline_ids: list[str] = Field(default_factory=list)
    journey_paths: dict[str, list[str | None]] = Field(default_factory=dict)


class ProcessRequestDTO(BaseModel):
    agent: AgentDTO
    customer: CustomerDTO
    history: list[HistoryEventDTO] = Field(default_factory=list)
    guidelines: list[InlineGuidelineDTO] = Field(default_factory=list)
    relationships: list[InlineRelationshipDTO] = Field(default_factory=list)
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
        tags=[TagId(t) for t in dto.tags],
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
                tags=[TagId(t) for t in g.tags],
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


# ── In-memory stores for per-request RelationalResolver ──────────────────────


def _to_relationships(dtos: list[InlineRelationshipDTO]) -> list["Relationship"]:
    from parlant.core.relationships import (
        Relationship as _Relationship,
        RelationshipId as _RelationshipId,
        RelationshipEntity as _RelationshipEntity,
        RelationshipEntityKind as _RelationshipEntityKind,
        RelationshipKind as _RelationshipKind,
    )

    def _parse_entity(
        entity_id: str, entity_kind: str
    ) -> _RelationshipEntity:
        kind = _RelationshipEntityKind(entity_kind)
        if kind == _RelationshipEntityKind.TOOL:
            return _RelationshipEntity(id=ToolId.from_string(entity_id), kind=kind)
        elif kind == _RelationshipEntityKind.TAG:
            return _RelationshipEntity(id=TagId(entity_id), kind=kind)
        else:
            return _RelationshipEntity(id=GuidelineId(entity_id), kind=kind)

    results = []
    for dto in dtos:
        results.append(
            _Relationship(
                id=_RelationshipId(dto.id),
                creation_utc=datetime.now(timezone.utc),
                source=_parse_entity(dto.source_id, dto.source_kind),
                target=_parse_entity(dto.target_id, dto.target_kind),
                kind=_RelationshipKind(dto.kind),
            )
        )
    return results


class _InMemoryRelationshipStore:
    """In-memory RelationshipStore that supports BFS for indirect queries.

    Only implements list_relationships (the only method RelationalResolver calls).
    Satisfies the RelationshipStore ABC at runtime via duck typing.
    """

    def __init__(self, relationships: Sequence["Relationship"]) -> None:
        from parlant.core.relationships import RelationshipKind as _RK

        self._relationships = list(relationships)
        # Pre-build directed graphs per kind for BFS
        self._graphs: dict[_RK, dict[str, list[tuple[str, "Relationship"]]]] = {}
        for r in self._relationships:
            kind_graph = self._graphs.setdefault(r.kind, {})
            src = r.source.id_to_string()
            kind_graph.setdefault(src, []).append((r.target.id_to_string(), r))

    async def list_relationships(
        self,
        kind=None,
        indirect: bool = False,
        source_id=None,
        target_id=None,
    ) -> Sequence["Relationship"]:
        from parlant.core.relationships import RelationshipKind as _RK

        # No filter — return all (or filtered by kind)
        if not source_id and not target_id:
            if kind:
                return [r for r in self._relationships if r.kind == kind]
            return list(self._relationships)

        kinds_to_check = [kind] if kind else list(_RK)
        results: list["Relationship"] = []

        for _kind in kinds_to_check:
            graph = self._graphs.get(_kind, {})

            if indirect:
                if source_id:
                    # BFS forward from source
                    results.extend(self._bfs_forward(graph, self._id_str(source_id)))
                if target_id:
                    # BFS backward to target — build reverse graph
                    rev = self._reverse_graph(graph)
                    results.extend(self._bfs_forward(rev, self._id_str(target_id)))
            else:
                sid = self._id_str(source_id) if source_id else None
                tid = self._id_str(target_id) if target_id else None
                for r in self._relationships:
                    if r.kind != _kind:
                        continue
                    if sid and r.source.id_to_string() != sid:
                        continue
                    if tid and r.target.id_to_string() != tid:
                        continue
                    results.append(r)

        return results

    @staticmethod
    def _id_str(entity_id) -> str:
        if hasattr(entity_id, "to_string"):
            return entity_id.to_string()
        return str(entity_id)

    @staticmethod
    def _bfs_forward(
        graph: dict[str, list[tuple[str, "Relationship"]]], start: str
    ) -> list["Relationship"]:
        visited: set[str] = set()
        queue = [start]
        results: list["Relationship"] = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for target, rel in graph.get(node, []):
                results.append(rel)
                if target not in visited:
                    queue.append(target)
        return results

    @staticmethod
    def _reverse_graph(
        graph: dict[str, list[tuple[str, "Relationship"]]]
    ) -> dict[str, list[tuple[str, "Relationship"]]]:
        rev: dict[str, list[tuple[str, "Relationship"]]] = {}
        for src, edges in graph.items():
            for tgt, rel in edges:
                rev.setdefault(tgt, []).append((src, rel))
        return rev

    # Stub methods to satisfy ABC (never called by RelationalResolver)
    async def create_relationship(self, *a, **kw):
        raise NotImplementedError

    async def read_relationship(self, *a, **kw):
        raise NotImplementedError

    async def delete_relationship(self, *a, **kw):
        raise NotImplementedError


class _InMemoryGuidelineStore:
    """In-memory GuidelineStore for per-request RelationalResolver.

    Only implements list_guidelines (the only method RelationalResolver calls).
    """

    def __init__(self, guidelines: Sequence[Guideline]) -> None:
        self._guidelines = list(guidelines)
        # Index: tag_id -> [guideline]
        self._by_tag: dict[str, list[Guideline]] = {}
        for g in self._guidelines:
            for tag in g.tags:
                self._by_tag.setdefault(str(tag), []).append(g)

    async def list_guidelines(
        self,
        tags=None,
        labels=None,
    ) -> Sequence[Guideline]:
        results = self._guidelines
        if tags:
            tag_strs = {str(t) for t in tags}
            matched: list[Guideline] = []
            seen: set[str] = set()
            for tag_str in tag_strs:
                for g in self._by_tag.get(tag_str, []):
                    if str(g.id) not in seen:
                        seen.add(str(g.id))
                        matched.append(g)
            results = matched
        if labels:
            results = [g for g in results if g.labels & labels]
        return results

    # Stub methods to satisfy ABC (never called by RelationalResolver)
    async def create_guideline(self, *a, **kw):
        raise NotImplementedError

    async def read_guideline(self, *a, **kw):
        raise NotImplementedError

    async def delete_guideline(self, *a, **kw):
        raise NotImplementedError

    async def update_guideline(self, *a, **kw):
        raise NotImplementedError

    async def find_guideline(self, *a, **kw):
        raise NotImplementedError

    async def upsert_tag(self, *a, **kw):
        raise NotImplementedError

    async def remove_tag(self, *a, **kw):
        raise NotImplementedError

    async def set_metadata(self, *a, **kw):
        raise NotImplementedError

    async def unset_metadata(self, *a, **kw):
        raise NotImplementedError

    async def upsert_labels(self, *a, **kw):
        raise NotImplementedError

    async def remove_labels(self, *a, **kw):
        raise NotImplementedError


# ── Inline service registry for per-request tool resolution ──────────────────


class _InlineServiceRegistry(ServiceRegistry):
    """Minimal ServiceRegistry backed by per-request CallbackToolService instances."""

    def __init__(self, services: dict[str, ToolService]) -> None:
        self._services = services

    async def read_tool_service(self, name: str) -> ToolService:
        if name not in self._services:
            from parlant.core.common import ItemNotFoundError, UniqueId
            raise ItemNotFoundError(item_id=UniqueId(name))
        return self._services[name]

    # Stubs — never called during stateless processing
    async def update_tool_service(self, *a, **kw) -> ToolService:
        raise NotImplementedError

    async def list_tool_services(self, *a, **kw):
        return list(self._services.items())

    async def delete_service(self, *a, **kw):
        raise NotImplementedError

    async def read_moderation_service(self, *a, **kw):
        raise NotImplementedError

    async def list_moderation_services(self, *a, **kw):
        return []

    async def read_nlp_service(self, *a, **kw):
        raise NotImplementedError

    async def list_nlp_services(self, *a, **kw):
        return []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


# ── Router factory ────────────────────────────────────────────────────────────


def create_router(
    logger: Logger,
    tracer: Tracer,
    meter: Meter,
    websocket_logger: WebSocketLogger,
    guideline_matcher: GuidelineMatcher,
    tool_event_generator: ToolEventGenerator,
    batcher: ToolCallBatcher,
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
        relationships = _to_relationships(request.relationships)
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

        # 4. Build one CallbackToolService per service_name so ToolCaller
        #    can look them up by the service_name in tool associations.
        callback_services: dict[str, CallbackToolService] = {}
        all_tools: dict[str, Tool] = {}
        for svc_name, svc_tools in tool_services.items():
            callback_services[svc_name] = CallbackToolService(
                service_name=svc_name,
                callback_url=request.tool_callback_url,
                tools=svc_tools,
                metadata=request.metadata,
            )
            all_tools.update(svc_tools)

        # Fallback service for InMemoryEntityQueries (returns any tool by name)
        fallback_service = CallbackToolService(
            service_name="nforce-tools",
            callback_url=request.tool_callback_url,
            tools=all_tools,
            metadata=request.metadata,
        )

        # Stash inline data for SimpleAgentHook via side-channel (not session
        # metadata, which gets serialized in tool callback payloads).
        from parlant.core.engines.alpha.simple_agent import (
            ToolDefinition,
            tool_to_openai_schema,
            set_inline_data,
        )

        inline_tool_defs: list[ToolDefinition] = []
        for svc_name, svc_tools in tool_services.items():
            svc = callback_services[svc_name]
            for tool in svc_tools.values():
                inline_tool_defs.append(
                    ToolDefinition(
                        service_name=svc_name,
                        service=svc,
                        tool=tool,
                        openai_schema=tool_to_openai_schema(tool),
                    )
                )

        inline_cv_pairs = [
            (var, cv_values[var.id])
            for var in context_variables
            if var.id in cv_values
        ]

        set_inline_data(session_id, inline_tool_defs, inline_cv_pairs)

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
            tool_service=fallback_service,
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

        # 8. Build per-request RelationalResolver with in-memory stores
        relational_resolver = RelationalResolver(
            relationship_store=_InMemoryRelationshipStore(relationships),
            guideline_store=_InMemoryGuidelineStore(guidelines),
            logger=logger,
            tracer=tracer,
        )

        # 9. Build per-request ToolCaller + ToolEventGenerator with inline service registry
        #    so tool lookups resolve against the request's inline tools, not the container's.
        inline_registry = _InlineServiceRegistry(callback_services)
        per_request_tool_caller = ToolCaller(
            logger=logger,
            meter=meter,
            service_registry=inline_registry,
            batcher=batcher,
        )
        per_request_tool_event_generator = ToolEventGenerator(
            logger=logger,
            meter=meter,
            tracer=tracer,
            tool_caller=per_request_tool_caller,
            service_registry=inline_registry,
        )

        # 10. Create per-request engine with in-memory backing
        engine = AlphaEngine(
            logger=logger,
            tracer=tracer,
            meter=meter,
            entity_queries=entity_queries,
            entity_commands=entity_commands,
            guideline_matcher=guideline_matcher,
            relational_resolver=relational_resolver,
            tool_event_generator=per_request_tool_event_generator,
            fluid_message_generator=message_generator,
            canned_response_generator=canned_response_generator,
            perceived_performance_policy_provider=perceived_performance_policy_provider,
            hooks=hooks,
        )

        # 10. Run engine in background task
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
