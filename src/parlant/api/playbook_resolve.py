# Copyright 2026 Avonian / nForce
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Playbook Resolve Endpoint

POST /playbooks/{playbookId}/resolve

Resolves the full playbook state (walking the inheritance chain) and returns
both resolved data (flat, for Parlant runtime) and source data (original
structures, for nForce to store for revert/audit).

Called by nForce during the "Release" operation.
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Optional, Sequence

from fastapi import APIRouter, HTTPException, status

from parlant.core.canned_responses import CannedResponse, CannedResponseStore
from parlant.core.common import DefaultBaseModel
from parlant.core.context_variables import ContextVariable, ContextVariableStore
from parlant.core.glossary import GlossaryStore, Term
from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationStore,
)
from parlant.core.journeys import Journey, JourneyStore
from parlant.core.playbooks import Playbook, PlaybookId, PlaybookStore
from parlant.core.relationships import (
    Relationship,
    RelationshipEntityKind,
    RelationshipStore,
)
from parlant.core.tags import TagId, TagStore


# ---------------------------------------------------------------------------
# Response DTOs
# ---------------------------------------------------------------------------


class ResolvedGuidelineDTO(DefaultBaseModel):
    id: str
    condition: str
    action: str | None = None
    description: str | None = None
    criticality: str = "medium"
    composition_mode: str | None = None
    track: bool = True
    labels: list[str] = []
    enabled: bool = True
    metadata: dict[str, Any] = {}
    tool_ids: list[dict[str, str]] = []
    journey_origin: dict[str, str] | None = None


class ResolvedRelationshipDTO(DefaultBaseModel):
    source_guideline_id: str
    target_guideline_id: str
    kind: str


class ResolvedTermDTO(DefaultBaseModel):
    id: str
    name: str
    description: str
    synonyms: list[str] = []


class ResolvedCannedResponseDTO(DefaultBaseModel):
    id: str
    value: str
    fields: list[dict[str, Any]] = []
    signals: list[str] = []
    field_dependencies: list[str] = []
    metadata: dict[str, Any] = {}


class ResolvedContextVariableDTO(DefaultBaseModel):
    id: str
    name: str
    description: str | None = None
    tool_id: dict[str, str] | None = None
    freshness_rules: str | None = None


class SourceGuidelineDTO(DefaultBaseModel):
    id: str
    condition: str
    action: str | None = None
    description: str | None = None
    criticality: str = "medium"
    composition_mode: str | None = None
    track: bool = True
    labels: list[str] = []
    tags: list[str] = []
    tool_ids: list[dict[str, str]] = []


class SourceJourneyNodeDTO(DefaultBaseModel):
    id: str
    action: str | None = None
    description: str | None = None
    tools: list[str] = []
    composition_mode: str | None = None
    metadata: dict[str, Any] = {}


class SourceJourneyEdgeDTO(DefaultBaseModel):
    id: str
    source: str
    target: str
    condition: str | None = None
    metadata: dict[str, Any] = {}


class SourceJourneyDTO(DefaultBaseModel):
    id: str
    title: str
    description: str
    conditions: list[str] = []
    tags: list[str] = []
    composition_mode: str | None = None
    nodes: list[SourceJourneyNodeDTO] = []
    edges: list[SourceJourneyEdgeDTO] = []


class SourceRelationshipDTO(DefaultBaseModel):
    source_id: str
    source_kind: str
    target_id: str
    target_kind: str
    kind: str


class SourceDataDTO(DefaultBaseModel):
    guidelines: list[SourceGuidelineDTO] = []
    journeys: list[SourceJourneyDTO] = []
    relationships: list[SourceRelationshipDTO] = []
    terms: list[ResolvedTermDTO] = []
    canned_responses: list[ResolvedCannedResponseDTO] = []
    context_variables: list[ResolvedContextVariableDTO] = []


class PlaybookResolveResponseDTO(DefaultBaseModel):
    """Full resolved + source data for a playbook."""

    resolved_guidelines: list[ResolvedGuidelineDTO] = []
    relationships: list[ResolvedRelationshipDTO] = []
    terms: list[ResolvedTermDTO] = []
    canned_responses: list[ResolvedCannedResponseDTO] = []
    context_variables: list[ResolvedContextVariableDTO] = []
    source: SourceDataDTO = SourceDataDTO()


# ---------------------------------------------------------------------------
# Helper: walk playbook chain and collect tags
# ---------------------------------------------------------------------------


async def _resolve_playbook_chain(
    playbook_store: PlaybookStore,
    playbook_id: PlaybookId,
) -> list[Playbook]:
    """Returns playbook chain from child to root (most specific first)."""
    playbook_chain: list[Playbook] = []
    current_id: Optional[PlaybookId] = playbook_id
    seen: set[PlaybookId] = set()

    while current_id and current_id not in seen:
        seen.add(current_id)
        playbook = await playbook_store.read_playbook(current_id)
        playbook_chain.append(playbook)
        current_id = playbook.parent_id

    return playbook_chain


async def _get_playbook_tag_ids(
    tag_store: TagStore,
    playbook_chain: list[Playbook],
) -> list[TagId]:
    """Returns actual tag IDs for the playbook chain."""
    if not playbook_chain:
        return []

    all_tags = await tag_store.list_tags()
    tag_name_to_id = {tag.name: tag.id for tag in all_tags}

    tag_ids: list[TagId] = []
    for pb in playbook_chain:
        tag_name = f"playbook:{pb.id}"
        if tag_name in tag_name_to_id:
            tag_ids.append(tag_name_to_id[tag_name])

    return tag_ids


def _get_disabled_rule_ids(
    playbook_chain: list[Playbook],
    rule_type: str,
) -> set[str]:
    """Collect disabled rule IDs of a given type from the playbook chain."""
    disabled_ids: set[str] = set()
    prefix = f"{rule_type}:"
    for pb in playbook_chain:
        for rule_ref in pb.disabled_rules:
            if rule_ref.startswith(prefix):
                disabled_ids.add(rule_ref[len(prefix) :])
    return disabled_ids


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def create_router(
    playbook_store: PlaybookStore,
    tag_store: TagStore,
    guideline_store: GuidelineStore,
    guideline_tool_association_store: GuidelineToolAssociationStore,
    relationship_store: RelationshipStore,
    glossary_store: GlossaryStore,
    canned_response_store: CannedResponseStore,
    context_variable_store: ContextVariableStore,
    journey_store: JourneyStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/{playbook_id}/resolve",
        operation_id="resolve_playbook",
        response_model=PlaybookResolveResponseDTO,
    )
    async def resolve_playbook(
        playbook_id: str,
    ) -> PlaybookResolveResponseDTO:
        """
        Resolves the full playbook state by walking the inheritance chain
        and collecting all guidelines, terms, canned responses, context
        variables, and relationships.

        Returns both resolved (flat) and source (original) data.
        """

        # 1. Walk playbook chain
        try:
            pb_chain = await _resolve_playbook_chain(
                playbook_store, PlaybookId(playbook_id)
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Playbook '{playbook_id}' not found",
            )

        # 2. Collect tags from chain
        playbook_tag_ids = await _get_playbook_tag_ids(tag_store, pb_chain)

        # 3. Collect disabled rules
        disabled_guideline_ids = _get_disabled_rule_ids(pb_chain, "guideline")
        disabled_term_ids = _get_disabled_rule_ids(pb_chain, "term")
        disabled_canrep_ids = _get_disabled_rule_ids(pb_chain, "canned_response")
        disabled_journey_ids = _get_disabled_rule_ids(pb_chain, "journey")
        disabled_cv_ids = _get_disabled_rule_ids(pb_chain, "context_variable")

        # 4. Collect source guidelines (playbook-scoped + global, minus disabled)
        playbook_guidelines: list[Guideline] = []
        if playbook_tag_ids:
            playbook_guidelines = list(
                await guideline_store.list_guidelines(tags=playbook_tag_ids)
            )
        global_guidelines = list(await guideline_store.list_guidelines(tags=[]))

        all_source_guidelines = [
            g
            for g in set(chain(playbook_guidelines, global_guidelines))
            if g.id not in disabled_guideline_ids
        ]

        # 5. Collect tool associations for guidelines
        all_tool_assocs = await guideline_tool_association_store.list_associations()
        source_guideline_ids = {g.id for g in all_source_guidelines}
        guideline_tool_map: dict[str, list[dict[str, str]]] = {}
        for a in all_tool_assocs:
            if a.guideline_id in source_guideline_ids:
                if a.guideline_id not in guideline_tool_map:
                    guideline_tool_map[a.guideline_id] = []
                guideline_tool_map[a.guideline_id].append(
                    {
                        "service_name": a.tool_id.service_name,
                        "tool_name": a.tool_id.tool_name,
                    }
                )

        # 6. Collect journeys (playbook-scoped + global, minus disabled)
        playbook_journeys: list[Journey] = []
        if playbook_tag_ids:
            playbook_journeys = list(
                await journey_store.list_journeys(tags=playbook_tag_ids)
            )
        global_journeys = list(await journey_store.list_journeys(tags=[]))

        all_journeys = [
            j
            for j in set(chain(playbook_journeys, global_journeys))
            if j.id not in disabled_journey_ids
        ]

        # 7. Build resolved guidelines (source guidelines + projected journey guidelines)
        resolved_guidelines: list[ResolvedGuidelineDTO] = []

        # Add source guidelines to resolved
        for g in all_source_guidelines:
            resolved_guidelines.append(
                ResolvedGuidelineDTO(
                    id=g.id,
                    condition=g.content.condition,
                    action=g.content.action,
                    description=g.content.description,
                    criticality=g.criticality.value if hasattr(g.criticality, "value") else str(g.criticality),
                    composition_mode=g.composition_mode.value if g.composition_mode and hasattr(g.composition_mode, "value") else (str(g.composition_mode) if g.composition_mode else None),
                    track=g.track,
                    labels=list(g.labels),
                    enabled=g.enabled,
                    metadata=dict(g.metadata),
                    tool_ids=guideline_tool_map.get(g.id, []),
                )
            )

        # Note: Journey guideline projection is not included here because
        # the resolve endpoint focuses on the source/modeling state.
        # Journey projections happen at runtime in the engine.
        # The source journeys are preserved for the nForce frontend.

        # 8. Collect relationships involving our resolved guideline IDs
        resolved_guideline_ids = {g.id for g in resolved_guidelines}
        all_relationships = await relationship_store.list_relationships(indirect=False)
        relevant_relationships: list[ResolvedRelationshipDTO] = []

        for r in all_relationships:
            source_is_guideline = r.source.kind == RelationshipEntityKind.GUIDELINE
            target_is_guideline = r.target.kind == RelationshipEntityKind.GUIDELINE
            source_relevant = source_is_guideline and r.source.id in resolved_guideline_ids
            target_relevant = target_is_guideline and r.target.id in resolved_guideline_ids

            if source_relevant or target_relevant:
                relevant_relationships.append(
                    ResolvedRelationshipDTO(
                        source_guideline_id=r.source.id,
                        target_guideline_id=r.target.id,
                        kind=r.kind.value if hasattr(r.kind, "value") else str(r.kind),
                    )
                )

        # 9. Collect terms (playbook-scoped + global, minus disabled)
        playbook_terms: list[Term] = []
        if playbook_tag_ids:
            playbook_terms = list(await glossary_store.list_terms(tags=playbook_tag_ids))
        global_terms = list(await glossary_store.list_terms(tags=[]))

        all_terms = [
            t
            for t in set(chain(playbook_terms, global_terms))
            if t.id not in disabled_term_ids
        ]

        resolved_terms = [
            ResolvedTermDTO(
                id=t.id,
                name=t.name,
                description=t.description,
                synonyms=list(t.synonyms),
            )
            for t in all_terms
        ]

        # 10. Collect canned responses (playbook-scoped + global, minus disabled)
        playbook_canreps: list[CannedResponse] = []
        if playbook_tag_ids:
            playbook_canreps = list(
                await canned_response_store.list_canned_responses(tags=playbook_tag_ids)
            )
        global_canreps = list(
            await canned_response_store.list_canned_responses(tags=[])
        )

        all_canreps = [
            cr
            for cr in set(chain(playbook_canreps, global_canreps))
            if cr.id not in disabled_canrep_ids
        ]

        resolved_canreps = [
            ResolvedCannedResponseDTO(
                id=cr.id,
                value=cr.value,
                fields=[
                    {
                        "name": f.name,
                        "description": f.description,
                        "examples": list(f.examples),
                    }
                    for f in cr.fields
                ],
                signals=list(cr.signals),
                field_dependencies=list(cr.field_dependencies),
                metadata=dict(cr.metadata),
            )
            for cr in all_canreps
        ]

        # 11. Collect context variables (playbook-scoped + global, minus disabled)
        playbook_cvs: list[ContextVariable] = []
        if playbook_tag_ids:
            playbook_cvs = list(
                await context_variable_store.list_variables(tags=playbook_tag_ids)
            )
        global_cvs = list(await context_variable_store.list_variables(tags=[]))

        all_cvs = [
            cv
            for cv in set(chain(playbook_cvs, global_cvs))
            if cv.id not in disabled_cv_ids
        ]

        resolved_cvs = [
            ResolvedContextVariableDTO(
                id=cv.id,
                name=cv.name,
                description=cv.description,
                tool_id=(
                    {
                        "service_name": cv.tool_id.service_name,
                        "tool_name": cv.tool_id.tool_name,
                    }
                    if cv.tool_id
                    else None
                ),
                freshness_rules=cv.freshness_rules,
            )
            for cv in all_cvs
        ]

        # 12. Build source data
        source_guidelines = [
            SourceGuidelineDTO(
                id=g.id,
                condition=g.content.condition,
                action=g.content.action,
                description=g.content.description,
                criticality=g.criticality.value if hasattr(g.criticality, "value") else str(g.criticality),
                composition_mode=g.composition_mode.value if g.composition_mode and hasattr(g.composition_mode, "value") else (str(g.composition_mode) if g.composition_mode else None),
                track=g.track,
                labels=list(g.labels),
                tags=list(g.tags),
                tool_ids=guideline_tool_map.get(g.id, []),
            )
            for g in all_source_guidelines
        ]

        source_journeys: list[SourceJourneyDTO] = []
        for j in all_journeys:
            try:
                journey_nodes = await journey_store.list_nodes(j.id)
                journey_edges = await journey_store.list_edges(j.id)
                nodes = [
                    SourceJourneyNodeDTO(
                        id=n.id,
                        action=n.action,
                        description=n.description,
                        tools=[
                            f"{t.service_name}:{t.tool_name}"
                            for t in n.tools
                        ],
                        composition_mode=n.composition_mode.value if n.composition_mode and hasattr(n.composition_mode, "value") else None,
                        metadata=dict(n.metadata),
                    )
                    for n in journey_nodes
                ]
                edges = [
                    SourceJourneyEdgeDTO(
                        id=e.id,
                        source=e.source,
                        target=e.target,
                        condition=e.condition,
                        metadata=dict(e.metadata),
                    )
                    for e in journey_edges
                ]
            except Exception:
                nodes = []
                edges = []

            source_journeys.append(
                SourceJourneyDTO(
                    id=j.id,
                    title=j.title,
                    description=j.description,
                    conditions=list(j.conditions),
                    tags=list(j.tags),
                    composition_mode=j.composition_mode.value if j.composition_mode and hasattr(j.composition_mode, "value") else None,
                    nodes=nodes,
                    edges=edges,
                )
            )

        source_relationships = [
            SourceRelationshipDTO(
                source_id=r.source.id,
                source_kind=r.source.kind.value if hasattr(r.source.kind, "value") else str(r.source.kind),
                target_id=r.target.id,
                target_kind=r.target.kind.value if hasattr(r.target.kind, "value") else str(r.target.kind),
                kind=r.kind.value if hasattr(r.kind, "value") else str(r.kind),
            )
            for r in all_relationships
            if (
                (r.source.kind == RelationshipEntityKind.GUIDELINE and r.source.id in resolved_guideline_ids)
                or (r.target.kind == RelationshipEntityKind.GUIDELINE and r.target.id in resolved_guideline_ids)
            )
        ]

        return PlaybookResolveResponseDTO(
            resolved_guidelines=resolved_guidelines,
            relationships=relevant_relationships,
            terms=resolved_terms,
            canned_responses=resolved_canreps,
            context_variables=resolved_cvs,
            source=SourceDataDTO(
                guidelines=source_guidelines,
                journeys=source_journeys,
                relationships=source_relationships,
                terms=resolved_terms,
                canned_responses=resolved_canreps,
                context_variables=resolved_cvs,
            ),
        )

    return router
