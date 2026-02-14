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
Static Playbook CRUD API

These endpoints are called by nForce to push/read/delete pre-resolved
playbook snapshots. The engine reads from the store at runtime.
"""

from typing import Any, Sequence

from fastapi import APIRouter, HTTPException, Path, status
from pydantic import Field

from parlant.core.common import DefaultBaseModel
from parlant.core.static_playbooks import (
    StaticCannedResponse,
    StaticContextVariable,
    StaticGuideline,
    StaticPlaybook,
    StaticPlaybookId,
    StaticPlaybookStore,
    StaticRelationship,
    StaticTerm,
)


# -- DTOs --


class StaticGuidelineDTO(DefaultBaseModel):
    id: str
    condition: str
    action: str | None = None
    description: str | None = None
    criticality: str = "medium"
    composition_mode: str | None = None
    track: bool = True
    labels: list[str] = []
    tool_ids: list[dict[str, str]] = []
    enabled: bool = True
    metadata: dict[str, Any] = {}


class StaticRelationshipDTO(DefaultBaseModel):
    source_guideline_id: str
    target_guideline_id: str
    kind: str


class StaticTermDTO(DefaultBaseModel):
    id: str
    name: str
    description: str
    synonyms: list[str] = []


class StaticCannedResponseDTO(DefaultBaseModel):
    id: str
    value: str
    fields: list[dict[str, Any]] = []
    signals: list[str] = []
    field_dependencies: list[str] = []
    metadata: dict[str, Any] = {}


class StaticContextVariableDTO(DefaultBaseModel):
    id: str
    name: str
    description: str | None = None
    tool_id: dict[str, str] | None = None
    freshness_rules: str | None = None


class StaticPlaybookDTO(DefaultBaseModel):
    id: str
    creation_utc: str
    guidelines: list[StaticGuidelineDTO] = []
    relationships: list[StaticRelationshipDTO] = []
    terms: list[StaticTermDTO] = []
    canned_responses: list[StaticCannedResponseDTO] = []
    context_variables: list[StaticContextVariableDTO] = []


class UpsertStaticPlaybookDTO(DefaultBaseModel):
    guidelines: list[StaticGuidelineDTO] = []
    relationships: list[StaticRelationshipDTO] = []
    terms: list[StaticTermDTO] = []
    canned_responses: list[StaticCannedResponseDTO] = []
    context_variables: list[StaticContextVariableDTO] = []


# -- Conversions --


def _to_domain_guideline(dto: StaticGuidelineDTO) -> StaticGuideline:
    return StaticGuideline(
        id=dto.id,
        condition=dto.condition,
        action=dto.action,
        description=dto.description,
        criticality=dto.criticality,
        composition_mode=dto.composition_mode,
        track=dto.track,
        labels=frozenset(dto.labels),
        tool_ids=tuple(dto.tool_ids),
        enabled=dto.enabled,
        metadata=dto.metadata,
    )


def _to_dto_guideline(g: StaticGuideline) -> StaticGuidelineDTO:
    return StaticGuidelineDTO(
        id=g.id,
        condition=g.condition,
        action=g.action,
        description=g.description,
        criticality=g.criticality,
        composition_mode=g.composition_mode,
        track=g.track,
        labels=list(g.labels),
        tool_ids=list(g.tool_ids),
        enabled=g.enabled,
        metadata=dict(g.metadata),
    )


def _to_dto(playbook: StaticPlaybook) -> StaticPlaybookDTO:
    return StaticPlaybookDTO(
        id=playbook.id,
        creation_utc=playbook.creation_utc.isoformat(),
        guidelines=[_to_dto_guideline(g) for g in playbook.guidelines],
        relationships=[
            StaticRelationshipDTO(
                source_guideline_id=r.source_guideline_id,
                target_guideline_id=r.target_guideline_id,
                kind=r.kind,
            )
            for r in playbook.relationships
        ],
        terms=[
            StaticTermDTO(
                id=t.id,
                name=t.name,
                description=t.description,
                synonyms=list(t.synonyms),
            )
            for t in playbook.terms
        ],
        canned_responses=[
            StaticCannedResponseDTO(
                id=cr.id,
                value=cr.value,
                fields=list(cr.fields),
                signals=list(cr.signals),
                field_dependencies=list(cr.field_dependencies),
                metadata=dict(cr.metadata),
            )
            for cr in playbook.canned_responses
        ],
        context_variables=[
            StaticContextVariableDTO(
                id=cv.id,
                name=cv.name,
                description=cv.description,
                tool_id=cv.tool_id,
                freshness_rules=cv.freshness_rules,
            )
            for cv in playbook.context_variables
        ],
    )


# -- Router --


def create_router(
    static_playbook_store: StaticPlaybookStore,
) -> APIRouter:
    router = APIRouter()

    @router.put(
        "/{static_playbook_id}",
        status_code=status.HTTP_200_OK,
        operation_id="upsert_static_playbook",
        response_model=StaticPlaybookDTO,
    )
    async def upsert_static_playbook(
        static_playbook_id: str,
        params: UpsertStaticPlaybookDTO,
    ) -> StaticPlaybookDTO:
        playbook = await static_playbook_store.upsert_static_playbook(
            id=StaticPlaybookId(static_playbook_id),
            guidelines=[_to_domain_guideline(g) for g in params.guidelines],
            relationships=[
                StaticRelationship(
                    source_guideline_id=r.source_guideline_id,
                    target_guideline_id=r.target_guideline_id,
                    kind=r.kind,
                )
                for r in params.relationships
            ],
            terms=[
                StaticTerm(
                    id=t.id,
                    name=t.name,
                    description=t.description,
                    synonyms=tuple(t.synonyms),
                )
                for t in params.terms
            ],
            canned_responses=[
                StaticCannedResponse(
                    id=cr.id,
                    value=cr.value,
                    fields=tuple(cr.fields),
                    signals=tuple(cr.signals),
                    field_dependencies=tuple(cr.field_dependencies),
                    metadata=cr.metadata,
                )
                for cr in params.canned_responses
            ],
            context_variables=[
                StaticContextVariable(
                    id=cv.id,
                    name=cv.name,
                    description=cv.description,
                    tool_id=cv.tool_id,
                    freshness_rules=cv.freshness_rules,
                )
                for cv in params.context_variables
            ],
        )
        return _to_dto(playbook)

    @router.get(
        "/{static_playbook_id}",
        operation_id="read_static_playbook",
        response_model=StaticPlaybookDTO,
    )
    async def read_static_playbook(
        static_playbook_id: str,
    ) -> StaticPlaybookDTO:
        try:
            playbook = await static_playbook_store.read_static_playbook(
                id=StaticPlaybookId(static_playbook_id),
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Static playbook '{static_playbook_id}' not found",
            )
        return _to_dto(playbook)

    @router.delete(
        "/{static_playbook_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_static_playbook",
    )
    async def delete_static_playbook(
        static_playbook_id: str,
    ) -> None:
        try:
            await static_playbook_store.delete_static_playbook(
                id=StaticPlaybookId(static_playbook_id),
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Static playbook '{static_playbook_id}' not found",
            )

    @router.get(
        "",
        operation_id="list_static_playbooks",
        response_model=list[StaticPlaybookDTO],
    )
    async def list_static_playbooks() -> list[StaticPlaybookDTO]:
        playbooks = await static_playbook_store.list_static_playbooks()
        return [_to_dto(p) for p in playbooks]

    return router
