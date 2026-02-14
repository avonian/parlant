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
Static Playbook Store

A static playbook is a pre-resolved, self-contained snapshot of a playbook's
full state (guidelines, relationships, terms, canned responses, context variables).
It is created by nForce during the "Release" operation and consumed by Parlant's
engine at runtime, bypassing the normal tag-based resolution.

This is part of the Playbook Versioning system — see devspace/docs/PLAYBOOK_VERSIONING_PLAN.md.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, NewType, Optional, Sequence, TypedDict, cast

from typing_extensions import Self

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import JSONSerializable, Version, generate_id
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    DocumentDatabase,
    ObjectId,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper

StaticPlaybookId = NewType("StaticPlaybookId", str)


# ---------------------------------------------------------------------------
# Domain models — what the engine consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticGuideline:
    """A resolved guideline within a static playbook."""

    id: str
    condition: str
    action: str | None
    description: str | None = None
    criticality: str = "medium"
    composition_mode: str | None = None
    track: bool = True
    labels: frozenset[str] = field(default_factory=frozenset)
    tool_ids: Sequence[dict[str, str]] = ()  # [{service_name, tool_name}]
    enabled: bool = True
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticRelationship:
    """A relationship between two resolved guidelines (by index)."""

    source_guideline_id: str
    target_guideline_id: str
    kind: str  # entailment, priority, dependency, disambiguation, reevaluation, overlap


@dataclass(frozen=True)
class StaticTerm:
    """A glossary term."""

    id: str
    name: str
    description: str
    synonyms: Sequence[str] = ()


@dataclass(frozen=True)
class StaticCannedResponse:
    """A canned response."""

    id: str
    value: str
    fields: Sequence[dict[str, Any]] = ()  # [{name, description, examples}]
    signals: Sequence[str] = ()
    field_dependencies: Sequence[str] = ()
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticContextVariable:
    """A context variable definition."""

    id: str
    name: str
    description: str | None = None
    tool_id: dict[str, str] | None = None  # {service_name, tool_name}
    freshness_rules: str | None = None


@dataclass(frozen=True)
class StaticPlaybook:
    """A self-contained, pre-resolved playbook snapshot.

    Created by nForce during release, consumed by the engine at runtime.
    The engine uses this to bypass tag-based guideline resolution.
    """

    id: StaticPlaybookId
    creation_utc: datetime
    guidelines: Sequence[StaticGuideline] = ()
    relationships: Sequence[StaticRelationship] = ()
    terms: Sequence[StaticTerm] = ()
    canned_responses: Sequence[StaticCannedResponse] = ()
    context_variables: Sequence[StaticContextVariable] = ()

    def __hash__(self) -> int:
        return hash(self.id)


# ---------------------------------------------------------------------------
# Abstract store interface
# ---------------------------------------------------------------------------


class StaticPlaybookStore(ABC):
    @abstractmethod
    async def upsert_static_playbook(
        self,
        id: StaticPlaybookId,
        guidelines: Sequence[StaticGuideline],
        relationships: Sequence[StaticRelationship],
        terms: Sequence[StaticTerm],
        canned_responses: Sequence[StaticCannedResponse],
        context_variables: Sequence[StaticContextVariable],
    ) -> StaticPlaybook: ...

    @abstractmethod
    async def read_static_playbook(
        self,
        id: StaticPlaybookId,
    ) -> StaticPlaybook: ...

    @abstractmethod
    async def delete_static_playbook(
        self,
        id: StaticPlaybookId,
    ) -> None: ...

    @abstractmethod
    async def list_static_playbooks(self) -> Sequence[StaticPlaybook]: ...


# ---------------------------------------------------------------------------
# Document models — what gets persisted
# ---------------------------------------------------------------------------


class _StaticPlaybookDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    # All content stored as JSON strings for simplicity — the engine
    # deserializes on read. This avoids complex nested document schemas.
    guidelines_json: str
    relationships_json: str
    terms_json: str
    canned_responses_json: str
    context_variables_json: str


# ---------------------------------------------------------------------------
# Document store implementation
# ---------------------------------------------------------------------------


class StaticPlaybookDocumentStore(StaticPlaybookStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        database: DocumentDatabase,
        allow_migration: bool = False,
    ) -> None:
        self._database = database
        self._collection: DocumentCollection[_StaticPlaybookDocument]
        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._database.get_or_create_collection(
                name="static_playbooks",
                schema=_StaticPlaybookDocument,
                document_loader=self._document_loader,
            )
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False

    async def _document_loader(
        self, doc: BaseDocument
    ) -> Optional[_StaticPlaybookDocument]:
        doc = cast(_StaticPlaybookDocument, doc)
        if doc["version"] == "0.1.0":
            return doc
        return None

    # -- Serialization --

    def _serialize(self, playbook: StaticPlaybook) -> _StaticPlaybookDocument:
        return _StaticPlaybookDocument(
            id=ObjectId(playbook.id),
            version=self.VERSION.to_string(),
            creation_utc=playbook.creation_utc.isoformat(),
            guidelines_json=json.dumps(
                [
                    {
                        "id": g.id,
                        "condition": g.condition,
                        "action": g.action,
                        "description": g.description,
                        "criticality": g.criticality,
                        "composition_mode": g.composition_mode,
                        "track": g.track,
                        "labels": list(g.labels),
                        "tool_ids": list(g.tool_ids),
                        "enabled": g.enabled,
                        "metadata": dict(g.metadata),
                    }
                    for g in playbook.guidelines
                ]
            ),
            relationships_json=json.dumps(
                [
                    {
                        "source_guideline_id": r.source_guideline_id,
                        "target_guideline_id": r.target_guideline_id,
                        "kind": r.kind,
                    }
                    for r in playbook.relationships
                ]
            ),
            terms_json=json.dumps(
                [
                    {
                        "id": t.id,
                        "name": t.name,
                        "description": t.description,
                        "synonyms": list(t.synonyms),
                    }
                    for t in playbook.terms
                ]
            ),
            canned_responses_json=json.dumps(
                [
                    {
                        "id": cr.id,
                        "value": cr.value,
                        "fields": list(cr.fields),
                        "signals": list(cr.signals),
                        "field_dependencies": list(cr.field_dependencies),
                        "metadata": dict(cr.metadata),
                    }
                    for cr in playbook.canned_responses
                ]
            ),
            context_variables_json=json.dumps(
                [
                    {
                        "id": cv.id,
                        "name": cv.name,
                        "description": cv.description,
                        "tool_id": cv.tool_id,
                        "freshness_rules": cv.freshness_rules,
                    }
                    for cv in playbook.context_variables
                ]
            ),
        )

    def _deserialize(self, doc: _StaticPlaybookDocument) -> StaticPlaybook:
        guidelines_data = json.loads(doc.get("guidelines_json", "[]"))
        relationships_data = json.loads(doc.get("relationships_json", "[]"))
        terms_data = json.loads(doc.get("terms_json", "[]"))
        canned_responses_data = json.loads(doc.get("canned_responses_json", "[]"))
        context_variables_data = json.loads(doc.get("context_variables_json", "[]"))

        return StaticPlaybook(
            id=StaticPlaybookId(doc["id"]),
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
            guidelines=tuple(
                StaticGuideline(
                    id=g["id"],
                    condition=g["condition"],
                    action=g.get("action"),
                    description=g.get("description"),
                    criticality=g.get("criticality", "medium"),
                    composition_mode=g.get("composition_mode"),
                    track=g.get("track", True),
                    labels=frozenset(g.get("labels", [])),
                    tool_ids=tuple(g.get("tool_ids", [])),
                    enabled=g.get("enabled", True),
                    metadata=g.get("metadata", {}),
                )
                for g in guidelines_data
            ),
            relationships=tuple(
                StaticRelationship(
                    source_guideline_id=r["source_guideline_id"],
                    target_guideline_id=r["target_guideline_id"],
                    kind=r["kind"],
                )
                for r in relationships_data
            ),
            terms=tuple(
                StaticTerm(
                    id=t["id"],
                    name=t["name"],
                    description=t["description"],
                    synonyms=tuple(t.get("synonyms", [])),
                )
                for t in terms_data
            ),
            canned_responses=tuple(
                StaticCannedResponse(
                    id=cr["id"],
                    value=cr["value"],
                    fields=tuple(cr.get("fields", [])),
                    signals=tuple(cr.get("signals", [])),
                    field_dependencies=tuple(cr.get("field_dependencies", [])),
                    metadata=cr.get("metadata", {}),
                )
                for cr in canned_responses_data
            ),
            context_variables=tuple(
                StaticContextVariable(
                    id=cv["id"],
                    name=cv["name"],
                    description=cv.get("description"),
                    tool_id=cv.get("tool_id"),
                    freshness_rules=cv.get("freshness_rules"),
                )
                for cv in context_variables_data
            ),
        )

    # -- CRUD --

    async def upsert_static_playbook(
        self,
        id: StaticPlaybookId,
        guidelines: Sequence[StaticGuideline] = (),
        relationships: Sequence[StaticRelationship] = (),
        terms: Sequence[StaticTerm] = (),
        canned_responses: Sequence[StaticCannedResponse] = (),
        context_variables: Sequence[StaticContextVariable] = (),
    ) -> StaticPlaybook:
        creation_utc = datetime.now(timezone.utc)

        playbook = StaticPlaybook(
            id=id,
            creation_utc=creation_utc,
            guidelines=guidelines,
            relationships=relationships,
            terms=terms,
            canned_responses=canned_responses,
            context_variables=context_variables,
        )

        doc = self._serialize(playbook)

        async with self._lock.writer_lock:
            existing = await self._collection.find_one(
                filters={"id": {"$eq": id}},
            )

            if existing:
                await self._collection.update_one(
                    filters={"id": {"$eq": id}},
                    params={
                        "creation_utc": doc["creation_utc"],
                        "guidelines_json": doc["guidelines_json"],
                        "relationships_json": doc["relationships_json"],
                        "terms_json": doc["terms_json"],
                        "canned_responses_json": doc["canned_responses_json"],
                        "context_variables_json": doc["context_variables_json"],
                    },
                )
            else:
                await self._collection.insert_one(document=doc)

        return playbook

    async def read_static_playbook(
        self,
        id: StaticPlaybookId,
    ) -> StaticPlaybook:
        async with self._lock.reader_lock:
            doc = await self._collection.find_one(
                filters={"id": {"$eq": id}},
            )

        if not doc:
            from parlant.core.common import ItemNotFoundError, UniqueId

            raise ItemNotFoundError(item_id=UniqueId(id))

        return self._deserialize(doc)

    async def delete_static_playbook(
        self,
        id: StaticPlaybookId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._collection.delete_one(
                filters={"id": {"$eq": id}},
            )

        if result.deleted_count == 0:
            from parlant.core.common import ItemNotFoundError, UniqueId

            raise ItemNotFoundError(item_id=UniqueId(id))

    async def list_static_playbooks(self) -> Sequence[StaticPlaybook]:
        async with self._lock.reader_lock:
            docs = await self._collection.find(filters={})

        return [self._deserialize(doc) for doc in docs]
