# Copyright 2026 Emcie Co Ltd.
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
Agent-Tool Associations for Simple Agents.

This module provides a store for associating tools directly with agents,
bypassing the guideline-based tool association used in standard Parlant.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self

from parlant.core.agents import AgentId
from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, Version, IdGenerator, UniqueId
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.tools import ToolId

AgentToolAssociationId = NewType("AgentToolAssociationId", str)


@dataclass(frozen=True)
class AgentToolAssociation:
    """Represents an association between an agent and a tool."""

    id: AgentToolAssociationId
    creation_utc: datetime
    agent_id: AgentId
    tool_id: ToolId

    def __hash__(self) -> int:
        return hash(self.id)


class AgentToolAssociationStore(ABC):
    """Abstract store for agent-tool associations."""

    @abstractmethod
    async def create_association(
        self,
        agent_id: AgentId,
        tool_id: ToolId,
        creation_utc: Optional[datetime] = None,
    ) -> AgentToolAssociation:
        """Create a new association between an agent and a tool."""
        ...

    @abstractmethod
    async def read_association(
        self,
        association_id: AgentToolAssociationId,
    ) -> AgentToolAssociation:
        """Read an association by ID."""
        ...

    @abstractmethod
    async def delete_association(
        self,
        association_id: AgentToolAssociationId,
    ) -> None:
        """Delete an association by ID."""
        ...

    @abstractmethod
    async def list_associations(
        self,
        agent_id: Optional[AgentId] = None,
    ) -> Sequence[AgentToolAssociation]:
        """List all associations, optionally filtered by agent_id."""
        ...

    @abstractmethod
    async def find_tools_for_agent(
        self,
        agent_id: AgentId,
    ) -> Sequence[ToolId]:
        """Find all tool IDs associated with a specific agent."""
        ...


class _AgentToolAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    agent_id: AgentId
    tool_id: str


class AgentToolAssociationDocumentStore(AgentToolAssociationStore):
    """Document-based implementation of the agent-tool association store."""

    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
    ) -> None:
        self._id_generator = id_generator
        self._database = database
        self._collection: DocumentCollection[_AgentToolAssociationDocument]
        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()

    async def _document_loader(
        self,
        doc: BaseDocument,
    ) -> Optional[_AgentToolAssociationDocument]:
        if doc["version"] == "0.1.0":
            return cast(_AgentToolAssociationDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._database.get_or_create_collection(
                name="agent_tool_associations",
                schema=_AgentToolAssociationDocument,
                document_loader=self._document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize(
        self,
        association: AgentToolAssociation,
    ) -> _AgentToolAssociationDocument:
        return _AgentToolAssociationDocument(
            id=ObjectId(association.id),
            version=self.VERSION.to_string(),
            creation_utc=association.creation_utc.isoformat(),
            agent_id=association.agent_id,
            tool_id=association.tool_id.to_string(),
        )

    def _deserialize(
        self,
        association_document: _AgentToolAssociationDocument,
    ) -> AgentToolAssociation:
        return AgentToolAssociation(
            id=AgentToolAssociationId(association_document["id"]),
            creation_utc=datetime.fromisoformat(association_document["creation_utc"]),
            agent_id=association_document["agent_id"],
            tool_id=ToolId.from_string(association_document["tool_id"]),
        )

    @override
    async def create_association(
        self,
        agent_id: AgentId,
        tool_id: ToolId,
        creation_utc: Optional[datetime] = None,
    ) -> AgentToolAssociation:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            # Check if association already exists
            existing = await self._collection.find_one(
                filters={
                    "$and": [
                        {"agent_id": {"$eq": agent_id}},
                        {"tool_id": {"$eq": tool_id.to_string()}},
                    ]
                }
            )
            if existing:
                return self._deserialize(existing)

            association_checksum = f"{agent_id}{tool_id.to_string()}"

            association = AgentToolAssociation(
                id=AgentToolAssociationId(self._id_generator.generate(association_checksum)),
                creation_utc=creation_utc,
                agent_id=agent_id,
                tool_id=tool_id,
            )

            await self._collection.insert_one(document=self._serialize(association))

        return association

    @override
    async def read_association(
        self,
        association_id: AgentToolAssociationId,
    ) -> AgentToolAssociation:
        async with self._lock.reader_lock:
            document = await self._collection.find_one(filters={"id": {"$eq": association_id}})

        if not document:
            raise ItemNotFoundError(item_id=UniqueId(association_id))

        return self._deserialize(document)

    @override
    async def delete_association(
        self,
        association_id: AgentToolAssociationId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._collection.delete_one(filters={"id": {"$eq": association_id}})

        if not result.deleted_document:
            raise ItemNotFoundError(item_id=UniqueId(association_id))

    @override
    async def list_associations(
        self,
        agent_id: Optional[AgentId] = None,
    ) -> Sequence[AgentToolAssociation]:
        async with self._lock.reader_lock:
            if agent_id:
                docs = await self._collection.find(filters={"agent_id": {"$eq": agent_id}})
            else:
                docs = await self._collection.find(filters={})
            return [self._deserialize(d) for d in docs]

    @override
    async def find_tools_for_agent(
        self,
        agent_id: AgentId,
    ) -> Sequence[ToolId]:
        associations = await self.list_associations(agent_id=agent_id)
        return [a.tool_id for a in associations]
