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

from typing import AsyncIterator

from pytest import fixture, raises

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.common import IdGenerator, ItemNotFoundError
from parlant.core.tools import ToolId
from parlant.core.agent_tool_associations import (
    AgentToolAssociationDocumentStore,
    AgentToolAssociationId,
    AgentToolAssociationStore,
)


@fixture
def id_generator() -> IdGenerator:
    return IdGenerator()


@fixture
async def association_store(
    id_generator: IdGenerator,
) -> AsyncIterator[AgentToolAssociationStore]:
    async with AgentToolAssociationDocumentStore(
        id_generator=id_generator,
        database=TransientDocumentDatabase(),
    ) as store:
        yield store


async def test_that_association_can_be_created(
    association_store: AgentToolAssociationStore,
) -> None:
    agent_id = AgentId("agent-123")
    tool_id = ToolId(service_name="test-service", tool_name="test-tool")

    association = await association_store.create_association(
        agent_id=agent_id,
        tool_id=tool_id,
    )

    assert association.agent_id == agent_id
    assert association.tool_id == tool_id
    assert association.id is not None
    assert association.creation_utc is not None


async def test_that_association_can_be_read(
    association_store: AgentToolAssociationStore,
) -> None:
    agent_id = AgentId("agent-123")
    tool_id = ToolId(service_name="test-service", tool_name="test-tool")

    created = await association_store.create_association(
        agent_id=agent_id,
        tool_id=tool_id,
    )

    read = await association_store.read_association(created.id)

    assert read.id == created.id
    assert read.agent_id == created.agent_id
    assert read.tool_id == created.tool_id


async def test_that_reading_nonexistent_association_raises_error(
    association_store: AgentToolAssociationStore,
) -> None:
    with raises(ItemNotFoundError):
        await association_store.read_association(AgentToolAssociationId("nonexistent-id"))


async def test_that_association_can_be_deleted(
    association_store: AgentToolAssociationStore,
) -> None:
    agent_id = AgentId("agent-123")
    tool_id = ToolId(service_name="test-service", tool_name="test-tool")

    association = await association_store.create_association(
        agent_id=agent_id,
        tool_id=tool_id,
    )

    await association_store.delete_association(association.id)

    with raises(ItemNotFoundError):
        await association_store.read_association(association.id)


async def test_that_deleting_nonexistent_association_raises_error(
    association_store: AgentToolAssociationStore,
) -> None:
    with raises(ItemNotFoundError):
        await association_store.delete_association(AgentToolAssociationId("nonexistent-id"))


async def test_that_associations_can_be_listed_for_agent(
    association_store: AgentToolAssociationStore,
) -> None:
    agent_id_1 = AgentId("agent-123")
    agent_id_2 = AgentId("agent-456")
    tool_id_1 = ToolId(service_name="service-1", tool_name="tool-1")
    tool_id_2 = ToolId(service_name="service-2", tool_name="tool-2")

    await association_store.create_association(agent_id=agent_id_1, tool_id=tool_id_1)
    await association_store.create_association(agent_id=agent_id_1, tool_id=tool_id_2)
    await association_store.create_association(agent_id=agent_id_2, tool_id=tool_id_1)

    associations = await association_store.list_associations(agent_id=agent_id_1)

    assert len(associations) == 2
    tool_ids = {a.tool_id for a in associations}
    assert tool_id_1 in tool_ids
    assert tool_id_2 in tool_ids


async def test_that_listing_associations_for_agent_with_none_returns_empty(
    association_store: AgentToolAssociationStore,
) -> None:
    associations = await association_store.list_associations(agent_id=AgentId("agent-123"))

    assert len(associations) == 0


async def test_that_tools_can_be_found_for_agent(
    association_store: AgentToolAssociationStore,
) -> None:
    agent_id = AgentId("agent-123")
    tool_id_1 = ToolId(service_name="service-1", tool_name="tool-1")
    tool_id_2 = ToolId(service_name="service-2", tool_name="tool-2")

    await association_store.create_association(agent_id=agent_id, tool_id=tool_id_1)
    await association_store.create_association(agent_id=agent_id, tool_id=tool_id_2)

    tool_ids = await association_store.find_tools_for_agent(agent_id=agent_id)

    assert len(tool_ids) == 2
    assert tool_id_1 in tool_ids
    assert tool_id_2 in tool_ids


async def test_that_finding_tools_for_agent_with_none_returns_empty(
    association_store: AgentToolAssociationStore,
) -> None:
    tool_ids = await association_store.find_tools_for_agent(agent_id=AgentId("agent-123"))

    assert len(tool_ids) == 0


async def test_that_multiple_agents_can_have_same_tool(
    association_store: AgentToolAssociationStore,
) -> None:
    agent_id_1 = AgentId("agent-1")
    agent_id_2 = AgentId("agent-2")
    tool_id = ToolId(service_name="shared-service", tool_name="shared-tool")

    await association_store.create_association(agent_id=agent_id_1, tool_id=tool_id)
    await association_store.create_association(agent_id=agent_id_2, tool_id=tool_id)

    tools_1 = await association_store.find_tools_for_agent(agent_id=agent_id_1)
    tools_2 = await association_store.find_tools_for_agent(agent_id=agent_id_2)

    assert tool_id in tools_1
    assert tool_id in tools_2
