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

from fastapi import status
import httpx
from lagom import Container

from parlant.core.agents import AgentStore
from parlant.core.agent_tool_associations import AgentToolAssociationStore
from parlant.core.tools import ToolId


async def test_that_tool_association_can_be_created(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent(name="Test Agent")

    response = await async_client.post(
        f"/agents/{agent.id}/tools",
        json={"tool_id": "test-service:test-tool"},
    )

    assert response.status_code == status.HTTP_201_CREATED

    association = response.json()
    assert association["agent_id"] == agent.id
    assert association["tool_id"] == "test-service:test-tool"
    assert "id" in association
    assert "creation_utc" in association


async def test_that_tool_associations_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent(name="Test Agent")

    # Create two associations
    await async_client.post(
        f"/agents/{agent.id}/tools",
        json={"tool_id": "service-1:tool-1"},
    )
    await async_client.post(
        f"/agents/{agent.id}/tools",
        json={"tool_id": "service-2:tool-2"},
    )

    response = await async_client.get(f"/agents/{agent.id}/tools")

    assert response.status_code == status.HTTP_200_OK

    associations = response.json()
    assert len(associations) == 2

    tool_ids = {a["tool_id"] for a in associations}
    assert "service-1:tool-1" in tool_ids
    assert "service-2:tool-2" in tool_ids


async def test_that_listing_associations_for_agent_with_none_returns_empty(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent(name="Test Agent")

    response = await async_client.get(f"/agents/{agent.id}/tools")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []


async def test_that_tool_association_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent(name="Test Agent")

    # Create an association
    create_response = await async_client.post(
        f"/agents/{agent.id}/tools",
        json={"tool_id": "test-service:test-tool"},
    )
    association_id = create_response.json()["id"]

    # Delete it
    delete_response = await async_client.delete(
        f"/agents/{agent.id}/tools/{association_id}"
    )

    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    # Verify it's gone
    list_response = await async_client.get(f"/agents/{agent.id}/tools")
    assert list_response.json() == []


async def test_that_deleting_nonexistent_association_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    agent = await agent_store.create_agent(name="Test Agent")

    response = await async_client.delete(
        f"/agents/{agent.id}/tools/nonexistent-id"
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_deleting_association_from_wrong_agent_returns_404(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    agent1 = await agent_store.create_agent(name="Agent 1")
    agent2 = await agent_store.create_agent(name="Agent 2")

    # Create association for agent1
    create_response = await async_client.post(
        f"/agents/{agent1.id}/tools",
        json={"tool_id": "test-service:test-tool"},
    )
    association_id = create_response.json()["id"]

    # Try to delete it using agent2's endpoint
    delete_response = await async_client.delete(
        f"/agents/{agent2.id}/tools/{association_id}"
    )

    assert delete_response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_associations_are_scoped_to_agent(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    agent1 = await agent_store.create_agent(name="Agent 1")
    agent2 = await agent_store.create_agent(name="Agent 2")

    # Create associations for both agents
    await async_client.post(
        f"/agents/{agent1.id}/tools",
        json={"tool_id": "service:tool-for-agent-1"},
    )
    await async_client.post(
        f"/agents/{agent2.id}/tools",
        json={"tool_id": "service:tool-for-agent-2"},
    )

    # List for agent1 should only show agent1's tools
    response1 = await async_client.get(f"/agents/{agent1.id}/tools")
    associations1 = response1.json()
    assert len(associations1) == 1
    assert associations1[0]["tool_id"] == "service:tool-for-agent-1"

    # List for agent2 should only show agent2's tools
    response2 = await async_client.get(f"/agents/{agent2.id}/tools")
    associations2 = response2.json()
    assert len(associations2) == 1
    assert associations2[0]["tool_id"] == "service:tool-for-agent-2"
