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
API endpoints for agent-tool associations.

These endpoints manage direct tool associations with agents,
used by simple agents that bypass the guideline engine.
"""

from typing import Annotated, Sequence, TypeAlias

from fastapi import APIRouter, Path, Request, status
from pydantic import Field

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import (
    ExampleJson,
    apigen_config,
    example_json_content,
)
from parlant.core.agents import AgentId
from parlant.core.common import DefaultBaseModel, ItemNotFoundError, UniqueId
from parlant.core.tools import ToolId
from parlant.core.agent_tool_associations import (
    AgentToolAssociationId,
    AgentToolAssociationStore,
)

API_GROUP = "agent_tools"

AgentToolAssociationIdPath: TypeAlias = Annotated[
    AgentToolAssociationId,
    Path(
        description="Unique identifier for the agent-tool association",
        examples=["ata_abc123"],
        min_length=1,
    ),
]

AgentIdPath: TypeAlias = Annotated[
    AgentId,
    Path(
        description="Unique identifier for the agent",
        examples=["IUCGT-lvpS"],
        min_length=1,
    ),
]

ToolIdField: TypeAlias = Annotated[
    str,
    Field(
        description="Tool ID in the format 'service_name:tool_name'",
        examples=["my-service:get_weather"],
    ),
]

agent_tool_association_example: ExampleJson = {
    "id": "ata_abc123",
    "creation_utc": "2024-03-24T12:00:00Z",
    "agent_id": "IUCGT-lvpS",
    "tool_id": "my-service:get_weather",
}


class AgentToolAssociationDTO(
    DefaultBaseModel,
    json_schema_extra={"example": agent_tool_association_example},
):
    """
    Represents an association between an agent and a tool.

    Used for simple agents that bypass the guideline engine.
    """

    id: AgentToolAssociationIdPath
    creation_utc: str
    agent_id: str
    tool_id: ToolIdField


agent_tool_association_create_params_example: ExampleJson = {
    "tool_id": "my-service:get_weather",
}


class AgentToolAssociationCreateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": agent_tool_association_create_params_example},
):
    """
    Parameters for creating an agent-tool association.
    """

    tool_id: ToolIdField


def create_router(
    authorization_policy: AuthorizationPolicy,
    association_store: AgentToolAssociationStore,
) -> APIRouter:
    """Create the agent-tool associations API router."""
    router = APIRouter()

    @router.post(
        "/{agent_id}/tools",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_agent_tool_association",
        response_model=AgentToolAssociationDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Tool association successfully created.",
                "content": example_json_content(agent_tool_association_example),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_agent_tool_association(
        request: Request,
        agent_id: AgentIdPath,
        params: AgentToolAssociationCreateParamsDTO,
    ) -> AgentToolAssociationDTO:
        """
        Associates a tool with an agent for use in simple agent mode.

        Simple agents (tagged with "simple-agent") use these associations
        instead of the guideline-based tool assignment.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_AGENT,
        )

        tool_id = ToolId.from_string(params.tool_id)

        association = await association_store.create_association(
            agent_id=agent_id,
            tool_id=tool_id,
        )

        return AgentToolAssociationDTO(
            id=association.id,
            creation_utc=association.creation_utc.isoformat(),
            agent_id=association.agent_id,
            tool_id=association.tool_id.to_string(),
        )

    @router.get(
        "/{agent_id}/tools",
        operation_id="list_agent_tool_associations",
        response_model=Sequence[AgentToolAssociationDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of tool associations for the agent.",
                "content": example_json_content([agent_tool_association_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_agent_tool_associations(
        request: Request,
        agent_id: AgentIdPath,
    ) -> Sequence[AgentToolAssociationDTO]:
        """
        Lists all tools associated with an agent.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_AGENT,
        )

        associations = await association_store.list_associations(agent_id=agent_id)

        return [
            AgentToolAssociationDTO(
                id=a.id,
                creation_utc=a.creation_utc.isoformat(),
                agent_id=a.agent_id,
                tool_id=a.tool_id.to_string(),
            )
            for a in associations
        ]

    @router.delete(
        "/{agent_id}/tools/{association_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_agent_tool_association",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Tool association successfully deleted.",
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Association not found.",
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_agent_tool_association(
        request: Request,
        agent_id: AgentIdPath,
        association_id: AgentToolAssociationIdPath,
    ) -> None:
        """
        Removes a tool association from an agent.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_AGENT,
        )

        # Verify the association exists and belongs to this agent
        association = await association_store.read_association(association_id)
        if association.agent_id != agent_id:
            raise ItemNotFoundError(item_id=UniqueId(association_id))

        await association_store.delete_association(association_id)

    return router
