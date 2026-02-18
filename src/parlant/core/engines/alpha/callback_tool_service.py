"""
Callback-based ToolService for the stateless /v2/process endpoint.

When the engine needs to call a tool, this service POSTs to nForce's
callback URL instead of using the SDK plugin protocol.
"""

from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import httpx
from typing_extensions import override

from parlant.core.common import ItemNotFoundError, JSONSerializable, UniqueId
from parlant.core.tools import (
    Tool,
    ToolContext,
    ToolExecutionError,
    ToolId,
    ToolOverlap,
    ToolParameterDescriptor,
    ToolParameterOptions,
    ToolResult,
    ToolService,
)


class CallbackToolService(ToolService):
    """
    ToolService that delegates tool execution to nForce via HTTP callback.

    Tools are defined inline in the request payload. When the engine calls
    a tool, we POST to the callback URL with the tool name and arguments.
    nForce executes the tool and returns the result.
    """

    def __init__(
        self,
        service_name: str,
        callback_url: str,
        tools: dict[str, Tool],
        metadata: dict[str, Any],
    ) -> None:
        self._service_name = service_name
        self._callback_url = callback_url
        self._tools = tools
        self._metadata = metadata

    @override
    async def list_tools(self) -> Sequence[Tool]:
        return list(self._tools.values())

    @override
    async def read_tool(self, name: str) -> Tool:
        if name not in self._tools:
            raise ItemNotFoundError(item_id=UniqueId(name))
        return self._tools[name]

    @override
    async def resolve_tool(self, name: str, context: ToolContext) -> Tool:
        return await self.read_tool(name)

    @override
    async def call_tool(
        self,
        name: str,
        context: ToolContext,
        arguments: Mapping[str, JSONSerializable],
    ) -> ToolResult:
        tool = await self.read_tool(name)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    self._callback_url,
                    json={
                        "tool_id": f"{self._service_name}:{name}",
                        "arguments": dict(arguments),
                        "context": {
                            "agent_id": context.agent_id,
                            "session_id": context.session_id,
                            "customer_id": context.customer_id,
                            "metadata": self._metadata,
                        },
                    },
                )
                resp.raise_for_status()
                body = resp.json()
        except httpx.HTTPStatusError as e:
            raise ToolExecutionError(
                name, f"Callback returned {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise ToolExecutionError(name, f"Callback failed: {e}")

        return ToolResult(
            data=body.get("data"),
            metadata=body.get("metadata", {}),
            control=body.get("control", {}),
            canned_responses=body.get("canned_responses", []),
            canned_response_fields=body.get("canned_response_fields", {}),
        )
