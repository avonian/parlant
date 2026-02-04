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
Simple Agent - A lightweight agent mode that bypasses Parlant's guideline engine
and uses a traditional tool-calling loop with LiteLLM.

Requirements:
    - Parlant must be configured with `nlp_service = "litellm"` in parlant.toml
    - Environment variables LITELLM_PROVIDER_MODEL_NAME and LITELLM_PROVIDER_API_KEY must be set

Usage:
    1. Tag an agent with "simple-agent" to enable this mode.
    2. Set the agent's description as the system prompt.
    3. Associate tools with the agent using the /agents/{agent_id}/tools API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, cast

import litellm  # type: ignore[import-not-found]

from parlant.core.agents import Agent, AgentId
from parlant.core.context_variables import ContextVariable, ContextVariableStore, ContextVariableValue
from parlant.core.engines.alpha.engine_context import EngineContext
from parlant.core.engines.alpha.hooks import EngineHooks, EngineHookResult
from parlant.core.loggers import Logger
from parlant.core.sessions import (
    EventKind,
    EventSource,
    MessageEventData,
    Participant,
    StatusEventData,
    ToolEventData,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import Tool, ToolContext, ToolId, ToolResult, ToolService
from parlant.core.agent_tool_associations import AgentToolAssociationStore
from parlant.core.tags import TagStore

SIMPLE_AGENT_TAG = "simple-agent"

# Environment variables - aligned with Parlant's LiteLLM service
LITELLM_MODEL_ENV = "LITELLM_PROVIDER_MODEL_NAME"
LITELLM_BASE_URL_ENV = "LITELLM_PROVIDER_BASE_URL"
SIMPLE_AGENT_MAX_ITERATIONS_ENV = "SIMPLE_AGENT_MAX_ITERATIONS"

DEFAULT_MAX_ITERATIONS = 10


class SimpleAgentConfigurationError(Exception):
    """Raised when simple agent is not properly configured."""

    pass


@dataclass
class ToolDefinition:
    """A tool definition in OpenAI format along with execution metadata."""

    service_name: str
    service: ToolService
    tool: Tool
    openai_schema: dict[str, Any]


def tool_to_openai_schema(tool: Tool) -> dict[str, Any]:
    """Convert a Parlant Tool to OpenAI tool schema format."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, (descriptor, options) in tool.parameters.items():
        param_schema: dict[str, Any] = {}

        # Map Parlant types to JSON Schema types
        type_mapping = {
            "string": "string",
            "str": "string",
            "integer": "integer",
            "int": "integer",
            "number": "number",
            "float": "number",
            "boolean": "boolean",
            "bool": "boolean",
            "array": "array",
            "list": "array",
            "object": "object",
            "dict": "object",
        }

        param_type = descriptor.get("type", "string")
        if isinstance(param_type, str):
            param_schema["type"] = type_mapping.get(param_type.lower(), "string")
        else:
            param_schema["type"] = "string"

        if "description" in descriptor:
            param_schema["description"] = descriptor["description"]

        if "enum" in descriptor:
            param_schema["enum"] = descriptor["enum"]

        properties[param_name] = param_schema

    required = list(tool.required) if tool.required else []

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or f"Tool: {tool.name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


async def gather_agent_tools(
    agent_id: AgentId,
    association_store: AgentToolAssociationStore,
    service_registry: ServiceRegistry,
) -> list[ToolDefinition]:
    """Gather tools associated with a specific agent."""
    tool_definitions: list[ToolDefinition] = []

    # Get tool IDs associated with this agent
    tool_ids = await association_store.find_tools_for_agent(agent_id)

    if not tool_ids:
        return tool_definitions

    # Build a map of service_name -> service for quick lookup
    services_by_name: dict[str, ToolService] = {}
    for service_name, service in await service_registry.list_tool_services():
        services_by_name[service_name] = service

    # Fetch each tool from its service
    for tool_id in tool_ids:
        maybe_service = services_by_name.get(tool_id.service_name)
        if maybe_service is None:
            continue

        try:
            tool = await maybe_service.read_tool(tool_id.tool_name)
            tool_definitions.append(
                ToolDefinition(
                    service_name=tool_id.service_name,
                    service=maybe_service,
                    tool=tool,
                    openai_schema=tool_to_openai_schema(tool),
                )
            )
        except Exception:
            # Skip tools that fail to load
            continue

    return tool_definitions


def build_conversation_messages(
    context: EngineContext,
) -> list[dict[str, Any]]:
    """Build conversation history from session events."""
    messages: list[dict[str, Any]] = []

    for event in context.interaction.events:
        if event.kind == EventKind.MESSAGE:
            data = cast(MessageEventData, event.data)
            role = "assistant" if event.source == EventSource.AI_AGENT else "user"
            messages.append(
                {
                    "role": role,
                    "content": data.get("message", ""),
                }
            )

    return messages


class SimpleAgentHook:
    """
    Engine hook that intercepts processing for agents tagged with "simple-agent"
    and runs a traditional LiteLLM tool-calling loop instead of the full engine.
    """

    def __init__(
        self,
        service_registry: ServiceRegistry,
        association_store: AgentToolAssociationStore,
        tag_store: TagStore,
        variable_store: ContextVariableStore,
        logger: Logger,
        model: str,
        base_url: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        self._service_registry = service_registry
        self._association_store = association_store
        self._tag_store = tag_store
        self._variable_store = variable_store
        self._logger = logger
        self._model = model
        self._base_url = base_url
        self._max_iterations = max_iterations or int(
            os.environ.get(SIMPLE_AGENT_MAX_ITERATIONS_ENV, str(DEFAULT_MAX_ITERATIONS))
        )

    async def _is_simple_agent(self, agent: Agent) -> bool:
        """Check if agent has a tag named 'simple-agent'."""
        for tag_id in agent.tags:
            try:
                tag = await self._tag_store.read_tag(tag_id)
                if tag.name == SIMPLE_AGENT_TAG:
                    return True
            except Exception:
                # Skip tags that can't be read
                continue
        return False

    async def _execute_tool(
        self,
        tool_def: ToolDefinition,
        arguments: Mapping[str, Any],
        context: EngineContext,
    ) -> ToolResult:
        """Execute a tool and return the result."""
        tool_context = ToolContext(
            agent_id=context.agent.id,
            session_id=context.session.id,
            customer_id=context.customer.id,
        )

        result = await tool_def.service.call_tool(
            name=tool_def.tool.name,
            context=tool_context,
            arguments=dict(arguments),
        )

        return result

    async def _run_tool_loop(
        self,
        context: EngineContext,
        system_prompt: str,
        tools: list[ToolDefinition],
    ) -> str:
        """Run the LiteLLM tool-calling loop."""
        self._logger.trace(
            f"SimpleAgent: Starting tool loop with model={self._model}, "
            f"tools={[t.tool.name for t in tools]}"
        )

        # Build initial messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(build_conversation_messages(context))

        # Build tool schemas
        tool_schemas = [t.openai_schema for t in tools] if tools else None
        tools_by_name = {t.tool.name: t for t in tools}

        iteration = 0
        while iteration < self._max_iterations:
            iteration += 1

            # Log the prompt
            self._logger.trace(
                f"SimpleAgent: LLM call iteration={iteration}\n"
                f"Messages:\n{json.dumps(messages, indent=2)}"
            )

            # Call LLM
            response = await litellm.acompletion(
                model=self._model,
                messages=messages,
                tools=tool_schemas,
                base_url=self._base_url,
            )

            choice = response.choices[0].message
            usage = response.usage

            # Log the response
            self._logger.trace(
                f"SimpleAgent: LLM response\n"
                f"Content: {choice.content}\n"
                f"Tool calls: {len(choice.tool_calls) if choice.tool_calls else 0}\n"
                f"Tokens: input={usage.prompt_tokens if usage else 0}, "
                f"output={usage.completion_tokens if usage else 0}"
            )

            # Check if model wants to call tools
            if choice.tool_calls:
                # Add assistant message with tool calls
                messages.append(choice.model_dump())

                # Execute each tool call
                for tool_call in choice.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    self._logger.trace(
                        f"SimpleAgent: Executing tool {function_name}\n"
                        f"Arguments: {json.dumps(function_args, indent=2)}"
                    )

                    tool_def = tools_by_name.get(function_name)
                    if tool_def:
                        try:
                            result = await self._execute_tool(tool_def, function_args, context)

                            # Add tool event to context for tracking
                            await context.add_tool_event(
                                tool_id=ToolId.from_string(
                                    f"{tool_def.service_name}:{tool_def.tool.name}"
                                ),
                                arguments=function_args,
                                result=result,
                            )

                            result_content = json.dumps(result.data)
                            self._logger.trace(f"SimpleAgent: Tool result\n{result_content}")
                        except Exception as e:
                            result_content = json.dumps({"error": str(e)})
                            self._logger.error(f"SimpleAgent: Tool error - {e}")
                    else:
                        result_content = json.dumps({"error": f"Unknown tool: {function_name}"})

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_content,
                        }
                    )
            else:
                # No tool calls - we have the final response
                self._logger.trace(f"SimpleAgent: Final response\n{choice.content}")
                return choice.content or ""

        # Max iterations reached
        self._logger.warning("SimpleAgent: Max iterations reached")
        return "I apologize, but I was unable to complete the request within the allowed number of steps."

    async def _load_context_variables(
        self,
        context: EngineContext,
    ) -> list[tuple[ContextVariable, ContextVariableValue]]:
        """Load context variable values for the customer in this session."""
        result: list[tuple[ContextVariable, ContextVariableValue]] = []

        try:
            variables = await self._variable_store.list_variables()

            keys_to_check = (
                [context.customer.id]
                + [f"tag:{tag_id}" for tag_id in context.customer.tags]
                + [ContextVariableStore.GLOBAL_KEY]
            )

            for variable in variables:
                for key in keys_to_check:
                    value = await self._variable_store.read_value(
                        variable_id=variable.id,
                        key=key,
                    )
                    if value is not None:
                        result.append((variable, value))
                        break
        except Exception as e:
            self._logger.warning(f"SimpleAgent: Failed to load context variables: {e}")

        return result

    def _format_context_variables(
        self,
        variables: list[tuple[ContextVariable, ContextVariableValue]],
    ) -> str:
        """Format context variables as a text block for the system prompt."""
        if not variables:
            return ""

        lines = [
            "\nThe following is contextual information relevant to this conversation:"
        ]
        for variable, value in variables:
            name = variable.name
            data = value.data
            if isinstance(data, dict):
                formatted = json.dumps(data)
            else:
                formatted = str(data)
            lines.append(f"- {name}: {formatted}")

        return "\n".join(lines)

    async def on_acknowledged(
        self,
        context: EngineContext,
        payload: Any,
        exc: Optional[Exception],
    ) -> EngineHookResult:
        """
        Hook called after acknowledgement. If this is a simple agent,
        run the tool loop and emit the response, then bail.
        """
        if not await self._is_simple_agent(context.agent):
            return EngineHookResult.CALL_NEXT

        # Get system prompt from agent description
        system_prompt = context.agent.description or "You are a helpful assistant."

        # Load and append context variables to the system prompt
        context_variables = await self._load_context_variables(context)
        context_section = self._format_context_variables(context_variables)
        if context_section:
            system_prompt = system_prompt + "\n" + context_section

        # Gather tools associated with this agent
        tools = await gather_agent_tools(
            agent_id=context.agent.id,
            association_store=self._association_store,
            service_registry=self._service_registry,
        )

        # Run the tool-calling loop
        response_content = await self._run_tool_loop(context, system_prompt, tools)

        # Emit the response message
        await context.session_event_emitter.emit_message_event(
            trace_id=context.tracer.trace_id,
            data=MessageEventData(
                message=response_content,
                participant=Participant(
                    id=context.agent.id,
                    display_name=context.agent.name,
                ),
            ),
        )

        # Emit tool events that were collected
        for tool_event in context.state.tool_events:
            await context.session_event_emitter.emit_tool_event(
                trace_id=context.tracer.trace_id,
                data=cast(ToolEventData, tool_event.data),
            )

        # Emit ready status
        await context.session_event_emitter.emit_status_event(
            trace_id=context.tracer.trace_id,
            data=StatusEventData(status="ready", data=None),
        )

        # Bail out of normal engine processing
        return EngineHookResult.BAIL


def register_simple_agent_hook(
    hooks: EngineHooks,
    service_registry: ServiceRegistry,
    association_store: AgentToolAssociationStore,
    tag_store: TagStore,
    variable_store: ContextVariableStore,
    logger: Logger,
    max_iterations: Optional[int] = None,
) -> Optional[SimpleAgentHook]:
    """
    Register the simple agent hook with the engine hooks.

    Requires LITELLM_PROVIDER_MODEL_NAME environment variable to be set.
    This ensures simple agents use the same LLM configuration as the rest of Parlant
    when configured with nlp_service = "litellm".

    Args:
        hooks: The EngineHooks instance to register with.
        service_registry: The service registry for accessing tools.
        association_store: The store for agent-tool associations.
        tag_store: The tag store for looking up tag names.
        variable_store: The context variable store for loading variable values.
        logger: Logger instance for logging messages.
        max_iterations: Optional max tool-calling iterations (defaults to 10).

    Returns:
        The SimpleAgentHook instance, or None if not configured.
    """
    model = os.environ.get(LITELLM_MODEL_ENV)

    if not model:
        logger.warning(
            f"Simple agent hook not registered: {LITELLM_MODEL_ENV} environment variable not set. "
            "To use simple agents, configure Parlant with nlp_service = 'litellm' in parlant.toml "
            "and set the required LITELLM_PROVIDER_* environment variables."
        )
        return None

    base_url = os.environ.get(LITELLM_BASE_URL_ENV)

    hook = SimpleAgentHook(
        service_registry=service_registry,
        association_store=association_store,
        tag_store=tag_store,
        variable_store=variable_store,
        logger=logger,
        model=model,
        base_url=base_url,
        max_iterations=max_iterations,
    )
    hooks.on_acknowledged.append(hook.on_acknowledged)

    logger.info(f"Simple agent hook registered with model: {model}")

    return hook
