#!/usr/bin/env python
"""
nForce Parlant Server

Stateless Parlant engine for nForce. All entity data (guidelines, terms,
tools, history, etc.) is sent inline with each /v2/process request by nForce.
No persistent stores are needed — Parlant boots with transient in-memory
defaults that satisfy the DI container but are never read at runtime.

Required environment variables (can be set in .env):
    LITELLM_PROVIDER_MODEL_NAME - LiteLLM model identifier
                                  (e.g., openai/gpt-4o, anthropic/claude-sonnet-4-20250514)

Optional environment variables:
    PARLANT_HOST           - Server host (default: 0.0.0.0)
    PARLANT_PORT           - Server port (default: 8800)

Usage:
    uv run python nforce_server.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv


async def run() -> None:
    if not os.environ.get("LITELLM_PROVIDER_MODEL_NAME"):
        print("Error: LITELLM_PROVIDER_MODEL_NAME environment variable is required")
        print("Example: openai/gpt-4o or anthropic/claude-sonnet-4-20250514")
        sys.exit(1)

    from parlant.core.agent_tool_associations import AgentToolAssociationStore
    from parlant.core.context_variables import ContextVariableStore
    from parlant.core.engines.alpha.hooks import EngineHooks
    from parlant.core.engines.alpha.simple_agent import register_simple_agent_hook
    from parlant.core.loggers import Logger
    from parlant.core.services.tools.service_registry import ServiceRegistry
    from parlant.core.tags import TagStore
    from parlant.sdk import Container, NLPServices, Server

    async def initialize_container(c: Container) -> None:
        register_simple_agent_hook(
            hooks=c[EngineHooks],
            service_registry=c[ServiceRegistry],
            association_store=c[AgentToolAssociationStore],
            tag_store=c[TagStore],
            variable_store=c[ContextVariableStore],
            logger=c[Logger],
        )

    host = os.environ.get("PARLANT_HOST", "0.0.0.0")
    port = int(os.environ.get("PARLANT_PORT", "8800"))

    async with Server(
        host=host,
        port=port,
        nlp_service=NLPServices.litellm,
        initialize_container=initialize_container,
    ):
        pass


def main() -> None:
    load_dotenv()
    asyncio.run(run())


if __name__ == "__main__":
    main()
