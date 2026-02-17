#!/usr/bin/env python
"""
nForce Parlant Server

Starts the Parlant engine configured for nForce:
  - LiteLLM NLP service (multi-provider: OpenAI, Anthropic, Google, etc.)
  - PostgreSQL + pgvector for persistence and vector search
  - Simple-agent hook enabled (agents tagged "simple-agent")
  - Optional auto-registration of the nForce plugin tool service

Required environment variables (can be set in .env):
    POSTGRES_CONNECTION_STRING  - PostgreSQL connection string
                                  (e.g., postgresql://postgres:password@localhost:5432/parlant)
    LITELLM_PROVIDER_MODEL_NAME - LiteLLM model identifier
                                  (e.g., openai/gpt-4o, anthropic/claude-sonnet-4-20250514)

Optional environment variables:
    PARLANT_HOST           - Server host (default: 0.0.0.0)
    PARLANT_PORT           - Server port (default: 8800)
    NFORCE_PLUGIN_URL      - nForce API plugin endpoint for tool service registration
                             (e.g., http://localhost:3333/parlant-plugin)
    NFORCE_PLUGIN_API_KEY  - Shared secret for plugin auth

Usage:
    uv run python nforce_server.py
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from lagom import Container


async def run() -> None:
    postgres_url = os.environ.get("POSTGRES_CONNECTION_STRING")
    if not postgres_url:
        print("Error: POSTGRES_CONNECTION_STRING environment variable is required")
        print("Example: postgresql://postgres:password@localhost:5432/parlant")
        sys.exit(1)

    if not os.environ.get("LITELLM_PROVIDER_MODEL_NAME"):
        print("Error: LITELLM_PROVIDER_MODEL_NAME environment variable is required")
        print("Example: openai/gpt-4o or anthropic/claude-sonnet-4-20250514")
        sys.exit(1)

    from parlant.adapters.db.postgres_db import PostgresDocumentDatabase
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase
    from parlant.core.agent_tool_associations import (
        AgentToolAssociationDocumentStore,
        AgentToolAssociationStore,
    )
    from parlant.core.agents import AgentDocumentStore, AgentStore
    from parlant.core.canned_responses import CannedResponseStore, CannedResponseVectorStore
    from parlant.core.capabilities import CapabilityStore, CapabilityVectorStore
    from parlant.core.common import IdGenerator
    from parlant.core.emission.event_publisher import EventEmitterFactory, EventPublisherFactory
    from parlant.core.services.tools.service_registry import (
        ServiceDocumentRegistry,
        ServiceRegistry,
    )
    from parlant.core.evaluations import EvaluationDocumentStore, EvaluationStore
    from parlant.core.glossary import GlossaryStore, GlossaryVectorStore
    from parlant.core.guideline_tool_associations import (
        GuidelineToolAssociationDocumentStore,
        GuidelineToolAssociationStore,
    )
    from parlant.core.guidelines import GuidelineDocumentStore, GuidelineStore
    from parlant.core.journeys import JourneyStore, JourneyVectorStore
    from parlant.core.loggers import Logger
    from parlant.core.nlp.embedding import EmbedderFactory, EmbeddingCache
    from parlant.core.nlp.service import NLPService
    from parlant.core.playbooks import PlaybookDocumentStore, PlaybookStore
    from parlant.core.relationships import RelationshipDocumentStore, RelationshipStore
    from parlant.core.static_playbooks import StaticPlaybookDocumentStore, StaticPlaybookStore
    from parlant.core.sessions import SessionStore
    from parlant.core.tags import TagDocumentStore, TagStore
    from parlant.core.test_suites import TestSuiteDocumentStore, TestSuiteStore
    from parlant.core.tracer import Tracer
    from parlant.sdk import NLPServices, Server

    host = os.environ.get("PARLANT_HOST", "0.0.0.0")
    port = int(os.environ.get("PARLANT_PORT", "8800"))

    exit_stack = AsyncExitStack()

    async def configure_stores(container: Container) -> Container:
        """Override transient stores with PostgreSQL-backed document stores
        and pgvector-backed vector stores for persistence."""
        logger = container[Logger]
        id_generator = container[IdGenerator]
        tracer = container[Tracer]

        async def make_pg_db(name: str) -> PostgresDocumentDatabase:
            """Create a PostgreSQL document database with a per-store table prefix.

            Each store needs its own database instance because the migration
            helper writes schema version metadata to a shared 'metadata'
            collection within the database. Separate table prefixes keep
            these from colliding."""
            return await exit_stack.enter_async_context(
                PostgresDocumentDatabase(
                    connection_string=postgres_url,
                    logger=logger,
                    table_prefix=f"{name}_",
                )
            )

        # Document stores — persist to PostgreSQL
        for interface, implementation, name in [
            (AgentStore, AgentDocumentStore, "agents"),
            (TagStore, TagDocumentStore, "tags"),
            (GuidelineStore, GuidelineDocumentStore, "guidelines"),
            (
                GuidelineToolAssociationStore,
                GuidelineToolAssociationDocumentStore,
                "guideline_tool_assocs",
            ),
            (
                AgentToolAssociationStore,
                AgentToolAssociationDocumentStore,
                "agent_tool_assocs",
            ),
            (RelationshipStore, RelationshipDocumentStore, "relationships"),
            (PlaybookStore, PlaybookDocumentStore, "playbooks"),
            (TestSuiteStore, TestSuiteDocumentStore, "test_suites"),
        ]:
            container[interface] = await exit_stack.enter_async_context(
                implementation(id_generator, await make_pg_db(name), allow_migration=True)
            )

        # Evaluation store (no id_generator)
        container[EvaluationStore] = await exit_stack.enter_async_context(
            EvaluationDocumentStore(await make_pg_db("evaluations"), allow_migration=True)
        )

        # Static playbook store (no id_generator — IDs come from nForce)
        container[StaticPlaybookStore] = await exit_stack.enter_async_context(
            StaticPlaybookDocumentStore(await make_pg_db("static_playbooks"), allow_migration=True)
        )

        # Vector stores — pgvector for similarity search, PostgreSQL for metadata
        embedder_factory = EmbedderFactory(container)

        async def get_embedder_type() -> type:
            return type(await container[NLPService].get_embedder())

        vector_db = await exit_stack.enter_async_context(
            PostgresVectorDatabase(
                connection_string=postgres_url,
                logger=logger,
                tracer=tracer,
                embedder_factory=embedder_factory,
                embedding_cache_provider=lambda: container[EmbeddingCache],
            )
        )

        for vector_store_interface, vector_store_type, name in [
            (GlossaryStore, GlossaryVectorStore, "glossary"),
            (CannedResponseStore, CannedResponseVectorStore, "canned_responses"),
            (JourneyStore, JourneyVectorStore, "journeys"),
            (CapabilityStore, CapabilityVectorStore, "capabilities"),
        ]:
            container[vector_store_interface] = await exit_stack.enter_async_context(
                vector_store_type(
                    id_generator=id_generator,
                    vector_db=vector_db,
                    document_db=await make_pg_db(name),
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
            )

        # Recreate EventEmitterFactory with the persistent AgentStore
        container[EventEmitterFactory] = EventPublisherFactory(
            agent_store=container[AgentStore],
            session_store=container[SessionStore],
        )

        # Recreate ServiceRegistry with persistent storage
        container[ServiceRegistry] = await exit_stack.enter_async_context(
            ServiceDocumentRegistry(
                database=await make_pg_db("services"),
                event_emitter_factory=container[EventEmitterFactory],
                logger=logger,
                tracer=tracer,
                nlp_services_provider=lambda: {"__nlp__": container[NLPService]},
                allow_migration=True,
            )
        )

        return container

    async def register_nforce_plugin(container: Container) -> None:
        """Register the nForce API as an SDK plugin tool service, if configured."""
        plugin_url = os.environ.get("NFORCE_PLUGIN_URL")
        if not plugin_url:
            return

        logger = container[Logger]
        registry = container[ServiceRegistry]

        try:
            await registry.update_tool_service(
                name="nforce-tools",
                kind="sdk",
                url=plugin_url,
            )
            logger.info(f"Registered nForce plugin tool service at {plugin_url}")
        except Exception as e:
            logger.warning(f"Failed to register nForce plugin tool service: {e}")

    async with exit_stack:
        async with Server(
            host=host,
            port=port,
            nlp_service=NLPServices.litellm,
            session_store=postgres_url,
            customer_store=postgres_url,
            variable_store=postgres_url,
            test_run_store=postgres_url,
            migrate=True,
            configure_container=configure_stores,
            initialize_container=register_nforce_plugin,
        ):
            pass


def main() -> None:
    load_dotenv()
    asyncio.run(run())


if __name__ == "__main__":
    main()
