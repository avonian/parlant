# nForce Fork Modifications

This document describes all modifications made to Parlant in the nForce fork,
per Apache License 2.0 Section 4(b). The upstream project is
[emcie-co/parlant](https://github.com/emcie-co/parlant).

Fork maintained by: Avonian (github.com/avonian/parlant)
Base upstream version at time of consolidation: v3.2.0 (upstream/develop @ 3406868d)

---

## 1. PostgreSQL + pgvector Adapters

**Purpose:** Replace MongoDB/Qdrant with PostgreSQL for document storage and pgvector for vector search, allowing nForce to run on a single database.

**Files added:**
- `src/parlant/adapters/db/postgres_db.py` — Full DocumentDatabase implementation using asyncpg
- `src/parlant/adapters/vector_db/pgvector.py` — Full VectorDatabase implementation using pgvector
- `tests/adapters/db/test_postgres_db.py` — PostgreSQL adapter tests
- `tests/adapters/vector_db/test_pgvector.py` — pgvector adapter tests
- `docs/adapters/persistence/postgresql.md` — PostgreSQL setup documentation
- `docs/adapters/vector_db/pgvector.md` — pgvector setup documentation
- `examples/run_postgres_server.py` — Example server configuration with PostgreSQL

**Files modified:**
- `pyproject.toml` — Added asyncpg, pgvector dependencies
- `pytest.ini` — Added postgres test markers
- `src/parlant/bin/server.py` — PostgreSQL adapter registration
- `tests/conftest.py` — PostgreSQL test fixtures

**Key decisions:**
- JSONB storage for documents (mirrors MongoDB's document model)
- pgvector's cosine similarity for vector search
- Explicit numeric casting for JSONB comparisons to avoid text ordering bugs

---

## 2. Playbook as First-Class Entity

**Purpose:** Promote playbooks from a tag-based grouping concept to a first-class entity with its own store, API, and inheritance model. Playbooks can contain guidelines, terms, journeys, and other playbooks (parent-child hierarchy).

**Files added:**
- `src/parlant/core/playbooks.py` — PlaybookStore with inheritance chain resolution
- `src/parlant/core/app_modules/playbooks.py` — PlaybookModule for dependency injection
- `src/parlant/api/playbooks.py` — Full CRUD REST API for playbooks

**Files modified:**
- `src/parlant/core/entity_cq.py` — Playbook-aware guideline/term/journey resolution:
  - `_resolve_playbook_chain()` — walks parent-child hierarchy
  - `_get_playbook_tags()` — collects tags from playbook chain
  - `_collect_disabled_rules()` — supports disabling inherited rules at any level
  - Modified all `find_*_for_context()` methods to include playbook-scoped entities
- `src/parlant/core/agents.py` — Added `playbook_id` field to Agent
- `src/parlant/core/tags.py` — Added `Tag.for_playbook_id()` helper
- `src/parlant/core/application.py` — Register PlaybookModule
- `src/parlant/api/app.py` — Mount playbook routes
- `src/parlant/api/agents.py` — Accept playbook_id on agent creation/update

**Tests:**
- `tests/api/test_playbooks.py`
- `tests/api/test_agent_playbook.py`
- `tests/core/test_playbooks.py`
- `tests/core/stable/test_playbook_inheritance.py`
- `tests/core/stable/test_entity_cq.py`

---

## 3. Simple Agent Mode

**Purpose:** A lightweight agent mode that bypasses Parlant's full guideline matching engine, using direct LLM tool-calling instead. Used for prompt-mode agents in nForce where the full playbook engine isn't needed.

**Files added:**
- `src/parlant/core/engines/alpha/simple_agent.py` — SimpleAgentHook with direct LLM invocation

**Files modified:**
- `src/parlant/core/engines/alpha/engine.py` — Hook into engine for simple agent bypass
- `src/parlant/core/engines/alpha/engine_context.py` — Added simple_agent flag to context
- `src/parlant/core/engines/alpha/message_generator.py` — Simple agent message path
- `src/parlant/core/engines/alpha/tool_calling/tool_caller.py` — LiteLLM-based tool calling for simple agent
- `src/parlant/core/engines/alpha/tool_calling/default_tool_call_batcher.py` — Minor fix
- `src/parlant/core/engines/alpha/tool_calling/overlapping_tools_batch.py` — Minor fix
- `src/parlant/core/engines/alpha/tool_calling/single_tool_batch.py` — Minor fix
- `src/parlant/core/engines/alpha/tool_event_generator.py` — Context variable loading for simple agent
- `src/parlant/core/agents.py` — Added `composition_mode` and simple agent config to Agent model
- `src/parlant/sdk.py` — SDK support for simple agent configuration
- `src/parlant/bin/server.py` — Simple agent hook registration

---

## 4. Agent Tool Associations

**Purpose:** Allow fine-grained control over which tools are available to which agents, independent of guideline-tool associations.

**Files added:**
- `src/parlant/core/agent_tool_associations.py` — AgentToolAssociationStore
- `src/parlant/api/agent_tool_associations.py` — REST API for agent-tool CRUD

**Files modified:**
- `src/parlant/api/app.py` — Mount agent tool association routes
- `src/parlant/core/application.py` — Register store

**Tests:**
- `tests/api/test_agent_tool_associations.py`
- `tests/core/test_agent_tool_associations.py`

---

## 5. Test Suites Infrastructure

**Purpose:** Database-stored test suites with scenario-based testing, parallel execution, WebSocket streaming of results, and structured failure details. Note: upstream had a testing MVP that was removed in v3.2.0; our implementation is independent.

**Files added:**
- `src/parlant/core/test_suites.py` — TestSuiteStore, TestRunStore, execution engine
- `src/parlant/core/app_modules/test_suites.py` — TestSuiteModule
- `src/parlant/api/test_suites.py` — Full REST + WebSocket API
- `src/parlant/core/background_tasks.py` — Background task runner for test execution
- `src/parlant/testing/database_suite.py` — Database-backed test suite runner

**Files modified:**
- `src/parlant/testing/cli.py` — Extended CLI for database-stored suites
- `src/parlant/testing/response.py` — Enhanced response assertions
- `src/parlant/api/app.py` — Mount test suite routes
- `src/parlant/core/application.py` — Register test suite modules

**Tests:**
- `tests/api/test_test_suites.py`
- `tests/core/test_test_suites.py`

---

## 6. Agent Model Name Override

**Purpose:** Allow per-agent LLM model override, so different agents can use different models without changing the global NLP service configuration.

**Files modified:**
- `src/parlant/core/agents.py` — Added `model_name` field to Agent
- `src/parlant/api/agents.py` — Accept model_name on agent creation/update

---

## 7. Customer Search by Name

**Purpose:** Allow searching customers by name, not just by ID.

**Files modified:**
- `src/parlant/core/customers.py` — Added name-based query to CustomerStore
- `src/parlant/api/customers.py` — Added search parameter to list endpoint

---

## 8. LiteLLM Service Improvements

**Purpose:** Fix issues with the LiteLLM NLP adapter and add embedding support.

**Status:** Upstream independently adopted similar fixes in v3.2.0. May produce merge conflicts but intent is aligned.

**Files modified:**
- `src/parlant/adapters/nlp/litellm_service.py` — Embedding support, error handling fixes
- `src/parlant/adapters/nlp/hugging_face.py` — Minor compatibility fix

**Tests:**
- `tests/adapters/nlp/test_litellm_service.py`

---

## 9. CORS Error Response Fix

**Purpose:** Ensure CORS headers are included in error/exception responses, not just successful ones.

**Status:** Upstream has their own CORS middleware. May conflict on merge.

**Files modified:**
- `src/parlant/api/app.py` — CORS middleware configuration
- `src/parlant/api/authorization.py` — CORS headers in auth responses

---

## 10. Minor Modifications

**Miscellaneous small changes across the fork:**

- `src/parlant/core/version.py` — Fork version identifier
- `src/parlant/core/guidelines.py` — Added `dependencies` field for playbook rule disabling
- `src/parlant/core/services/tools/service_registry.py` — Version compatibility fix
- `src/parlant/api/glossary.py` — Minor fix
- `.gitignore` — Added `.idea` (JetBrains)
- `CHANGELOG.md` — Fork-specific changelog entries

---

## 11. Static Playbook Store & Versioning Infrastructure

**Purpose:** Support playbook versioning by providing a store for pre-resolved, self-contained playbook snapshots ("static playbooks"). When nForce releases a playbook version, it resolves the full playbook state (walking the inheritance chain, collecting all guidelines/terms/canned responses/context variables/relationships) and pushes the result to Parlant as a static playbook. At runtime, conversations bound to a released version bypass the normal tag-based resolution and use the static snapshot directly.

**Files added:**
- `src/parlant/core/static_playbooks.py` — StaticPlaybook domain models (StaticGuideline, StaticRelationship, StaticTerm, StaticCannedResponse, StaticContextVariable, StaticPlaybook) and StaticPlaybookDocumentStore implementation
- `src/parlant/api/static_playbooks.py` — CRUD REST API (PUT upsert, GET read, DELETE, GET list) at `/static-playbooks`
- `src/parlant/api/playbook_resolve.py` — `POST /playbooks/{playbookId}/resolve` endpoint that walks the playbook inheritance chain, collects all tag-scoped entities, filters disabled rules, and returns both resolved (flat) and source (original) data

**Files modified:**
- `src/parlant/core/entity_cq.py` — Added `StaticPlaybookStore` dependency and `static_playbook_id` bypass to:
  - `find_guidelines_for_context()` — returns static guidelines when static_playbook_id is set
  - `finds_journeys_for_context()` — returns empty list (journeys are pre-projected in static playbooks)
  - `find_glossary_terms_for_context()` — returns static terms
  - `find_canned_responses_for_context()` — returns static canned responses
  - `find_context_variables_for_context()` — returns static context variables
  - Added conversion helpers (`_static_*_to_domain`) to map static models back to core domain objects
- `src/parlant/core/engines/alpha/engine.py` — Pass `session.metadata["static_playbook_id"]` to all entity resolution calls
- `src/parlant/core/engines/alpha/canned_response_generator.py` — Pass `session.metadata["static_playbook_id"]` to canned response resolution calls
- `src/parlant/api/app.py` — Mount static playbook and resolve routes, add store dependencies
- `src/parlant/bin/server.py` — Register StaticPlaybookStore in container
- `nforce_server.py` — Register StaticPlaybookDocumentStore with PostgreSQL backend

**Key decisions:**
- Static playbook ID is passed via session metadata (`{"static_playbook_id": "..."}`) rather than a first-class Session field, avoiding session schema migration
- All `find_*_for_context` methods accept an optional `static_playbook_id` parameter with `None` default, maintaining backwards compatibility
- The resolve endpoint returns both resolved (flat, for runtime) and source (original with tags/hierarchy, for audit/revert) data
- Journey resolution is completely skipped for static playbooks since journey guidelines are already projected into the static guideline set

---

## 12. Stateless /v2/process Endpoint

**Purpose:** Replace the sync-based integration pattern (where nForce mirrors agents, sessions, guidelines, tools, etc. via 15+ CRUD endpoints) with a single stateless `POST /v2/process` endpoint. nForce sends all context inline in each request; Parlant processes it and streams back SSE events. No pre-synced state required.

**Files added:**
- `src/parlant/api/v2_process.py` — Stateless `POST /v2/process` endpoint. Accepts full context (agent, customer, history, guidelines, terms, canned responses, context variables, tools, engine state) in a single request. Builds `InMemoryEntityQueries` and `InMemorySessionStore` from the payload, runs the engine, and streams SSE events (status, tool, message, state, log).
- `src/parlant/core/engines/alpha/in_memory_entity_queries.py` — `InMemoryEntityQueries` implementation of the `EntityQueries` interface that returns data from the request payload instead of the database. The engine runs unmodified against this interface.
- `src/parlant/core/engines/alpha/in_memory_session_store.py` — Ephemeral session/event store for within-request event tracking. The engine writes and reads events during processing (e.g., staged tool events); this handles that lifecycle without touching any database.
- `src/parlant/core/engines/alpha/callback_tool_service.py` — `CallbackToolService` implementing `ToolService`. When the engine calls a tool, it POSTs to nForce's callback URL instead of using the SDK plugin protocol.
- `src/parlant/core/engines/alpha/sse_event_emitter.py` — `SSEEventEmitter` implementing `EventEmitter`. Collects engine events into an async queue and yields SSE-formatted strings for `StreamingResponse`.

**Files modified:**
- `src/parlant/api/app.py` — Mount `/v2/process` router. HTTP middlewares (`handle_cancellation`, `add_trace_id`) bypass `/v2/process` via `_STREAMING_PATHS` to avoid Starlette's `call_next()` response buffering which breaks SSE streaming.
- `src/parlant/bin/server.py` — Set `timeout_keep_alive=300` in uvicorn config. The default (5s) kills SSE streaming connections during engine processing.
- `src/parlant/adapters/loggers/websocket.py` — Added SSE sink mechanism (`register_sse_sink`/`unregister_sse_sink`). The `/v2/process` endpoint registers the SSE emitter's queue as a sink so engine trace logs from all sub-components (guideline matcher, message generator, etc.) are piped into the SSE stream as `log` events.

**Note:** All existing upstream CRUD routers (agents, sessions, customers, etc.) are kept intact. nForce no longer calls them for sync, but they remain available for standalone Parlant tooling and to minimize merge conflicts with upstream.

**SSE streaming details:**
- The event stream sends an initial `": stream-start\n\n"` SSE comment to confirm the connection is live
- Keepalive comments (`": keepalive\n\n"`) are sent every 2 seconds during idle periods
- History events with `kind=message` that lack a `participant` field get a synthetic one injected (defensive fix for engine compatibility)

**Key decisions:**
- The engine (`engines/alpha/`) is completely untouched — `InMemoryEntityQueries` implements the same `EntityQueries` interface the engine already uses
- Tool execution uses HTTP callbacks (Parlant POSTs to nForce) instead of the SDK plugin protocol
- `engine_state` (applied guideline IDs, journey paths) is round-tripped: sent in the request, returned in a `state` SSE event for the caller to persist
- CRUD routers for playbooks, guidelines, terms, journeys, etc. are kept — nForce's frontend still uses them through a proxy for playbook authoring
- Both `/v2/process` and remaining CRUD endpoints coexist during migration

---

## 13. Per-Request RelationalResolver with In-Memory Stores

**Purpose:** Eliminate `/v2/process`'s runtime dependency on Parlant's persistent `GuidelineStore` and `RelationshipStore`. Previously, `RelationalResolver` was a container singleton that read from PostgreSQL at request time to resolve guideline dependencies, priorities, and entailments. Now it is built per-request from inline data sent by nForce, making `/v2/process` fully stateless.

**Files modified:**
- `src/parlant/api/v2_process.py`:
  - Added `InlineRelationshipDTO` (source/target ID + kind) and `relationships` field to `ProcessRequestDTO`
  - Added `tags` field to `InlineGuidelineDTO` (for tag-based relationship resolution)
  - Added `_to_relationships()` converter (DTO → domain `Relationship` objects)
  - Added `_InMemoryRelationshipStore` — in-memory `RelationshipStore` implementation with BFS graph traversal for `indirect=True` queries (transitive dependency/priority/entailment chains). Builds per-kind directed graphs at construction; supports forward BFS (source_id) and reverse BFS (target_id)
  - Added `_InMemoryGuidelineStore` — in-memory `GuidelineStore` implementation with tag-based filtering. Indexes guidelines by tag at construction for efficient `list_guidelines(tags=[...])` lookups
  - `RelationalResolver` is now constructed per-request inside the `process()` handler using the in-memory stores, instead of being received as a `create_router()` parameter
  - `_to_guidelines()` now populates `Guideline.tags` from the request payload (previously hardcoded to `[]`)
  - Removed `relational_resolver` from `create_router()` signature
- `src/parlant/api/app.py` — Removed `relational_resolver` parameter from `v2_process.create_router()` call; removed `RelationalResolver` import
- `src/parlant/bin/server.py` — Removed `RelationalResolver` singleton from lagom container (`_define_singleton` call)

**Key decisions:**
- `_InMemoryRelationshipStore` replicates the BFS traversal logic from `RelationshipDocumentStore` (which uses networkx) using a simple queue-based BFS over adjacency lists — no external dependency needed
- Both in-memory stores implement only the methods actually called by `RelationalResolver` (`list_relationships`, `list_guidelines`); all other ABC methods raise `NotImplementedError`
- nForce sends source-level relationships (ID-based, including tag-targeted relationships) rather than pre-expanded guideline-to-guideline pairs, preserving `RelationalResolver`'s existing resolution algorithm
- The persistent stores are no longer registered in `nforce_server.py` (see §14); the transient in-memory defaults satisfy the DI chain

---

## 14. Stateless nforce_server.py — No PostgreSQL Required

**Purpose:** Eliminate Parlant's PostgreSQL dependency entirely when running under nForce. All entity data is sent inline with each `/v2/process` request, so persistent stores are unnecessary.

**Files modified:**
- `nforce_server.py` — Stripped from 237 lines to ~50. Removed `POSTGRES_CONNECTION_STRING` requirement, `configure_stores` callback (which overrode 16 transient stores with PostgreSQL/pgvector implementations), `AsyncExitStack`, and nForce plugin registration. Server now boots with `NLPServices.litellm` only — transient in-memory store defaults from `server.py` satisfy the DI container. The orphaned CRUD routers in `app.py` still mount against these empty transient stores (harmless, never called by nForce).

**Key decisions:**
- The transient stores are never read at runtime via `/v2/process` — all data comes inline from nForce
- CRUD routers remain mounted (removing them would require refactoring `app.py`'s `create_api_app`), but they operate on empty ephemeral stores
- Only `LITELLM_PROVIDER_MODEL_NAME` is required to boot

---

## 15. Honest Per-Request Model Reporting in Generation Metrics

**Purpose:** Report the actual model used per request in the `gen` histogram and `gen.request_*` trace events, instead of always showing the env-var default (`LITELLM_PROVIDER_MODEL_NAME`). Without this, `model.name` in traces shows the fallback model even when an agent's `model_name` override is in effect — making it impossible to tell from logs which model actually served a turn.

**Files modified:**
- `src/parlant/core/nlp/generation.py` — In `BaseSchematicGenerator.generate()`, resolve `effective_model_name = hints.get("model_name") or self.model_name` once at entry and use it for the histogram label, `gen.request_failed` event, and `gen.request_completed` event.

**Key decisions:**
- Falls through to `self.model_name` when no hint is set, preserving behavior for adapters that don't pass an override
- Only the metric/trace attributes change; no engine behavior is affected

---

## 16. Inline Journeys in the Stateless Engine

**Purpose:** Make journeys work through `/v2/process`. The stateless rewrite (§12–14) built in-memory stores for guidelines and relationships but left journeys stubbed: the DTO had no `journeys` field and the in-memory entity queries returned empty lists, so journey activation, node tools, and branching never ran in stateless mode. nForce now sends journeys inline (nodes, edges, conditions, per-node tools) and the engine consumes them per-request.

**Files added:**
- `src/parlant/api/inline_guideline_store.py` — `InlineAwareGuidelineStore`, a ContextVar-backed wrapper around the transient `GuidelineStore`. The journey node-selection batch reads condition guidelines via the container's `GuidelineStore` singleton (held fixed at strategy construction), which is empty in stateless mode; this wrapper serves the request's inline guidelines instead. Set per request via `set_inline_guidelines()` in `v2_process.py`; transparent when unset.

**Files modified:**
- `src/parlant/api/v2_process.py` — Added `InlineJourneyDTO` / node / edge DTOs and a `journeys` request field; `_InMemoryJourneyStore`; per-request `JourneyGuidelineProjection`; registers inline guidelines with the `InlineAwareGuidelineStore` for the duration of the request
- `src/parlant/core/engines/alpha/in_memory_entity_queries.py` — Un-stubbed `finds_journeys_for_context()`, `find_journey_node_tool_associations()`, and `find_journey_related_guidelines()`; `find_guidelines_for_context()` now runs journey guideline projection; added `_attach_reachable_follow_ups()`, which derives `metadata.journey_node.reachable_follow_ups` from the projected follow-ups so branch selection works without the offline reachability-indexing service (covers direct forks; multi-hop reachability not expanded)
- `src/parlant/sdk.py` — `override_stores_with_transient_versions` wraps the transient `GuidelineDocumentStore` in `InlineAwareGuidelineStore`. It must live here: Lagom forbids redefining a binding (`DuplicateDefinition`), and the SDK defines stores before `bin/server.py` loads the app, so overrides in `nforce_server.py` / `bin/server.py` do not work

**Key decisions:**
- Follows the §13 precedent: per-request in-memory stores, engine untouched
- No-ops when a request carries no inline journeys (transition-safe)
- Verified end-to-end in stateless mode: trigger-guideline activation → journey node selection → node tool execution → two-way branching on condition edges

---

## 17. LiteLLM Runtime Quirks & Model Compatibility

**Purpose:** Keep the LiteLLM path working across provider/model changes without code edits per model.

**Files modified:**
- `src/parlant/adapters/nlp/litellm_service.py`:
  - **Temperature quirk learning** — some models reject the `temperature` param outright with a 400. On such an error the adapter drops `temperature`, retries, and persists the model name to `.litellm_temperature_quirks.json` (path override via `LITELLM_TEMPERATURE_QUIRKS_PATH`) so later calls skip it up front. The quirks file is gitignored.
  - **`LITELLM_DISABLE_EMBEDDER`** — stateless deployments never invoke the embedder, but `get_embedder()`'s Jina fallback eagerly loaded torch + jina-embeddings-v2 at boot (~4.6 GB resident). With the flag set, `NullEmbedder` is returned and the `hugging_face` import is lazy so torch never loads. Default behavior without the flag is unchanged.
  - **GPT-5.6 reasoning opt-out** — GPT-5.6 models 400 on `/v1/chat/completions` when function tools are combined with reasoning ("... use /v1/responses or set reasoning_effort to 'none'"). `do_generate()` sends `reasoning_effort="none"` for `gpt-5.6*` models (pre-5.6 models defaulted to none).
- `src/parlant/core/engines/alpha/simple_agent.py` — Same `reasoning_effort="none"` opt-out in `_run_tool_loop()`, which calls `litellm.acompletion()` directly with the agent's function tools (see §3) and hit the identical 400.

---

## 18. /v2/process Latency Instrumentation

**Purpose:** Diagnose where time goes inside a `/v2/process` request.

**Files modified:**
- `src/parlant/api/v2_process.py` — Captures milestones (`dto_to_domain`, `entity_queries_built`, `engine_ready`, `first_engine_event`, `first_message_event`, `engine_complete`) as ms-since-request and logs them alongside the engine-completed line. Gated on `LATENCY_TRACE=true` so prod stays quiet.

---

## Merge Strategy

When pulling upstream updates:

```bash
git fetch upstream
git merge upstream/develop
```

Areas most likely to conflict:
1. **`entity_cq.py`** — Heavy modifications for playbook resolution
2. **`agents.py` (core)** — New fields (playbook_id, model_name, composition_mode)
3. **`app.py` / `server.py`** — Route mounting, middleware bypass for SSE, and uvicorn config changes
4. **`sdk.py`** — Extended SDK for playbooks, simple agent, test suites
5. **`litellm_service.py`** — Overlapping fixes with upstream, plus fork-only quirk handling (§8, §17)
6. **`tool_caller.py`** — Simple agent tool calling additions
7. **`core/nlp/generation.py`** — `BaseSchematicGenerator.generate()` per-request model reporting (§15)
8. **`v2_process.py` / `in_memory_entity_queries.py`** — Fork-only files, but they track engine-internal interfaces (`EntityQueries`, journey projection, node-selection metadata) that upstream may reshape (§12–13, §16, §18)
9. **`sdk.py`** — Now also wraps the transient `GuidelineDocumentStore` in `InlineAwareGuidelineStore` inside `override_stores_with_transient_versions` (§16) — a store-definition change upstream touches regularly

Note: §12–14 make `/v2/process` fully stateless and eliminate PostgreSQL from `nforce_server.py`. Upstream CRUD routers are kept intact but run against empty transient stores. Conflicts are unlikely unless upstream restructures the middleware stack, uvicorn config, or `RelationalResolver` wiring.
