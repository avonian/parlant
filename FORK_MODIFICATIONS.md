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

## Merge Strategy

When pulling upstream updates:

```bash
git fetch upstream
git merge upstream/develop
```

Areas most likely to conflict:
1. **`entity_cq.py`** — Heavy modifications for playbook resolution
2. **`agents.py` (core + api)** — New fields (playbook_id, model_name, composition_mode)
3. **`app.py` / `server.py`** — Route mounting and adapter registration
4. **`sdk.py`** — Extended SDK for playbooks, simple agent, test suites
5. **`litellm_service.py`** — Overlapping fixes with upstream
6. **`tool_caller.py`** — Simple agent tool calling additions
