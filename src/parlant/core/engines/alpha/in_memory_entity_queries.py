"""
In-memory EntityQueries implementation for the stateless /v2/process endpoint.

Returns data from the request payload instead of querying persistent stores.
The engine works through the EntityQueries interface — this implementation
feeds it inline data so the engine runs without modification.
"""

from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

from parlant.core.agents import Agent, AgentId, CompositionMode
from parlant.core.capabilities import Capability
from parlant.core.canned_responses import CannedResponse, CannedResponseField, CannedResponseId
from parlant.core.common import Criticality, JSONSerializable
from parlant.core.context_variables import ContextVariable, ContextVariableId, ContextVariableValue
from parlant.core.customers import Customer, CustomerId
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolCallEvaluation, ToolInsights
from parlant.core.glossary import Term, TermId
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociation,
    GuidelineToolAssociationId,
)
from parlant.core.journeys import Journey, JourneyNodeId
from parlant.core.sessions import Event, Session, SessionId, SessionStore
from parlant.core.tools import ToolId, ToolService


class InMemoryEntityQueries:
    """
    EntityQueries-compatible object that returns data from the request payload.

    This is NOT a subclass of EntityQueries (which is a concrete class, not an ABC).
    Instead, it's a duck-type-compatible replacement that the engine can use
    through the same method signatures.
    """

    def __init__(
        self,
        agent: Agent,
        customer: Customer,
        session: Session,
        session_store: SessionStore,
        guidelines: Sequence[Guideline],
        terms: Sequence[Term],
        context_variables: Sequence[ContextVariable],
        context_variable_values: Mapping[ContextVariableId, ContextVariableValue],
        tool_associations: Sequence[GuidelineToolAssociation],
        tool_service: ToolService,
        canned_responses: Sequence[CannedResponse],
    ) -> None:
        self._agent = agent
        self._customer = customer
        self._session = session
        self._session_store = session_store
        self._guidelines = guidelines
        self._terms = terms
        self._context_variables = context_variables
        self._context_variable_values = context_variable_values
        self._tool_associations = tool_associations
        self._tool_service = tool_service
        self._canned_responses = canned_responses

        # Cache for guideline_and_journeys_it_depends_on (engine accesses this attribute)
        from cachetools import TTLCache
        from parlant.core.guidelines import GuidelineId
        from parlant.core.journeys import Journey

        self.guideline_and_journeys_it_depends_on = TTLCache[GuidelineId, list[Journey]](
            maxsize=1024, ttl=120
        )

    async def read_agent(self, agent_id: AgentId) -> Agent:
        return self._agent

    async def read_session(self, session_id: SessionId) -> Session:
        return self._session

    async def read_customer(self, customer_id: CustomerId) -> Customer:
        return self._customer

    async def resolve_effective_model_name(self, agent_id: AgentId) -> Optional[str]:
        return self._agent.model_name

    async def find_events(self, session_id: SessionId) -> Sequence[Event]:
        return await self._session_store.list_events(session_id)

    async def find_guidelines_for_context(
        self,
        agent_id: AgentId,
        journeys: Sequence[Journey],
        static_playbook_id: Optional[str] = None,
    ) -> Sequence[Guideline]:
        return self._guidelines

    async def find_context_variables_for_context(
        self,
        agent_id: AgentId,
        static_playbook_id: Optional[str] = None,
    ) -> Sequence[ContextVariable]:
        return self._context_variables

    async def read_context_variable_value(
        self,
        variable_id: ContextVariableId,
        key: str,
    ) -> Optional[ContextVariableValue]:
        return self._context_variable_values.get(variable_id)

    async def find_guideline_tool_associations(self) -> Sequence[GuidelineToolAssociation]:
        return self._tool_associations

    async def find_journey_node_tool_associations(
        self, node_id: JourneyNodeId
    ) -> Sequence[ToolId]:
        return []

    async def find_glossary_terms_for_context(
        self,
        agent_id: AgentId,
        query: str,
        static_playbook_id: Optional[str] = None,
    ) -> Sequence[Term]:
        return self._terms

    async def read_tool_service(self, service_name: str) -> ToolService:
        return self._tool_service

    async def finds_journeys_for_context(
        self,
        agent_id: AgentId,
        static_playbook_id: Optional[str] = None,
    ) -> Sequence[Journey]:
        return []

    async def sort_journeys_by_contextual_relevance(
        self,
        available_journeys: Sequence[Journey],
        query: str,
    ) -> Sequence[Journey]:
        return available_journeys

    async def find_capabilities_for_agent(
        self,
        agent_id: AgentId,
        query: str,
        max_count: int,
    ) -> Sequence[Capability]:
        return []

    async def find_canned_responses_for_context(
        self,
        agent: Agent,
        journeys: Sequence[Journey],
        guidelines: Sequence[Guideline],
        static_playbook_id: Optional[str] = None,
    ) -> Sequence[CannedResponse]:
        return self._canned_responses

    async def find_canned_responses_for_guidelines(
        self,
        guidelines: Sequence[Guideline],
    ) -> Sequence[CannedResponse]:
        return self._canned_responses

    async def find_journey_related_guidelines(
        self,
        journey: Journey,
    ) -> Sequence[GuidelineId]:
        return []

    async def find_guidelines_that_need_reevaluation(
        self,
        available_guidelines: dict[GuidelineId, Guideline],
        active_journeys: Sequence[Journey],
        tool_insights: ToolInsights,
    ) -> Sequence[Guideline]:
        # Stateless mode: no relationship store to check reevaluation relationships.
        # Return empty — guidelines won't be reevaluated based on tool calls.
        return []
