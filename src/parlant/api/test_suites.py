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

"""Test Suites API endpoints.

Provides REST API for managing test suites, scenarios, and test runs.
Includes WebSocket endpoint for real-time test progress streaming.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Sequence, TypeAlias

from fastapi import APIRouter, HTTPException, Path, Query, Request, WebSocket, status
from pydantic import Field
from starlette.websockets import WebSocketDisconnect

from parlant.api import common
from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import ExampleJson, apigen_config
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel, ItemNotFoundError
from parlant.core.customers import CustomerId
from parlant.core.test_suites import (
    TestRun,
    TestRunId,
    TestScenario,
    TestScenarioId,
    TestScenarioResult,
    TestScenarioUpdateParams,
    TestStep,
    TestStepResult,
    TestSuite,
    TestSuiteId,
    TestSuiteUpdateParams,
)

API_GROUP = "test-suites"


# Field type aliases

TestSuiteIdPath: TypeAlias = Annotated[
    str,
    Path(
        description="Unique identifier of the test suite",
        examples=["ts_abc123"],
    ),
]

TestScenarioIdPath: TypeAlias = Annotated[
    str,
    Path(
        description="Unique identifier of the test scenario",
        examples=["tsc_xyz789"],
    ),
]

TestRunIdPath: TypeAlias = Annotated[
    str,
    Path(
        description="Unique identifier of the test run",
        examples=["tr_def456"],
    ),
]

AgentIdQuery: TypeAlias = Annotated[
    str | None,
    Query(
        description="Filter test suites by agent ID",
        examples=["agent_123"],
    ),
]

SuiteIdQuery: TypeAlias = Annotated[
    str | None,
    Query(
        description="Filter test runs by suite ID",
        examples=["ts_abc123"],
    ),
]

LimitQuery: TypeAlias = Annotated[
    int,
    Query(
        description="Maximum number of items to return",
        ge=1,
        le=100,
    ),
]


# DTOs

test_step_example: ExampleJson = {
    "role": "customer",
    "content": "Hello, I need help with my order",
    "should": None,
    "should_weight": 1.0,
}

tool_step_example: ExampleJson = {
    "role": "tool",
    "content": "appointments:check_availability",
    "tool_arguments": {"doctor": "Dr. Smith"},
    "tool_response": {"available_slots": ["9:00 AM", "2:00 PM"]},
}


class TestStepDTO(DefaultBaseModel):
    """A single step in a test scenario.

    Steps can be:
    - customer: Send a message to the agent
    - agent: Assert the agent's response
    - tool: Expect a tool call and return a mock response
    """

    role: Literal["customer", "agent", "tool"] = Field(description="Role that performs this step")
    content: str = Field(
        description="For customer/agent: message content. For tool: tool_id (e.g., 'service:tool_name')"
    )
    should: str | None = Field(
        default=None,
        description="Assertion condition for agent steps (e.g., 'greet the customer')",
    )
    should_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Weight for scoring when multiple conditions",
    )
    tool_arguments: Dict[str, Any] | None = Field(
        default=None,
        description="For tool steps: expected arguments to assert (optional)",
    )
    tool_response: Dict[str, Any] | None = Field(
        default=None,
        description="For tool steps: mock response to return",
    )


test_suite_example: ExampleJson = {
    "id": "ts_abc123",
    "agent_id": "agent_123",
    "name": "Customer Support Tests",
    "description": "Tests for customer support agent behavior",
    "creation_utc": "2024-03-24T12:00:00Z",
}


class TestSuiteDTO(DefaultBaseModel, json_schema_extra={"example": test_suite_example}):
    """A collection of test scenarios for an agent."""

    id: str = Field(description="Unique identifier")
    agent_id: str = Field(description="The agent being tested")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Description of what this suite tests")
    creation_utc: datetime = Field(description="When this suite was created")


test_scenario_example: ExampleJson = {
    "id": "tsc_xyz789",
    "suite_id": "ts_abc123",
    "name": "test_greeting",
    "description": "Test that the agent greets customers properly",
    "steps": [
        {"role": "customer", "content": "Hello!", "should": None, "should_weight": 1.0},
        {
            "role": "agent",
            "content": "Hello! How can I help you today?",
            "should": "greet the customer warmly",
            "should_weight": 1.0,
        },
    ],
    "customer_id": None,
    "repetitions": 1,
    "creation_utc": "2024-03-24T12:00:00Z",
}


class TestScenarioDTO(DefaultBaseModel, json_schema_extra={"example": test_scenario_example}):
    """A single test scenario definition."""

    id: str = Field(description="Unique identifier")
    suite_id: str = Field(description="The suite this scenario belongs to")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Description of what this scenario tests")
    steps: Sequence[TestStepDTO] = Field(description="Conversation steps")
    customer_id: str | None = Field(default=None, description="Customer to use (None = guest)")
    repetitions: int = Field(default=1, ge=1, description="Number of times to run")
    creation_utc: datetime = Field(description="When this scenario was created")


tool_call_record_example: ExampleJson = {
    "tool_id": "local:get_available_slots",
    "tool_name": "get_available_slots",
    "arguments": {"date": "2024-01-15"},
    "result": {"slots": ["09:00", "10:00", "14:00"]},
}


class ToolCallRecordDTO(DefaultBaseModel, json_schema_extra={"example": tool_call_record_example}):
    """Record of a tool call made during a test step."""

    tool_id: str = Field(description="Full tool identifier (service:tool_name)")
    tool_name: str = Field(description="Tool name")
    arguments: Dict[str, Any] = Field(description="Arguments passed to the tool")
    result: Any = Field(description="Result returned by the tool")


test_step_result_example: ExampleJson = {
    "step_index": 0,
    "role": "customer",
    "content": "Hello!",
    "actual_response": "Hello! How can I help you today?",
    "tool_calls": [tool_call_record_example],
    "assertion": "greet the customer warmly",
    "assertion_passed": True,
    "assertion_reasoning": "The response warmly greets the customer",
    "assertion_score": 100.0,
    "trace_id": "abc123-def456-ghi789",
}


class TestStepResultDTO(DefaultBaseModel, json_schema_extra={"example": test_step_result_example}):
    """Result of executing a single test step."""

    step_index: int = Field(description="Index of the step in the scenario")
    role: str = Field(description="Role that performed this step")
    content: str = Field(description="Original step content")
    actual_response: str | None = Field(default=None, description="Actual agent response")
    tool_calls: List[ToolCallRecordDTO] | None = Field(
        default=None, description="Tool calls made during this step"
    )
    assertion: str | None = Field(default=None, description="Assertion that was tested")
    assertion_passed: bool | None = Field(default=None, description="Whether assertion passed")
    assertion_reasoning: str | None = Field(default=None, description="NLP reasoning for pass/fail")
    assertion_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Assertion score (0-100)"
    )
    trace_id: str | None = Field(default=None, description="Trace ID for debugging agent responses")


test_scenario_result_example: ExampleJson = {
    "scenario_id": "tsc_xyz789",
    "scenario_name": "test_greeting",
    "status": "passed",
    "duration_ms": 1234.5,
    "step_results": [test_step_result_example],
    "error": None,
    "repetition": 1,
}


class TestScenarioResultDTO(
    DefaultBaseModel, json_schema_extra={"example": test_scenario_result_example}
):
    """Result of running a single scenario."""

    scenario_id: str = Field(description="ID of the scenario")
    scenario_name: str = Field(description="Name of the scenario")
    status: Literal["passed", "failed", "error", "skipped"] = Field(description="Overall status")
    duration_ms: float = Field(ge=0.0, description="Time taken in milliseconds")
    step_results: Sequence[TestStepResultDTO] = Field(description="Results per step")
    error: str | None = Field(default=None, description="Error message if failed")
    repetition: int = Field(default=1, ge=1, description="Which repetition this was")


test_run_example: ExampleJson = {
    "id": "tr_def456",
    "suite_id": "ts_abc123",
    "agent_id": "agent_123",
    "status": "completed",
    "creation_utc": "2024-03-24T12:00:00Z",
    "completion_utc": "2024-03-24T12:01:00Z",
    "total": 5,
    "passed": 4,
    "failed": 1,
    "errors": 0,
    "duration_ms": 60000.0,
    "scenario_results": [test_scenario_result_example],
}


class TestRunDTO(DefaultBaseModel, json_schema_extra={"example": test_run_example}):
    """Record of a test suite execution."""

    id: str = Field(description="Unique identifier")
    suite_id: str = Field(description="The suite that was run")
    agent_id: str = Field(description="The agent that was tested")
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = Field(
        description="Current status"
    )
    creation_utc: datetime = Field(description="When the run started")
    completion_utc: datetime | None = Field(default=None, description="When the run finished")
    total: int = Field(ge=0, description="Total number of scenarios")
    passed: int = Field(ge=0, description="Number passed")
    failed: int = Field(ge=0, description="Number failed")
    errors: int = Field(ge=0, description="Number errored")
    duration_ms: float = Field(ge=0.0, description="Total time in milliseconds")
    scenario_results: Sequence[TestScenarioResultDTO] = Field(description="Results per scenario")


class DeleteRunsResponseDTO(DefaultBaseModel):
    """Response for bulk delete of test runs."""

    deleted_count: int = Field(ge=0, description="Number of runs deleted")


# Creation/update DTOs

suite_creation_example: ExampleJson = {
    "agent_id": "agent_123",
    "name": "Customer Support Tests",
    "description": "Tests for customer support agent behavior",
}


class TestSuiteCreationParamsDTO(
    DefaultBaseModel, json_schema_extra={"example": suite_creation_example}
):
    """Parameters for creating a test suite."""

    agent_id: str = Field(description="The agent to test")
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="Description")


class TestSuiteUpdateParamsDTO(DefaultBaseModel):
    """Parameters for updating a test suite."""

    name: str | None = Field(default=None, description="New name")
    description: str | None = Field(default=None, description="New description")


scenario_creation_example: ExampleJson = {
    "name": "test_greeting",
    "description": "Test that the agent greets customers properly",
    "steps": [
        {"role": "customer", "content": "Hello!", "should": None, "should_weight": 1.0},
        {
            "role": "agent",
            "content": "Hello! How can I help you today?",
            "should": "greet the customer warmly",
            "should_weight": 1.0,
        },
    ],
    "customer_id": None,
    "repetitions": 1,
}


class TestScenarioCreationParamsDTO(
    DefaultBaseModel, json_schema_extra={"example": scenario_creation_example}
):
    """Parameters for creating a test scenario."""

    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="Description")
    steps: Sequence[TestStepDTO] = Field(description="Conversation steps")
    customer_id: str | None = Field(default=None, description="Customer to use (None = guest)")
    repetitions: int = Field(default=1, ge=1, description="Number of times to run")


class TestScenarioUpdateParamsDTO(DefaultBaseModel):
    """Parameters for updating a test scenario."""

    name: str | None = Field(default=None, description="New name")
    description: str | None = Field(default=None, description="New description")
    steps: Sequence[TestStepDTO] | None = Field(default=None, description="New steps")
    customer_id: str | None = Field(default=None, description="New customer")
    repetitions: int | None = Field(default=None, ge=1, description="New repetitions")


# Conversion functions


def _test_step_to_dto(step: TestStep) -> TestStepDTO:
    return TestStepDTO(
        role=step.role,
        content=step.content,
        should=step.should,
        should_weight=step.should_weight,
        tool_arguments=dict(step.tool_arguments) if step.tool_arguments else None,
        tool_response=dict(step.tool_response) if step.tool_response else None,
    )


def _test_step_from_dto(dto: TestStepDTO) -> TestStep:
    return TestStep(
        role=dto.role,
        content=dto.content,
        should=dto.should,
        should_weight=dto.should_weight,
        tool_arguments=dto.tool_arguments,
        tool_response=dto.tool_response,
    )


def _test_suite_to_dto(suite: TestSuite) -> TestSuiteDTO:
    return TestSuiteDTO(
        id=suite.id,
        agent_id=suite.agent_id,
        name=suite.name,
        description=suite.description,
        creation_utc=suite.creation_utc,
    )


def _test_scenario_to_dto(scenario: TestScenario) -> TestScenarioDTO:
    return TestScenarioDTO(
        id=scenario.id,
        suite_id=scenario.suite_id,
        name=scenario.name,
        description=scenario.description,
        steps=[_test_step_to_dto(s) for s in scenario.steps],
        customer_id=scenario.customer_id,
        repetitions=scenario.repetitions,
        creation_utc=scenario.creation_utc,
    )


def _test_step_result_to_dto(result: TestStepResult) -> TestStepResultDTO:
    tool_calls_dto: List[ToolCallRecordDTO] | None = None
    if result.tool_calls:
        tool_calls_dto = [
            ToolCallRecordDTO(
                tool_id=tc.tool_id,
                tool_name=tc.tool_name,
                arguments=dict(tc.arguments),
                result=tc.result,
            )
            for tc in result.tool_calls
        ]

    return TestStepResultDTO(
        step_index=result.step_index,
        role=result.role,
        content=result.content,
        actual_response=result.actual_response,
        tool_calls=tool_calls_dto,
        assertion=result.assertion,
        assertion_passed=result.assertion_passed,
        assertion_reasoning=result.assertion_reasoning,
        assertion_score=result.assertion_score,
        trace_id=result.trace_id,
    )


def _test_scenario_result_to_dto(result: TestScenarioResult) -> TestScenarioResultDTO:
    return TestScenarioResultDTO(
        scenario_id=result.scenario_id,
        scenario_name=result.scenario_name,
        status=result.status.value,
        duration_ms=result.duration_ms,
        step_results=[_test_step_result_to_dto(sr) for sr in result.step_results],
        error=result.error,
        repetition=result.repetition,
    )


def _test_run_to_dto(run: TestRun) -> TestRunDTO:
    return TestRunDTO(
        id=run.id,
        suite_id=run.suite_id,
        agent_id=run.agent_id,
        status=run.status.value,
        creation_utc=run.creation_utc,
        completion_utc=run.completion_utc,
        total=run.total,
        passed=run.passed,
        failed=run.failed,
        errors=run.errors,
        duration_ms=run.duration_ms,
        scenario_results=[_test_scenario_result_to_dto(sr) for sr in run.scenario_results],
    )


class WebSocketTestEventListener:
    """Event listener that streams test events over WebSocket.

    Satisfies the TestEventListenerProtocol from app_modules.test_suites.

    Events are sent as JSON messages with the following structure:
    {
        "type": "event_type",
        "data": { ... event data ... }
    }
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._closed = False

    async def _send(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send an event over the WebSocket."""
        if self._closed:
            return

        try:
            await self._websocket.send_json(
                {
                    "type": event_type,
                    "data": data,
                }
            )
        except Exception:
            self._closed = True

    async def on_suite_start(self, suite_name: str, total_tests: int) -> None:
        await self._send(
            "suite_start",
            {
                "suite_name": suite_name,
                "total_tests": total_tests,
            },
        )

    async def on_test_start(self, test_name: str) -> None:
        await self._send(
            "test_start",
            {
                "test_name": test_name,
            },
        )

    async def on_message_sent(self, test_name: str, role: str, content: str) -> None:
        await self._send(
            "message_sent",
            {
                "test_name": test_name,
                "role": role,
                "content": content,
            },
        )

    async def on_waiting_for_agent(self, test_name: str) -> None:
        await self._send(
            "waiting_for_agent",
            {
                "test_name": test_name,
            },
        )

    async def on_message_received(
        self,
        test_name: str,
        role: str,
        content: str,
        tool_calls: Any = None,
        trace_id: str | None = None,
    ) -> None:
        await self._send(
            "message_received",
            {
                "test_name": test_name,
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
                "trace_id": trace_id,
            },
        )

    async def on_evaluating(self, test_name: str, conditions: List[str]) -> None:
        await self._send(
            "evaluating",
            {
                "test_name": test_name,
                "conditions": conditions,
            },
        )

    async def on_condition_result(self, test_name: str, condition: str, passed: bool) -> None:
        await self._send(
            "condition_result",
            {
                "test_name": test_name,
                "condition": condition,
                "passed": passed,
            },
        )

    async def on_assertion_score(self, test_name: str, score: float) -> None:
        await self._send(
            "assertion_score",
            {
                "test_name": test_name,
                "score": score,
            },
        )

    async def on_test_passed(self, test_name: str, duration_ms: float, details: Any = None) -> None:
        await self._send(
            "test_passed",
            {
                "test_name": test_name,
                "duration_ms": duration_ms,
                "details": details,
            },
        )

    async def on_test_failed(
        self, test_name: str, duration_ms: float, error: str, details: Any
    ) -> None:
        await self._send(
            "test_failed",
            {
                "test_name": test_name,
                "duration_ms": duration_ms,
                "error": error,
                "details": details,
            },
        )

    async def on_suite_end(self, report: Any) -> None:
        # report may be None from our module or a TestReport from the testing framework
        if report is None:
            return
        await self._send(
            "suite_end",
            {
                "total": getattr(report, "total", 0),
                "passed": getattr(report, "passed", 0),
                "failed": getattr(report, "failed", 0),
                "skipped": getattr(report, "skipped", 0),
                "errors": getattr(report, "errors", 0),
                "duration_ms": getattr(report, "duration_ms", 0),
            },
        )


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    router = APIRouter()

    # Test Suite endpoints

    @router.get(
        "",
        operation_id="list_test_suites",
        response_model=Sequence[TestSuiteDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of test suites",
                "content": common.example_json_content([test_suite_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_test_suites(
        request: Request,
        agent_id: AgentIdQuery = None,
    ) -> Sequence[TestSuiteDTO]:
        """List all test suites, optionally filtered by agent."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_TEST_SUITE,
        )

        from parlant.core.agents import AgentId

        suites = await app.test_suites.list_suites(agent_id=AgentId(agent_id) if agent_id else None)
        return [_test_suite_to_dto(s) for s in suites]

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_test_suite",
        response_model=TestSuiteDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Test suite created",
                "content": common.example_json_content(test_suite_example),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_test_suite(
        request: Request,
        params: TestSuiteCreationParamsDTO,
    ) -> TestSuiteDTO:
        """Create a new test suite."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.CREATE_TEST_SUITE,
        )

        from parlant.core.agents import AgentId

        try:
            suite = await app.test_suites.create_suite(
                agent_id=AgentId(params.agent_id),
                name=params.name,
                description=params.description,
            )
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {params.agent_id}",
            )

        return _test_suite_to_dto(suite)

    # Test Run endpoints (must be before /{suite_id} to avoid route conflicts)

    @router.get(
        "/runs",
        operation_id="list_test_runs",
        response_model=Sequence[TestRunDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of test runs",
                "content": common.example_json_content([test_run_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list_runs"),
    )
    async def list_test_runs(
        request: Request,
        suite_id: SuiteIdQuery = None,
        limit: LimitQuery = 50,
    ) -> Sequence[TestRunDTO]:
        """List test runs, optionally filtered by suite."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_TEST_RUN,
        )

        runs = await app.test_suites.list_runs(
            suite_id=TestSuiteId(suite_id) if suite_id else None,
            limit=limit,
        )
        return [_test_run_to_dto(r) for r in runs]

    @router.get(
        "/runs/{run_id}",
        operation_id="read_test_run",
        response_model=TestRunDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Test run details",
                "content": common.example_json_content(test_run_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Test run not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve_run"),
    )
    async def read_test_run(
        request: Request,
        run_id: TestRunIdPath,
    ) -> TestRunDTO:
        """Get a test run by ID."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_TEST_RUN,
        )

        try:
            run = await app.test_suites.read_run(TestRunId(run_id))
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test run not found: {run_id}",
            )

        return _test_run_to_dto(run)

    @router.delete(
        "/runs/{run_id}",
        operation_id="delete_test_run",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Test run deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "Test run not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete_run"),
    )
    async def delete_test_run(
        request: Request,
        run_id: TestRunIdPath,
    ) -> None:
        """Delete a test run."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.DELETE_TEST_RUN,
        )

        try:
            await app.test_suites.delete_run(TestRunId(run_id))
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test run not found: {run_id}",
            )

    @router.delete(
        "/runs",
        operation_id="delete_test_runs",
        response_model=DeleteRunsResponseDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Number of runs deleted",
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete_runs"),
    )
    async def delete_test_runs(
        request: Request,
        suite_id: SuiteIdQuery = None,
    ) -> DeleteRunsResponseDTO:
        """Delete test runs, optionally filtered by suite."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.DELETE_TEST_RUNS,
        )

        deleted_count = await app.test_suites.delete_runs(
            suite_id=TestSuiteId(suite_id) if suite_id else None,
        )
        return DeleteRunsResponseDTO(deleted_count=deleted_count)

    # Test Suite endpoints (with path parameters)

    @router.get(
        "/{suite_id}",
        operation_id="read_test_suite",
        response_model=TestSuiteDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Test suite details",
                "content": common.example_json_content(test_suite_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Test suite not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_test_suite(
        request: Request,
        suite_id: TestSuiteIdPath,
    ) -> TestSuiteDTO:
        """Get a test suite by ID."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_TEST_SUITE,
        )

        try:
            suite = await app.test_suites.read_suite(TestSuiteId(suite_id))
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test suite not found: {suite_id}",
            )

        return _test_suite_to_dto(suite)

    @router.patch(
        "/{suite_id}",
        operation_id="update_test_suite",
        response_model=TestSuiteDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Test suite updated",
                "content": common.example_json_content(test_suite_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Test suite not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_test_suite(
        request: Request,
        suite_id: TestSuiteIdPath,
        params: TestSuiteUpdateParamsDTO,
    ) -> TestSuiteDTO:
        """Update a test suite."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_TEST_SUITE,
        )

        update_params: TestSuiteUpdateParams = {}
        if params.name is not None:
            update_params["name"] = params.name
        if params.description is not None:
            update_params["description"] = params.description

        try:
            suite = await app.test_suites.update_suite(TestSuiteId(suite_id), update_params)
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test suite not found: {suite_id}",
            )

        return _test_suite_to_dto(suite)

    @router.delete(
        "/{suite_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_test_suite",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Test suite deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "Test suite not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_test_suite(
        request: Request,
        suite_id: TestSuiteIdPath,
    ) -> None:
        """Delete a test suite and all its scenarios."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.DELETE_TEST_SUITE,
        )

        try:
            await app.test_suites.delete_suite(TestSuiteId(suite_id))
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test suite not found: {suite_id}",
            )

    # Test Scenario endpoints

    @router.get(
        "/{suite_id}/scenarios",
        operation_id="list_test_scenarios",
        response_model=Sequence[TestScenarioDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of test scenarios",
                "content": common.example_json_content([test_scenario_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list_scenarios"),
    )
    async def list_test_scenarios(
        request: Request,
        suite_id: TestSuiteIdPath,
    ) -> Sequence[TestScenarioDTO]:
        """List all scenarios in a test suite."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_TEST_SCENARIO,
        )

        scenarios = await app.test_suites.list_scenarios(TestSuiteId(suite_id))
        return [_test_scenario_to_dto(s) for s in scenarios]

    @router.post(
        "/{suite_id}/scenarios",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_test_scenario",
        response_model=TestScenarioDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Test scenario created",
                "content": common.example_json_content(test_scenario_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Test suite not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="create_scenario"),
    )
    async def create_test_scenario(
        request: Request,
        suite_id: TestSuiteIdPath,
        params: TestScenarioCreationParamsDTO,
    ) -> TestScenarioDTO:
        """Create a new test scenario in a suite."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.CREATE_TEST_SCENARIO,
        )

        try:
            scenario = await app.test_suites.create_scenario(
                suite_id=TestSuiteId(suite_id),
                name=params.name,
                description=params.description,
                steps=[_test_step_from_dto(s) for s in params.steps],
                customer_id=CustomerId(params.customer_id) if params.customer_id else None,
                repetitions=params.repetitions,
            )
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test suite not found: {suite_id}",
            )

        return _test_scenario_to_dto(scenario)

    # Scenario endpoints (independent paths)

    @router.get(
        "/scenarios/{scenario_id}",
        operation_id="read_test_scenario",
        response_model=TestScenarioDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Test scenario details",
                "content": common.example_json_content(test_scenario_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Test scenario not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve_scenario"),
    )
    async def read_test_scenario(
        request: Request,
        scenario_id: TestScenarioIdPath,
    ) -> TestScenarioDTO:
        """Get a test scenario by ID."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_TEST_SCENARIO,
        )

        try:
            scenario = await app.test_suites.read_scenario(TestScenarioId(scenario_id))
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test scenario not found: {scenario_id}",
            )

        return _test_scenario_to_dto(scenario)

    @router.patch(
        "/scenarios/{scenario_id}",
        operation_id="update_test_scenario",
        response_model=TestScenarioDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Test scenario updated",
                "content": common.example_json_content(test_scenario_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Test scenario not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="update_scenario"),
    )
    async def update_test_scenario(
        request: Request,
        scenario_id: TestScenarioIdPath,
        params: TestScenarioUpdateParamsDTO,
    ) -> TestScenarioDTO:
        """Update a test scenario."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_TEST_SCENARIO,
        )

        update_params: TestScenarioUpdateParams = {}
        if params.name is not None:
            update_params["name"] = params.name
        if params.description is not None:
            update_params["description"] = params.description
        if params.steps is not None:
            update_params["steps"] = [_test_step_from_dto(s) for s in params.steps]
        if params.customer_id is not None:
            update_params["customer_id"] = (
                CustomerId(params.customer_id) if params.customer_id else None
            )
        if params.repetitions is not None:
            update_params["repetitions"] = params.repetitions

        try:
            scenario = await app.test_suites.update_scenario(
                TestScenarioId(scenario_id), update_params
            )
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test scenario not found: {scenario_id}",
            )

        return _test_scenario_to_dto(scenario)

    @router.delete(
        "/scenarios/{scenario_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_test_scenario",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Test scenario deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "Test scenario not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete_scenario"),
    )
    async def delete_test_scenario(
        request: Request,
        scenario_id: TestScenarioIdPath,
    ) -> None:
        """Delete a test scenario."""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.DELETE_TEST_SCENARIO,
        )

        try:
            await app.test_suites.delete_scenario(TestScenarioId(scenario_id))
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test scenario not found: {scenario_id}",
            )

    # Test Run endpoints

    @router.post(
        "/{suite_id}/runs",
        status_code=status.HTTP_202_ACCEPTED,
        operation_id="start_test_run",
        response_model=TestRunDTO,
        responses={
            status.HTTP_202_ACCEPTED: {
                "description": "Test run started",
                "content": common.example_json_content(test_run_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Test suite not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="start_run"),
    )
    async def start_test_run(
        request: Request,
        suite_id: TestSuiteIdPath,
    ) -> TestRunDTO:
        """Start a new test run for a suite.

        The tests are executed synchronously and the results are returned.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.CREATE_TEST_RUN,
        )

        try:
            run = await app.test_suites.run_suite(TestSuiteId(suite_id))
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test suite not found: {suite_id}",
            )

        return _test_run_to_dto(run)

    # WebSocket endpoint for streaming test execution

    @router.websocket("/{suite_id}/run-ws")
    async def run_test_suite_ws(
        websocket: WebSocket,
        suite_id: str,
    ) -> None:
        """Run a test suite with real-time progress streaming via WebSocket.

        Connect to this endpoint to start a test run and receive real-time events.

        Events sent over WebSocket:
        - suite_start: Test suite execution started
        - test_start: Individual test started
        - message_sent: Customer message sent
        - waiting_for_agent: Waiting for agent response
        - message_received: Agent response received
        - evaluating: Evaluating assertions
        - condition_result: Single assertion result
        - assertion_score: Assertion score
        - test_passed: Test passed
        - test_failed: Test failed
        - suite_end: Test suite execution completed

        The final message will include the complete test run results.
        """
        await websocket.accept()

        try:
            # Create WebSocket listener
            listener = WebSocketTestEventListener(websocket)

            # Run the suite with the listener
            run = await app.test_suites.run_suite(
                suite_id=TestSuiteId(suite_id),
                listener=listener,
            )

            # Send final results
            await websocket.send_json(
                {
                    "type": "run_complete",
                    "data": {
                        "run_id": run.id,
                        "status": run.status.value,
                        "total": run.total,
                        "passed": run.passed,
                        "failed": run.failed,
                        "errors": run.errors,
                        "duration_ms": run.duration_ms,
                    },
                }
            )

        except ItemNotFoundError as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {"message": str(e)},
                }
            )
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {"message": f"Test execution error: {e}"},
                    }
                )
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    @router.websocket("/scenarios/{scenario_id}/run-ws")
    async def run_test_scenario_ws(
        websocket: WebSocket,
        scenario_id: str,
    ) -> None:
        """Run a single test scenario with real-time progress streaming via WebSocket.

        Similar to run_test_suite_ws but executes a single scenario.
        """
        await websocket.accept()

        try:
            # Create WebSocket listener
            listener = WebSocketTestEventListener(websocket)

            # Run the scenario with the listener
            result = await app.test_suites.run_scenario(
                scenario_id=TestScenarioId(scenario_id),
                listener=listener,
            )

            # Send final results
            await websocket.send_json(
                {
                    "type": "scenario_complete",
                    "data": {
                        "scenario_id": result.scenario_id,
                        "scenario_name": result.scenario_name,
                        "status": result.status.value,
                        "duration_ms": result.duration_ms,
                        "error": result.error,
                    },
                }
            )

        except ItemNotFoundError as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {"message": str(e)},
                }
            )
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {"message": f"Test execution error: {e}"},
                    }
                )
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    return router
