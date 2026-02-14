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
Tests for TestSuiteDocumentStore and TestSuiteModule.

Note: Basic CRUD operations are primarily tested via API tests in
tests/api/test_test_suites.py. This file focuses on:
- Store-specific behavior (cascading deletes, serialization)
- Module execution logic with mocks
- Event listener protocol
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator
from unittest.mock import patch

import pytest
from lagom import Container
from pytest import fixture

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.agents import AgentId, AgentStore
from parlant.core.app_modules.test_suites import (
    NullEventListener,
    TestRunCancelledError,
    TestSuiteModule,
)
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.common import IdGenerator, ItemNotFoundError
from parlant.core.loggers import Logger
from parlant.core.nlp.service import NLPService
from parlant.core.test_suites import (
    TestScenarioResult,
    TestStep,
    TestStepStatus,
    TestSuiteDocumentStore,
    TestSuiteStore,
    TestRunDocumentStore,
    TestRunStore,
    TestRunStatus,
    TestRunUpdateParams,
)


@fixture
def id_generator() -> IdGenerator:
    return IdGenerator()


@fixture
async def test_suite_store(
    id_generator: IdGenerator,
) -> AsyncIterator[TestSuiteStore]:
    async with TestSuiteDocumentStore(
        id_generator=id_generator,
        database=TransientDocumentDatabase(),
    ) as store:
        yield store


@fixture
async def test_run_store(
    id_generator: IdGenerator,
) -> AsyncIterator[TestRunStore]:
    async with TestRunDocumentStore(
        id_generator=id_generator,
        database=TransientDocumentDatabase(),
    ) as store:
        yield store


# =============================================================================
# Store-Specific Behavior Tests
# =============================================================================


async def test_that_deleting_suite_cascades_to_scenarios(
    test_suite_store: TestSuiteStore,
) -> None:
    """Verify cascading delete - scenarios are deleted when suite is deleted."""
    agent_id = AgentId("agent_123")

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="Suite with Scenarios",
        description="",
    )

    scenario = await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Test Scenario",
        description="",
        steps=[TestStep(role="customer", content="Hello")],
    )

    await test_suite_store.delete_suite(suite.id)

    with pytest.raises(ItemNotFoundError):
        await test_suite_store.read_scenario(scenario.id)


async def test_that_tool_steps_are_serialized_correctly(
    test_suite_store: TestSuiteStore,
) -> None:
    """Verify tool step data survives round-trip through store."""
    agent_id = AgentId("agent_123")

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    tool_arguments = {"customer_id": "123", "include_past": True}
    tool_response = {"appointments": [{"date": "2024-01-15", "time": "10:00"}]}

    scenario = await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Tool Test",
        description="",
        steps=[
            TestStep(role="customer", content="Check appointments"),
            TestStep(
                role="tool",
                content="appointments:get_appointments",
                tool_arguments=tool_arguments,
                tool_response=tool_response,
            ),
            TestStep(role="agent", content="", should="mention appointment"),
        ],
    )

    # Read back and verify
    read_scenario = await test_suite_store.read_scenario(scenario.id)
    tool_step = read_scenario.steps[1]

    assert tool_step.role == "tool"
    assert tool_step.tool_arguments == tool_arguments
    assert tool_step.tool_response == tool_response


async def test_that_run_results_are_stored_correctly(
    test_suite_store: TestSuiteStore,
    test_run_store: TestRunStore,
) -> None:
    """Verify complex run results survive round-trip through store."""
    agent_id = AgentId("agent_123")

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    scenario = await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Test Scenario",
        description="",
        steps=[TestStep(role="customer", content="Hello")],
    )

    run = await test_run_store.create_run(
        suite_id=suite.id,
        agent_id=agent_id,
    )

    scenario_results = [
        TestScenarioResult(
            scenario_id=scenario.id,
            scenario_name="Test Scenario",
            status=TestStepStatus.PASSED,
            duration_ms=1234.5,
            step_results=[],
            repetition=1,
        )
    ]

    await test_run_store.update_run(
        run.id,
        TestRunUpdateParams(
            status=TestRunStatus.COMPLETED,
            total=1,
            passed=1,
            failed=0,
            errors=0,
            duration_ms=1234.5,
            scenario_results=scenario_results,
        ),
    )

    # Read back and verify
    read_run = await test_run_store.read_run(run.id)

    assert read_run.status == TestRunStatus.COMPLETED
    assert read_run.total == 1
    assert read_run.passed == 1
    assert len(read_run.scenario_results) == 1
    assert read_run.scenario_results[0].scenario_name == "Test Scenario"
    assert read_run.scenario_results[0].duration_ms == 1234.5


# =============================================================================
# NullEventListener and TestRunCancelledError Tests
# =============================================================================


async def test_that_null_event_listener_is_not_cancelled() -> None:
    listener = NullEventListener()
    assert listener.is_cancelled() is False


async def test_that_null_event_listener_methods_are_no_ops() -> None:
    listener = NullEventListener()

    # All methods should complete without error
    await listener.on_suite_start("suite-1", "Test Suite", "agent-1", 2, ["test1", "test2"])
    await listener.on_test_start("test1")
    await listener.on_message_sent("test1", "customer", "Hello")
    await listener.on_message_received("test1", "agent", "Hi", None, "trace-123")
    await listener.on_evaluating("test1", ["should greet"])
    await listener.on_condition_result("test1", "should greet", True)
    await listener.on_assertion_score("test1", 85.0)
    await listener.on_test_passed("test1", 1234.5, None)
    await listener.on_test_failed("test1", 1234.5, "Failed", {})
    await listener.on_suite_end({})


def test_that_test_run_cancelled_error_is_an_exception() -> None:
    error = TestRunCancelledError()
    assert isinstance(error, Exception)

    with pytest.raises(TestRunCancelledError):
        raise TestRunCancelledError()


# =============================================================================
# TestSuiteModule Execution Tests (Mocked)
# =============================================================================


class MockResponse:
    """Mock response from session.send()."""

    def __init__(
        self,
        message: str = "Hello! How can I help you?",
        tool_calls: list[Any] | None = None,
        trace_id: str | None = "trace-123",
    ) -> None:
        self.message = message
        self.tool_calls = tool_calls
        self.trace_id = trace_id
        self._should_pass = True
        self._should_score = 85.0

    async def should(self, condition: Any) -> float:
        if not self._should_pass:
            raise AssertionError(f"Assertion failed (score: {self._should_score}/100)")
        return self._should_score


class MockSession:
    """Mock session from Suite.session()."""

    def __init__(self, session_id: str = "session-123") -> None:
        self.id = session_id
        self._responses: list[MockResponse] = []
        self._response_index = 0

    def add_response(self, response: MockResponse) -> None:
        self._responses.append(response)

    async def send(self, content: str) -> MockResponse:
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
            return response
        return MockResponse()


class MockSuite:
    """Mock Suite class."""

    def __init__(self, **kwargs: Any) -> None:
        self._session = MockSession()

    def session(self, transient: bool = False) -> "MockSuiteSessionContext":
        return MockSuiteSessionContext(self._session)

    async def delete_queued_sessions(self) -> None:
        pass


class MockSuiteSessionContext:
    """Async context manager for mock session."""

    def __init__(self, session: MockSession) -> None:
        self._session = session

    async def __aenter__(self) -> MockSession:
        return self._session

    async def __aexit__(self, *args: Any) -> None:
        pass


@fixture
async def test_suite_module(container: Container) -> TestSuiteModule:
    return TestSuiteModule(
        logger=container[Logger],
        test_suite_store=container[TestSuiteStore],
        test_run_store=container[TestRunStore],
        agent_store=container[AgentStore],
        background_task_service=container[BackgroundTaskService],
        nlp_service=container[NLPService],
        server_url="http://localhost:8800",
    )


async def test_that_run_suite_executes_scenarios_and_records_results(
    container: Container,
    test_suite_module: TestSuiteModule,
) -> None:
    agent_store = container[AgentStore]
    test_suite_store = container[TestSuiteStore]

    agent = await agent_store.create_agent(name="Test Agent")
    suite = await test_suite_store.create_suite(
        agent_id=agent.id, name="Test Suite", description=""
    )

    for i in range(3):
        await test_suite_store.create_scenario(
            suite_id=suite.id,
            name=f"Test {i + 1}",
            description="",
            steps=[TestStep(role="customer", content=f"Message {i + 1}")],
        )

    with patch("parlant.testing.suite.Suite", MockSuite):
        run = await test_suite_module.run_suite(suite.id)

    assert run.status == TestRunStatus.COMPLETED
    assert run.total == 3
    assert run.passed == 3
    assert len(run.scenario_results) == 3


async def test_that_run_suite_records_failed_assertions(
    container: Container,
    test_suite_module: TestSuiteModule,
) -> None:
    agent_store = container[AgentStore]
    test_suite_store = container[TestSuiteStore]

    agent = await agent_store.create_agent(name="Test Agent")
    suite = await test_suite_store.create_suite(
        agent_id=agent.id, name="Test Suite", description=""
    )
    await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Failing Test",
        description="",
        steps=[
            TestStep(role="customer", content="Hello"),
            TestStep(role="agent", content="", should="do something impossible"),
        ],
    )

    class FailingMockSuite(MockSuite):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            failing_response = MockResponse()
            failing_response._should_pass = False
            self._session.add_response(failing_response)

    with patch("parlant.testing.suite.Suite", FailingMockSuite):
        run = await test_suite_module.run_suite(suite.id)

    assert run.passed == 0
    assert run.failed == 1
    assert run.scenario_results[0].status == TestStepStatus.FAILED


async def test_that_run_suite_handles_cancellation(
    container: Container,
    test_suite_module: TestSuiteModule,
) -> None:
    agent_store = container[AgentStore]
    test_suite_store = container[TestSuiteStore]

    agent = await agent_store.create_agent(name="Test Agent")
    suite = await test_suite_store.create_suite(
        agent_id=agent.id, name="Test Suite", description=""
    )

    for i in range(5):
        await test_suite_store.create_scenario(
            suite_id=suite.id,
            name=f"Test {i + 1}",
            description="",
            steps=[TestStep(role="customer", content=f"Message {i + 1}")],
        )

    class CancellingListener(NullEventListener):
        def __init__(self) -> None:
            self._test_count = 0

        def is_cancelled(self) -> bool:
            return self._test_count > 0

        async def on_test_passed(
            self, test_name: str, duration_ms: float, details: Any = None
        ) -> None:
            self._test_count += 1

    with patch("parlant.testing.suite.Suite", MockSuite):
        run = await test_suite_module.run_suite(suite.id, listener=CancellingListener())

    assert run.status == TestRunStatus.CANCELLED
    assert run.passed < 5


async def test_that_event_listener_receives_correct_data(
    container: Container,
    test_suite_module: TestSuiteModule,
) -> None:
    agent_store = container[AgentStore]
    test_suite_store = container[TestSuiteStore]

    agent = await agent_store.create_agent(name="Test Agent")
    suite = await test_suite_store.create_suite(
        agent_id=agent.id, name="My Test Suite", description=""
    )
    await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Greeting Scenario",
        description="",
        steps=[TestStep(role="customer", content="Hello there!")],
    )

    @dataclass
    class EventRecord:
        name: str
        data: dict[str, Any]

    class DataCapturingListener(NullEventListener):
        def __init__(self) -> None:
            self.events: list[EventRecord] = []

        async def on_suite_start(
            self,
            suite_id: str,
            suite_name: str,
            agent_id: str,
            total_tests: int,
            test_names: list[str],
        ) -> None:
            self.events.append(
                EventRecord(
                    "suite_start",
                    {
                        "suite_id": suite_id,
                        "suite_name": suite_name,
                        "agent_id": agent_id,
                        "total_tests": total_tests,
                        "test_names": test_names,
                    },
                )
            )

        async def on_message_sent(self, test_name: str, role: str, content: str) -> None:
            self.events.append(EventRecord("message_sent", {"role": role, "content": content}))

        async def on_message_received(
            self,
            test_name: str,
            role: str,
            content: str,
            tool_calls: Any = None,
            trace_id: str | None = None,
        ) -> None:
            self.events.append(
                EventRecord(
                    "message_received",
                    {"role": role, "content": content, "trace_id": trace_id},
                )
            )

    listener = DataCapturingListener()

    with patch("parlant.testing.suite.Suite", MockSuite):
        await test_suite_module.run_suite(suite.id, listener=listener)

    # Verify suite_start data
    suite_start = next(e for e in listener.events if e.name == "suite_start")
    assert suite_start.data["suite_name"] == "My Test Suite"
    assert suite_start.data["total_tests"] == 1
    assert suite_start.data["test_names"] == ["Greeting Scenario"]

    # Verify message_sent data
    msg_sent = next(e for e in listener.events if e.name == "message_sent")
    assert msg_sent.data["role"] == "customer"
    assert msg_sent.data["content"] == "Hello there!"

    # Verify message_received data
    msg_recv = next(e for e in listener.events if e.name == "message_received")
    assert msg_recv.data["role"] == "agent"
    assert msg_recv.data["content"] == "Hello! How can I help you?"
    assert msg_recv.data["trace_id"] == "trace-123"
