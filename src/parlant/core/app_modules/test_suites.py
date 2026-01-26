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

"""Test Suites application module.

Provides the application-level interface for test suite management and execution.
"""

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

from parlant.core.agents import AgentId, AgentStore
from parlant.core.customers import CustomerId
from parlant.core.loggers import Logger
from parlant.core.test_suites import (
    TestRun,
    TestRunId,
    TestRunStatus,
    TestRunUpdateParams,
    TestScenario,
    TestScenarioId,
    TestScenarioResult,
    TestScenarioUpdateParams,
    TestStep,
    TestStepResult,
    TestStepStatus,
    TestSuite,
    TestSuiteId,
    TestSuiteStore,
    TestSuiteUpdateParams,
)

if TYPE_CHECKING:
    from parlant.testing.runner import TestEventListener


# Define a minimal protocol for TestEventListener to avoid circular imports
class TestEventListenerProtocol(Protocol):
    """Protocol for test event listeners."""

    async def on_suite_start(self, suite_name: str, total_tests: int) -> None: ...
    async def on_test_start(self, test_name: str) -> None: ...
    async def on_message_sent(self, test_name: str, role: str, content: str) -> None: ...
    async def on_message_received(self, test_name: str, role: str, content: str) -> None: ...
    async def on_evaluating(self, test_name: str, conditions: list[str]) -> None: ...
    async def on_condition_result(self, test_name: str, condition: str, passed: bool) -> None: ...
    async def on_assertion_score(self, test_name: str, score: float) -> None: ...
    async def on_test_passed(self, test_name: str, duration_ms: float) -> None: ...
    async def on_test_failed(
        self, test_name: str, duration_ms: float, error: str, details: Any
    ) -> None: ...
    async def on_suite_end(self, report: Any) -> None: ...


class NullEventListener:
    """No-op event listener."""

    async def on_suite_start(self, suite_name: str, total_tests: int) -> None:
        pass

    async def on_test_start(self, test_name: str) -> None:
        pass

    async def on_message_sent(self, test_name: str, role: str, content: str) -> None:
        pass

    async def on_message_received(self, test_name: str, role: str, content: str) -> None:
        pass

    async def on_evaluating(self, test_name: str, conditions: list[str]) -> None:
        pass

    async def on_condition_result(self, test_name: str, condition: str, passed: bool) -> None:
        pass

    async def on_assertion_score(self, test_name: str, score: float) -> None:
        pass

    async def on_test_passed(self, test_name: str, duration_ms: float) -> None:
        pass

    async def on_test_failed(
        self, test_name: str, duration_ms: float, error: str, details: Any
    ) -> None:
        pass

    async def on_suite_end(self, report: Any) -> None:
        pass


class TestSuiteModule:
    """Application module for test suite management and execution."""

    def __init__(
        self,
        logger: Logger,
        test_suite_store: TestSuiteStore,
        agent_store: AgentStore,
        server_url: str = "http://localhost:8800",
    ) -> None:
        """Initialize the test suite module.

        Args:
            logger: Logger instance.
            test_suite_store: Store for test suite persistence.
            agent_store: Store for agent lookup.
            server_url: URL of the Parlant server for test execution.
        """
        self._logger = logger
        self._store = test_suite_store
        self._agent_store = agent_store
        self._server_url = server_url

    # TestSuite CRUD

    async def create_suite(
        self,
        agent_id: AgentId,
        name: str,
        description: str,
    ) -> TestSuite:
        """Create a new test suite.

        Args:
            agent_id: The agent this suite tests.
            name: Human-readable name.
            description: Description of what this suite tests.

        Returns:
            The created TestSuite.
        """
        # Verify agent exists
        await self._agent_store.read_agent(agent_id=agent_id)

        return await self._store.create_suite(
            agent_id=agent_id,
            name=name,
            description=description,
        )

    async def read_suite(self, suite_id: TestSuiteId) -> TestSuite:
        """Read a test suite by ID."""
        return await self._store.read_suite(suite_id)

    async def list_suites(
        self,
        agent_id: Optional[AgentId] = None,
    ) -> Sequence[TestSuite]:
        """List test suites, optionally filtered by agent."""
        return await self._store.list_suites(agent_id=agent_id)

    async def update_suite(
        self,
        suite_id: TestSuiteId,
        params: TestSuiteUpdateParams,
    ) -> TestSuite:
        """Update a test suite."""
        return await self._store.update_suite(suite_id, params)

    async def delete_suite(self, suite_id: TestSuiteId) -> None:
        """Delete a test suite and all its scenarios."""
        return await self._store.delete_suite(suite_id)

    # TestScenario CRUD

    async def create_scenario(
        self,
        suite_id: TestSuiteId,
        name: str,
        description: str,
        steps: Sequence[TestStep],
        customer_id: Optional[CustomerId] = None,
        repetitions: int = 1,
    ) -> TestScenario:
        """Create a new test scenario."""
        return await self._store.create_scenario(
            suite_id=suite_id,
            name=name,
            description=description,
            steps=steps,
            customer_id=customer_id,
            repetitions=repetitions,
        )

    async def read_scenario(self, scenario_id: TestScenarioId) -> TestScenario:
        """Read a test scenario by ID."""
        return await self._store.read_scenario(scenario_id)

    async def list_scenarios(self, suite_id: TestSuiteId) -> Sequence[TestScenario]:
        """List all scenarios in a suite."""
        return await self._store.list_scenarios(suite_id)

    async def update_scenario(
        self,
        scenario_id: TestScenarioId,
        params: TestScenarioUpdateParams,
    ) -> TestScenario:
        """Update a test scenario."""
        return await self._store.update_scenario(scenario_id, params)

    async def delete_scenario(self, scenario_id: TestScenarioId) -> None:
        """Delete a test scenario."""
        return await self._store.delete_scenario(scenario_id)

    # TestRun operations

    async def read_run(self, run_id: TestRunId) -> TestRun:
        """Read a test run by ID."""
        return await self._store.read_run(run_id)

    async def list_runs(
        self,
        suite_id: Optional[TestSuiteId] = None,
        limit: int = 50,
    ) -> Sequence[TestRun]:
        """List test runs, optionally filtered by suite."""
        return await self._store.list_runs(suite_id=suite_id, limit=limit)

    async def delete_run(self, run_id: TestRunId) -> None:
        """Delete a test run."""
        return await self._store.delete_run(run_id)

    async def delete_runs(self, suite_id: Optional[TestSuiteId] = None) -> int:
        """Delete test runs, optionally filtered by suite.

        Returns:
            Number of runs deleted.
        """
        return await self._store.delete_runs(suite_id=suite_id)

    # Test execution

    async def run_suite(
        self,
        suite_id: TestSuiteId,
        listener: Optional[TestEventListenerProtocol] = None,
    ) -> TestRun:
        """Execute all scenarios in a test suite.

        Args:
            suite_id: The suite to run.
            listener: Optional event listener for real-time updates.

        Returns:
            The completed TestRun with results.
        """
        suite = await self._store.read_suite(suite_id)
        scenarios = await self._store.list_scenarios(suite_id)

        # Create the run record
        run = await self._store.create_run(
            suite_id=suite_id,
            agent_id=suite.agent_id,
        )

        # Update to running status
        run = await self._store.update_run(
            run.id,
            TestRunUpdateParams(status=TestRunStatus.RUNNING),
        )

        listener = listener or NullEventListener()

        # Calculate total tests (with repetitions)
        total = sum(s.repetitions for s in scenarios)
        await listener.on_suite_start(suite.name, total)

        start_time = time.time()
        scenario_results: list[TestScenarioResult] = []
        passed = 0
        failed = 0
        errors = 0

        # Execute each scenario
        for scenario in scenarios:
            for rep in range(1, scenario.repetitions + 1):
                test_name = scenario.name
                if scenario.repetitions > 1:
                    test_name = f"{scenario.name}[rep_{rep}/{scenario.repetitions}]"

                try:
                    result = await self._execute_scenario(
                        scenario=scenario,
                        agent_id=suite.agent_id,
                        test_name=test_name,
                        repetition=rep,
                        listener=listener,
                    )
                    scenario_results.append(result)

                    if result.status == TestStepStatus.PASSED:
                        passed += 1
                    elif result.status == TestStepStatus.FAILED:
                        failed += 1
                    elif result.status == TestStepStatus.ERROR:
                        errors += 1

                except Exception as e:
                    self._logger.error(f"Error executing scenario {test_name}: {e}")
                    scenario_results.append(
                        TestScenarioResult(
                            scenario_id=scenario.id,
                            scenario_name=scenario.name,
                            status=TestStepStatus.ERROR,
                            duration_ms=0,
                            step_results=[],
                            error=str(e),
                            repetition=rep,
                        )
                    )
                    errors += 1
                    await listener.on_test_failed(test_name, 0, str(e), None)

        duration_ms = (time.time() - start_time) * 1000

        # Update run with final results
        run = await self._store.update_run(
            run.id,
            TestRunUpdateParams(
                status=TestRunStatus.COMPLETED,
                completion_utc=datetime.now(timezone.utc),
                total=total,
                passed=passed,
                failed=failed,
                errors=errors,
                duration_ms=duration_ms,
                scenario_results=scenario_results,
            ),
        )

        await listener.on_suite_end(None)  # type: ignore

        return run

    async def run_scenario(
        self,
        scenario_id: TestScenarioId,
        listener: Optional[TestEventListenerProtocol] = None,
    ) -> TestScenarioResult:
        """Execute a single scenario.

        Args:
            scenario_id: The scenario to run.
            listener: Optional event listener for real-time updates.

        Returns:
            The scenario result.
        """
        scenario = await self._store.read_scenario(scenario_id)
        suite = await self._store.read_suite(scenario.suite_id)

        listener = listener or NullEventListener()

        await listener.on_suite_start(scenario.name, 1)

        result = await self._execute_scenario(
            scenario=scenario,
            agent_id=suite.agent_id,
            test_name=scenario.name,
            repetition=1,
            listener=listener,
        )

        await listener.on_suite_end(None)  # type: ignore

        return result

    async def _execute_scenario(
        self,
        scenario: TestScenario,
        agent_id: AgentId,
        test_name: str,
        repetition: int,
        listener: TestEventListenerProtocol,
    ) -> TestScenarioResult:
        """Execute a single scenario and collect results.

        Uses the parlant.testing framework under the hood.
        """
        await listener.on_test_start(test_name)
        start_time = time.time()

        step_results: list[TestStepResult] = []

        try:
            # Lazy import to avoid circular imports at module load time
            from parlant.testing.suite import Suite
            from parlant.testing.response import Should

            # Create a Suite for this execution
            test_suite = Suite(
                server_url=self._server_url,
                agent_id=str(agent_id),
                customer_id=str(scenario.customer_id) if scenario.customer_id else None,
            )

            # Execute using session
            async with test_suite.session(transient=True) as session:
                conversation_history: list[tuple[str, str]] = []

                for idx, step in enumerate(scenario.steps):
                    if step.role == "customer":
                        # Send customer message
                        await listener.on_message_sent(test_name, "customer", step.content)
                        response = await session.send(step.content)
                        conversation_history.append(("Customer", step.content))

                        # Get actual agent response
                        actual_response = response.message
                        await listener.on_message_received(test_name, "agent", actual_response)
                        conversation_history.append(("Agent", actual_response))

                        step_results.append(
                            TestStepResult(
                                step_index=idx,
                                role="customer",
                                content=step.content,
                                actual_response=actual_response,
                            )
                        )

                        # If the next step is an agent step with assertion, evaluate it
                        if idx + 1 < len(scenario.steps):
                            next_step = scenario.steps[idx + 1]
                            if next_step.role == "agent" and next_step.should:
                                await listener.on_evaluating(test_name, [next_step.should])

                                try:
                                    score = await response.should(
                                        Should(next_step.should, next_step.should_weight)
                                    )
                                    await listener.on_condition_result(
                                        test_name, next_step.should, True
                                    )
                                    await listener.on_assertion_score(test_name, score)

                                    step_results.append(
                                        TestStepResult(
                                            step_index=idx + 1,
                                            role="agent",
                                            content=next_step.content,
                                            actual_response=actual_response,
                                            assertion=next_step.should,
                                            assertion_passed=True,
                                            assertion_score=score,
                                        )
                                    )
                                except AssertionError as e:
                                    await listener.on_condition_result(
                                        test_name, next_step.should, False
                                    )

                                    duration_ms = (time.time() - start_time) * 1000
                                    step_results.append(
                                        TestStepResult(
                                            step_index=idx + 1,
                                            role="agent",
                                            content=next_step.content,
                                            actual_response=actual_response,
                                            assertion=next_step.should,
                                            assertion_passed=False,
                                            assertion_reasoning=str(e),
                                        )
                                    )

                                    await listener.on_test_failed(
                                        test_name,
                                        duration_ms,
                                        str(e),
                                        {"actual": actual_response, "expected": next_step.should},
                                    )

                                    return TestScenarioResult(
                                        scenario_id=scenario.id,
                                        scenario_name=scenario.name,
                                        status=TestStepStatus.FAILED,
                                        duration_ms=duration_ms,
                                        step_results=step_results,
                                        error=str(e),
                                        repetition=repetition,
                                    )

                    elif step.role == "agent":
                        # Agent steps without preceding customer message are just placeholders
                        # They're processed alongside the customer step above
                        pass

            duration_ms = (time.time() - start_time) * 1000
            await listener.on_test_passed(test_name, duration_ms)

            return TestScenarioResult(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                status=TestStepStatus.PASSED,
                duration_ms=duration_ms,
                step_results=step_results,
                repetition=repetition,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await listener.on_test_failed(test_name, duration_ms, str(e), None)

            return TestScenarioResult(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                status=TestStepStatus.ERROR,
                duration_ms=duration_ms,
                step_results=step_results,
                error=str(e),
                repetition=repetition,
            )
