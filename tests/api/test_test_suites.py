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

from fastapi import status
import httpx
from lagom import Container

from parlant.core.agents import AgentId, AgentStore
from parlant.core.test_suites import (
    TestStep,
    TestSuiteStore,
    TestRunStore,
    TestRunStatus,
    TestRunUpdateParams,
    TestStepStatus,
    TestScenarioResult,
)


# =============================================================================
# TestSuite API Tests
# =============================================================================


async def test_that_a_test_suite_can_be_created_via_api(
    async_client: httpx.AsyncClient,
    agent_id: AgentId,
) -> None:
    response = await async_client.post(
        "/test-suites",
        json={
            "agent_id": agent_id,
            "name": "My Test Suite",
            "description": "A suite for testing",
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    suite = response.json()
    assert suite["agent_id"] == agent_id
    assert suite["name"] == "My Test Suite"
    assert suite["description"] == "A suite for testing"
    assert "id" in suite
    assert "creation_utc" in suite


async def test_that_creating_suite_for_nonexistent_agent_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/test-suites",
        json={
            "agent_id": "nonexistent_agent",
            "name": "My Test Suite",
            "description": "",
        },
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_test_suite_can_be_read_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Test Suite",
        description="A test suite",
    )

    response = await async_client.get(f"/test-suites/{suite.id}")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["id"] == suite.id
    assert data["name"] == "My Test Suite"
    assert data["description"] == "A test suite"


async def test_that_reading_nonexistent_suite_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/test-suites/nonexistent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_test_suites_can_be_listed_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    await test_suite_store.create_suite(
        agent_id=agent_id,
        name="Suite 1",
        description="",
    )
    await test_suite_store.create_suite(
        agent_id=agent_id,
        name="Suite 2",
        description="",
    )

    response = await async_client.get("/test-suites")

    assert response.status_code == status.HTTP_200_OK

    suites = response.json()
    assert len(suites) == 2
    assert any(s["name"] == "Suite 1" for s in suites)
    assert any(s["name"] == "Suite 2" for s in suites)


async def test_that_test_suites_can_be_filtered_by_agent_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    agent_store = container[AgentStore]
    test_suite_store = container[TestSuiteStore]

    agent_1 = await agent_store.create_agent(name="Agent 1")
    agent_2 = await agent_store.create_agent(name="Agent 2")

    await test_suite_store.create_suite(
        agent_id=agent_1.id,
        name="Suite for Agent 1",
        description="",
    )
    await test_suite_store.create_suite(
        agent_id=agent_2.id,
        name="Suite for Agent 2",
        description="",
    )

    response = await async_client.get(f"/test-suites?agent_id={agent_1.id}")

    assert response.status_code == status.HTTP_200_OK

    suites = response.json()
    assert len(suites) == 1
    assert suites[0]["name"] == "Suite for Agent 1"


async def test_that_a_test_suite_can_be_updated_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="Original Name",
        description="Original Description",
    )

    response = await async_client.patch(
        f"/test-suites/{suite.id}",
        json={
            "name": "Updated Name",
            "description": "Updated Description",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["id"] == suite.id
    assert data["name"] == "Updated Name"
    assert data["description"] == "Updated Description"


async def test_that_updating_nonexistent_suite_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.patch(
        "/test-suites/nonexistent",
        json={"name": "New Name"},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_test_suite_can_be_deleted_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="Suite to Delete",
        description="",
    )

    response = await async_client.delete(f"/test-suites/{suite.id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify it's gone
    read_response = await async_client.get(f"/test-suites/{suite.id}")
    assert read_response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_deleting_nonexistent_suite_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.delete("/test-suites/nonexistent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


# =============================================================================
# TestScenario API Tests
# =============================================================================


async def test_that_a_test_scenario_can_be_created_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    response = await async_client.post(
        f"/test-suites/{suite.id}/scenarios",
        json={
            "name": "Greeting Test",
            "description": "Tests greeting behavior",
            "steps": [
                {"role": "customer", "content": "Hello"},
                {"role": "agent", "content": "", "should": "greet the customer"},
            ],
            "repetitions": 3,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    scenario = response.json()
    assert scenario["suite_id"] == suite.id
    assert scenario["name"] == "Greeting Test"
    assert scenario["description"] == "Tests greeting behavior"
    assert len(scenario["steps"]) == 2
    assert scenario["steps"][0]["role"] == "customer"
    assert scenario["steps"][0]["content"] == "Hello"
    assert scenario["steps"][1]["role"] == "agent"
    assert scenario["steps"][1]["should"] == "greet the customer"
    assert scenario["repetitions"] == 3


async def test_that_a_test_scenario_with_tool_step_can_be_created_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    response = await async_client.post(
        f"/test-suites/{suite.id}/scenarios",
        json={
            "name": "Tool Test",
            "description": "",
            "steps": [
                {"role": "customer", "content": "Check my appointment"},
                {
                    "role": "tool",
                    "content": "appointments:get_appointments",
                    "tool_arguments": {"customer_id": "123"},
                    "tool_response": {"appointments": []},
                },
                {"role": "agent", "content": "", "should": "say no appointments found"},
            ],
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    scenario = response.json()
    tool_step = scenario["steps"][1]
    assert tool_step["role"] == "tool"
    assert tool_step["content"] == "appointments:get_appointments"
    assert tool_step["tool_arguments"] == {"customer_id": "123"}
    assert tool_step["tool_response"] == {"appointments": []}


async def test_that_creating_scenario_for_nonexistent_suite_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.post(
        "/test-suites/nonexistent/scenarios",
        json={
            "name": "Test",
            "description": "",
            "steps": [{"role": "customer", "content": "Hello"}],
        },
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_test_scenario_can_be_read_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    scenario = await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Test Scenario",
        description="A test scenario",
        steps=[TestStep(role="customer", content="Hello")],
    )

    response = await async_client.get(f"/test-suites/scenarios/{scenario.id}")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["id"] == scenario.id
    assert data["name"] == "Test Scenario"
    assert data["description"] == "A test scenario"


async def test_that_reading_nonexistent_scenario_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/test-suites/scenarios/nonexistent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_test_scenarios_can_be_listed_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Scenario 1",
        description="",
        steps=[TestStep(role="customer", content="Hello")],
    )
    await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Scenario 2",
        description="",
        steps=[TestStep(role="customer", content="Hi")],
    )

    response = await async_client.get(f"/test-suites/{suite.id}/scenarios")

    assert response.status_code == status.HTTP_200_OK

    scenarios = response.json()
    assert len(scenarios) == 2
    assert any(s["name"] == "Scenario 1" for s in scenarios)
    assert any(s["name"] == "Scenario 2" for s in scenarios)


async def test_that_a_test_scenario_can_be_updated_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    scenario = await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Original Name",
        description="Original Description",
        steps=[TestStep(role="customer", content="Hello")],
        repetitions=1,
    )

    response = await async_client.patch(
        f"/test-suites/scenarios/{scenario.id}",
        json={
            "name": "Updated Name",
            "description": "Updated Description",
            "steps": [
                {"role": "customer", "content": "Updated message"},
                {"role": "agent", "content": "", "should": "respond"},
            ],
            "repetitions": 5,
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["id"] == scenario.id
    assert data["name"] == "Updated Name"
    assert data["description"] == "Updated Description"
    assert len(data["steps"]) == 2
    assert data["steps"][0]["content"] == "Updated message"
    assert data["repetitions"] == 5


async def test_that_updating_nonexistent_scenario_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.patch(
        "/test-suites/scenarios/nonexistent",
        json={"name": "New Name"},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_a_test_scenario_can_be_deleted_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    scenario = await test_suite_store.create_scenario(
        suite_id=suite.id,
        name="Scenario to Delete",
        description="",
        steps=[TestStep(role="customer", content="Hello")],
    )

    response = await async_client.delete(f"/test-suites/scenarios/{scenario.id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify it's gone
    read_response = await async_client.get(f"/test-suites/scenarios/{scenario.id}")
    assert read_response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_deleting_nonexistent_scenario_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.delete("/test-suites/scenarios/nonexistent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


# =============================================================================
# TestRun API Tests
# =============================================================================


async def test_that_a_test_run_can_be_read_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]
    test_run_store = container[TestRunStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    run = await test_run_store.create_run(
        suite_id=suite.id,
        agent_id=agent_id,
    )

    response = await async_client.get(f"/test-suites/runs/{run.id}")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["id"] == run.id
    assert data["suite_id"] == suite.id
    assert data["status"] == "pending"


async def test_that_reading_nonexistent_run_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/test-suites/runs/nonexistent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_test_runs_can_be_listed_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]
    test_run_store = container[TestRunStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    await test_run_store.create_run(suite_id=suite.id, agent_id=agent_id)
    await test_run_store.create_run(suite_id=suite.id, agent_id=agent_id)

    response = await async_client.get("/test-suites/runs")

    assert response.status_code == status.HTTP_200_OK

    runs = response.json()
    assert len(runs) == 2


async def test_that_test_runs_can_be_filtered_by_suite_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]
    test_run_store = container[TestRunStore]

    suite_1 = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="Suite 1",
        description="",
    )
    suite_2 = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="Suite 2",
        description="",
    )

    await test_run_store.create_run(suite_id=suite_1.id, agent_id=agent_id)
    await test_run_store.create_run(suite_id=suite_1.id, agent_id=agent_id)
    await test_run_store.create_run(suite_id=suite_2.id, agent_id=agent_id)

    response = await async_client.get(f"/test-suites/runs?suite_id={suite_1.id}")

    assert response.status_code == status.HTTP_200_OK

    runs = response.json()
    assert len(runs) == 2
    assert all(r["suite_id"] == suite_1.id for r in runs)


async def test_that_a_test_run_can_be_deleted_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]
    test_run_store = container[TestRunStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    run = await test_run_store.create_run(
        suite_id=suite.id,
        agent_id=agent_id,
    )

    response = await async_client.delete(f"/test-suites/runs/{run.id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify it's gone
    read_response = await async_client.get(f"/test-suites/runs/{run.id}")
    assert read_response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_deleting_nonexistent_run_returns_404(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.delete("/test-suites/runs/nonexistent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_that_test_runs_can_be_bulk_deleted_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]
    test_run_store = container[TestRunStore]

    suite = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="My Suite",
        description="",
    )

    await test_run_store.create_run(suite_id=suite.id, agent_id=agent_id)
    await test_run_store.create_run(suite_id=suite.id, agent_id=agent_id)
    await test_run_store.create_run(suite_id=suite.id, agent_id=agent_id)

    response = await async_client.delete("/test-suites/runs")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["deleted_count"] == 3

    # Verify they're all gone
    list_response = await async_client.get("/test-suites/runs")
    assert list_response.json() == []


async def test_that_test_runs_can_be_bulk_deleted_by_suite_via_api(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]
    test_run_store = container[TestRunStore]

    suite_1 = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="Suite 1",
        description="",
    )
    suite_2 = await test_suite_store.create_suite(
        agent_id=agent_id,
        name="Suite 2",
        description="",
    )

    await test_run_store.create_run(suite_id=suite_1.id, agent_id=agent_id)
    await test_run_store.create_run(suite_id=suite_1.id, agent_id=agent_id)
    await test_run_store.create_run(suite_id=suite_2.id, agent_id=agent_id)

    response = await async_client.delete(f"/test-suites/runs?suite_id={suite_1.id}")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["deleted_count"] == 2

    # Verify suite_2's run is still there
    list_response = await async_client.get("/test-suites/runs")
    runs = list_response.json()
    assert len(runs) == 1
    assert runs[0]["suite_id"] == suite_2.id


async def test_that_test_run_includes_scenario_results(
    async_client: httpx.AsyncClient,
    container: Container,
    agent_id: AgentId,
) -> None:
    test_suite_store = container[TestSuiteStore]
    test_run_store = container[TestRunStore]

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

    # Update with results
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

    response = await async_client.get(f"/test-suites/runs/{run.id}")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "completed"
    assert data["total"] == 1
    assert data["passed"] == 1
    assert len(data["scenario_results"]) == 1
    assert data["scenario_results"][0]["scenario_name"] == "Test Scenario"
    assert data["scenario_results"][0]["status"] == "passed"
    assert data["scenario_results"][0]["duration_ms"] == 1234.5
