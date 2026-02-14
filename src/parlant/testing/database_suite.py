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

"""Adapter to run database-stored test scenarios via CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Sequence

from parlant.core.test_suites import TestScenario
from parlant.testing.suite import Scenario, Suite


@dataclass
class DatabaseScenario:
    """Wraps a database TestScenario as an executable Scenario."""

    db_scenario: TestScenario
    suite: "DatabaseSuite"

    @property
    def name(self) -> str:
        return self.db_scenario.name

    @property
    def repetitions(self) -> int:
        return self.db_scenario.repetitions

    @property
    def func(self) -> Callable[..., Awaitable[None]]:
        """Return an async function that executes this scenario."""
        return self._execute

    async def _execute(self) -> None:
        """Execute the scenario steps."""
        from parlant.testing.response import Should

        steps = self.db_scenario.steps
        customer_id = str(self.db_scenario.customer_id) if self.db_scenario.customer_id else None

        async with self.suite.session(customer_id=customer_id) as session:
            for idx, step in enumerate(steps):
                # Skip tool steps - they would need to be registered as mocks
                # For now, we only support customer/agent step execution
                if step.role == "tool":
                    continue

                if step.role == "customer":
                    # Send customer message
                    response = await session.send(step.content)

                    # Check if next step is an agent assertion
                    if idx + 1 < len(steps):
                        next_step = steps[idx + 1]
                        if next_step.role == "agent" and next_step.should:
                            # Evaluate the assertion
                            await response.should(Should(next_step.should, next_step.should_weight))

                elif step.role == "agent":
                    # Agent steps are processed alongside customer steps above
                    pass


class DatabaseSuite(Suite):
    """Adapter that wraps database TestScenarios to run via CLI.

    This allows running tests defined in the database using the same
    TestRunner infrastructure as file-based tests.
    """

    def __init__(
        self,
        server_url: str,
        agent_id: str,
        scenarios: Sequence[TestScenario],
        customer_id: Optional[str] = None,
        response_timeout: int = 60,
    ) -> None:
        """Initialize the database suite adapter.

        Args:
            server_url: URL of the Parlant server.
            agent_id: Default agent ID for sessions.
            scenarios: Database scenarios to run.
            customer_id: Default customer ID (None = guest).
            response_timeout: Default timeout for agent responses.
        """
        super().__init__(
            server_url=server_url,
            agent_id=agent_id,
            customer_id=customer_id,
            response_timeout=response_timeout,
        )
        self._db_scenarios = scenarios

    def get_scenarios(self) -> List[Scenario]:
        """Get all registered scenarios as Scenario objects."""
        result: List[Scenario] = []
        for db_scenario in self._db_scenarios:
            wrapper = DatabaseScenario(db_scenario=db_scenario, suite=self)
            # Convert to Scenario dataclass expected by runner
            result.append(
                Scenario(
                    name=wrapper.name,
                    func=wrapper.func,
                    repetitions=wrapper.repetitions,
                )
            )
        return result
