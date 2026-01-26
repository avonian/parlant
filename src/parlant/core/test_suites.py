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

"""Test Suites module for automated agent testing.

This module provides data models and store interface for managing
test suites, scenarios, and test runs for Parlant agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Mapping, NewType, Optional, Sequence, cast

from typing_extensions import Self, TypedDict, override

from parlant.core.agents import AgentId
from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import (
    IdGenerator,
    ItemNotFoundError,
    UniqueId,
    Version,
    md5_checksum,
)
from parlant.core.customers import CustomerId
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    DocumentDatabase,
)

# Type definitions
TestSuiteId = NewType("TestSuiteId", str)
TestScenarioId = NewType("TestScenarioId", str)
TestRunId = NewType("TestRunId", str)


class TestRunStatus(str, Enum):
    """Status of a test run execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TestStepStatus(str, Enum):
    """Status of a single test step."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class TestStep:
    """A single step in a test scenario.

    Attributes:
        role: Either "customer" (sends message) or "agent" (expects response).
        content: The message content. For customer, what to send.
                 For agent, the reference response used in subsequent history.
        should: Assertion condition for agent steps. Formatted as
                "The message should {should}" during evaluation.
                If None, step is just history without assertion.
        should_weight: Weight for scoring when multiple conditions (default 1.0).
    """

    role: Literal["customer", "agent"]
    content: str
    should: Optional[str] = None
    should_weight: float = 1.0


@dataclass(frozen=True)
class TestScenario:
    """A single test scenario definition.

    A scenario defines a sequence of conversation steps between
    a customer and an agent, with assertions on agent responses.

    Attributes:
        id: Unique identifier for this scenario.
        suite_id: The test suite this scenario belongs to.
        name: Human-readable name for the scenario.
        description: Description of what this scenario tests.
        steps: Sequence of conversation steps.
        customer_id: Customer to use for test. None = guest customer.
        repetitions: Number of times to run this scenario (default 1).
        creation_utc: When this scenario was created.
    """

    id: TestScenarioId
    suite_id: TestSuiteId
    name: str
    description: str
    steps: Sequence[TestStep]
    customer_id: Optional[CustomerId]
    repetitions: int
    creation_utc: datetime


@dataclass(frozen=True)
class TestSuite:
    """Collection of test scenarios for an agent.

    Attributes:
        id: Unique identifier for this suite.
        agent_id: The agent this suite tests.
        name: Human-readable name for the suite.
        description: Description of what this suite tests.
        creation_utc: When this suite was created.
    """

    id: TestSuiteId
    agent_id: AgentId
    name: str
    description: str
    creation_utc: datetime


@dataclass(frozen=True)
class ToolCallRecord:
    """Record of a tool call made during test execution.

    Attributes:
        tool_id: Full tool identifier (service:tool_name).
        tool_name: Short tool name.
        arguments: Arguments passed to the tool.
        result: Result returned by the tool.
    """

    tool_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    result: Any


@dataclass(frozen=True)
class TestStepResult:
    """Result of executing a single test step.

    Attributes:
        step_index: Index of the step in the scenario.
        role: Role that performed this step ("customer" or "agent").
        content: The original step content.
        actual_response: For agent steps, the actual agent response.
        tool_calls: List of tool calls made during this step.
        assertion: The assertion condition that was evaluated.
        assertion_passed: Whether the assertion passed.
        assertion_reasoning: NLP reasoning for pass/fail.
        assertion_score: Normalized score from 0-100.
    """

    step_index: int
    role: str
    content: str
    actual_response: Optional[str] = None
    tool_calls: Optional[Sequence[ToolCallRecord]] = None
    assertion: Optional[str] = None
    assertion_passed: Optional[bool] = None
    assertion_reasoning: Optional[str] = None
    assertion_score: Optional[float] = None


@dataclass(frozen=True)
class TestScenarioResult:
    """Result of running a single scenario.

    Attributes:
        scenario_id: ID of the scenario that was run.
        scenario_name: Name of the scenario.
        status: Overall status of the scenario run.
        duration_ms: Time taken to run in milliseconds.
        step_results: Results for each step.
        error: Error message if the scenario failed with an error.
        repetition: Which repetition this is (1-indexed).
    """

    scenario_id: TestScenarioId
    scenario_name: str
    status: TestStepStatus
    duration_ms: float
    step_results: Sequence[TestStepResult]
    error: Optional[str] = None
    repetition: int = 1


@dataclass(frozen=True)
class TestRun:
    """Record of a test suite execution.

    Attributes:
        id: Unique identifier for this run.
        suite_id: The suite that was run.
        agent_id: The agent that was tested.
        status: Current status of the run.
        creation_utc: When the run started.
        completion_utc: When the run finished (if completed).
        total: Total number of scenarios/repetitions.
        passed: Number that passed.
        failed: Number that failed.
        errors: Number that errored.
        duration_ms: Total time taken in milliseconds.
        scenario_results: Results for each scenario execution.
    """

    id: TestRunId
    suite_id: TestSuiteId
    agent_id: AgentId
    status: TestRunStatus
    creation_utc: datetime
    completion_utc: Optional[datetime]
    total: int
    passed: int
    failed: int
    errors: int
    duration_ms: float
    scenario_results: Sequence[TestScenarioResult]


# Update parameter TypedDicts


class TestSuiteUpdateParams(TypedDict, total=False):
    """Parameters for updating a test suite."""

    name: str
    description: str


class TestScenarioUpdateParams(TypedDict, total=False):
    """Parameters for updating a test scenario."""

    name: str
    description: str
    steps: Sequence[TestStep]
    customer_id: Optional[CustomerId]
    repetitions: int


class TestRunUpdateParams(TypedDict, total=False):
    """Parameters for updating a test run."""

    status: TestRunStatus
    completion_utc: datetime
    total: int
    passed: int
    failed: int
    errors: int
    duration_ms: float
    scenario_results: Sequence[TestScenarioResult]


# Store interface


class TestSuiteStore(ABC):
    """Abstract store for test suites, scenarios, and runs."""

    # TestSuite CRUD

    @abstractmethod
    async def create_suite(
        self,
        agent_id: AgentId,
        name: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        id: Optional[TestSuiteId] = None,
    ) -> TestSuite:
        """Create a new test suite.

        Args:
            agent_id: The agent this suite tests.
            name: Human-readable name.
            description: Description of what this suite tests.
            creation_utc: Creation time (defaults to now).
            id: Optional specific ID to use.

        Returns:
            The created TestSuite.
        """
        ...

    @abstractmethod
    async def read_suite(self, suite_id: TestSuiteId) -> TestSuite:
        """Read a test suite by ID.

        Args:
            suite_id: The suite ID.

        Returns:
            The TestSuite.

        Raises:
            ItemNotFoundError: If suite doesn't exist.
        """
        ...

    @abstractmethod
    async def list_suites(
        self,
        agent_id: Optional[AgentId] = None,
    ) -> Sequence[TestSuite]:
        """List test suites, optionally filtered by agent.

        Args:
            agent_id: Filter to suites for this agent.

        Returns:
            Sequence of TestSuites.
        """
        ...

    @abstractmethod
    async def update_suite(
        self,
        suite_id: TestSuiteId,
        params: TestSuiteUpdateParams,
    ) -> TestSuite:
        """Update a test suite.

        Args:
            suite_id: The suite to update.
            params: Fields to update.

        Returns:
            The updated TestSuite.

        Raises:
            ItemNotFoundError: If suite doesn't exist.
        """
        ...

    @abstractmethod
    async def delete_suite(self, suite_id: TestSuiteId) -> None:
        """Delete a test suite and all its scenarios.

        Args:
            suite_id: The suite to delete.

        Raises:
            ItemNotFoundError: If suite doesn't exist.
        """
        ...

    # TestScenario CRUD

    @abstractmethod
    async def create_scenario(
        self,
        suite_id: TestSuiteId,
        name: str,
        description: str,
        steps: Sequence[TestStep],
        customer_id: Optional[CustomerId] = None,
        repetitions: int = 1,
        creation_utc: Optional[datetime] = None,
        id: Optional[TestScenarioId] = None,
    ) -> TestScenario:
        """Create a new test scenario.

        Args:
            suite_id: The suite this scenario belongs to.
            name: Human-readable name.
            description: Description of what this scenario tests.
            steps: The conversation steps.
            customer_id: Customer to use (None = guest).
            repetitions: Number of times to run.
            creation_utc: Creation time (defaults to now).
            id: Optional specific ID to use.

        Returns:
            The created TestScenario.

        Raises:
            ItemNotFoundError: If suite doesn't exist.
        """
        ...

    @abstractmethod
    async def read_scenario(self, scenario_id: TestScenarioId) -> TestScenario:
        """Read a test scenario by ID.

        Args:
            scenario_id: The scenario ID.

        Returns:
            The TestScenario.

        Raises:
            ItemNotFoundError: If scenario doesn't exist.
        """
        ...

    @abstractmethod
    async def list_scenarios(self, suite_id: TestSuiteId) -> Sequence[TestScenario]:
        """List all scenarios in a suite.

        Args:
            suite_id: The suite to list scenarios for.

        Returns:
            Sequence of TestScenarios.
        """
        ...

    @abstractmethod
    async def update_scenario(
        self,
        scenario_id: TestScenarioId,
        params: TestScenarioUpdateParams,
    ) -> TestScenario:
        """Update a test scenario.

        Args:
            scenario_id: The scenario to update.
            params: Fields to update.

        Returns:
            The updated TestScenario.

        Raises:
            ItemNotFoundError: If scenario doesn't exist.
        """
        ...

    @abstractmethod
    async def delete_scenario(self, scenario_id: TestScenarioId) -> None:
        """Delete a test scenario.

        Args:
            scenario_id: The scenario to delete.

        Raises:
            ItemNotFoundError: If scenario doesn't exist.
        """
        ...

    # TestRun operations

    @abstractmethod
    async def create_run(
        self,
        suite_id: TestSuiteId,
        agent_id: AgentId,
        creation_utc: Optional[datetime] = None,
        id: Optional[TestRunId] = None,
    ) -> TestRun:
        """Create a new test run record.

        Args:
            suite_id: The suite being run.
            agent_id: The agent being tested.
            creation_utc: Start time (defaults to now).
            id: Optional specific ID to use.

        Returns:
            The created TestRun with PENDING status.
        """
        ...

    @abstractmethod
    async def read_run(self, run_id: TestRunId) -> TestRun:
        """Read a test run by ID.

        Args:
            run_id: The run ID.

        Returns:
            The TestRun.

        Raises:
            ItemNotFoundError: If run doesn't exist.
        """
        ...

    @abstractmethod
    async def list_runs(
        self,
        suite_id: Optional[TestSuiteId] = None,
        limit: int = 50,
    ) -> Sequence[TestRun]:
        """List test runs, optionally filtered by suite.

        Args:
            suite_id: Filter to runs for this suite.
            limit: Maximum number of runs to return.

        Returns:
            Sequence of TestRuns, ordered by creation_utc descending.
        """
        ...

    @abstractmethod
    async def update_run(
        self,
        run_id: TestRunId,
        params: TestRunUpdateParams,
    ) -> TestRun:
        """Update a test run.

        Args:
            run_id: The run to update.
            params: Fields to update.

        Returns:
            The updated TestRun.

        Raises:
            ItemNotFoundError: If run doesn't exist.
        """
        ...

    @abstractmethod
    async def delete_run(self, run_id: TestRunId) -> None:
        """Delete a test run.

        Args:
            run_id: The run to delete.

        Raises:
            ItemNotFoundError: If run doesn't exist.
        """
        ...

    @abstractmethod
    async def delete_runs(self, suite_id: Optional[TestSuiteId] = None) -> int:
        """Delete test runs, optionally filtered by suite.

        Args:
            suite_id: If provided, only delete runs for this suite.
                      If None, delete all runs.

        Returns:
            Number of runs deleted.
        """
        ...


# Document TypedDicts for persistence


class TestStepDocument(TypedDict):
    """Document format for a test step."""

    role: str
    content: str
    should: Optional[str]
    should_weight: float


class TestSuiteDocument(TypedDict, total=False):
    """Document format for a test suite."""

    id: ObjectId
    version: Version.String
    creation_utc: str
    agent_id: str
    name: str
    description: str


class TestScenarioDocument(TypedDict, total=False):
    """Document format for a test scenario."""

    id: ObjectId
    version: Version.String
    creation_utc: str
    suite_id: str
    name: str
    description: str
    steps: Sequence[TestStepDocument]
    customer_id: Optional[str]
    repetitions: int


class ToolCallRecordDocument(TypedDict):
    """Document format for a tool call record."""

    tool_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    result: Any


class TestStepResultDocument(TypedDict):
    """Document format for a test step result."""

    step_index: int
    role: str
    content: str
    actual_response: Optional[str]
    tool_calls: Optional[Sequence[ToolCallRecordDocument]]
    assertion: Optional[str]
    assertion_passed: Optional[bool]
    assertion_reasoning: Optional[str]
    assertion_score: Optional[float]


class TestScenarioResultDocument(TypedDict):
    """Document format for a test scenario result."""

    scenario_id: str
    scenario_name: str
    status: str
    duration_ms: float
    step_results: Sequence[TestStepResultDocument]
    error: Optional[str]
    repetition: int


class TestRunDocument(TypedDict, total=False):
    """Document format for a test run."""

    id: ObjectId
    version: Version.String
    creation_utc: str
    completion_utc: Optional[str]
    suite_id: str
    agent_id: str
    status: str
    total: int
    passed: int
    failed: int
    errors: int
    duration_ms: float
    scenario_results: Sequence[TestScenarioResultDocument]


class TestSuiteDocumentStore(TestSuiteStore):
    """Document store implementation for test suites, scenarios, and runs."""

    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
    ) -> None:
        self._id_generator = id_generator
        self._database = database
        self._allow_migration = allow_migration
        self._suite_collection: DocumentCollection[TestSuiteDocument]
        self._scenario_collection: DocumentCollection[TestScenarioDocument]
        self._run_collection: DocumentCollection[TestRunDocument]
        self._lock = ReaderWriterLock()

    async def _suite_document_loader(self, doc: BaseDocument) -> Optional[TestSuiteDocument]:
        """Load and migrate suite documents."""
        if doc["version"] == "0.1.0":
            return cast(TestSuiteDocument, doc)
        return None

    async def _scenario_document_loader(self, doc: BaseDocument) -> Optional[TestScenarioDocument]:
        """Load and migrate scenario documents."""
        if doc["version"] == "0.1.0":
            return cast(TestScenarioDocument, doc)
        return None

    async def _run_document_loader(self, doc: BaseDocument) -> Optional[TestRunDocument]:
        """Load and migrate run documents."""
        if doc["version"] == "0.1.0":
            return cast(TestRunDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        self._suite_collection = await self._database.get_or_create_collection(
            name="test_suites",
            schema=TestSuiteDocument,
            document_loader=self._suite_document_loader,
        )

        self._scenario_collection = await self._database.get_or_create_collection(
            name="test_scenarios",
            schema=TestScenarioDocument,
            document_loader=self._scenario_document_loader,
        )

        self._run_collection = await self._database.get_or_create_collection(
            name="test_runs",
            schema=TestRunDocument,
            document_loader=self._run_document_loader,
        )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    # Serialization helpers

    def _serialize_step(self, step: TestStep) -> TestStepDocument:
        return TestStepDocument(
            role=step.role,
            content=step.content,
            should=step.should,
            should_weight=step.should_weight,
        )

    def _deserialize_step(self, doc: TestStepDocument) -> TestStep:
        return TestStep(
            role=cast(Literal["customer", "agent"], doc["role"]),
            content=doc["content"],
            should=doc.get("should"),
            should_weight=doc.get("should_weight", 1.0),
        )

    def _serialize_tool_call(self, tc: ToolCallRecord) -> ToolCallRecordDocument:
        return ToolCallRecordDocument(
            tool_id=tc.tool_id,
            tool_name=tc.tool_name,
            arguments=dict(tc.arguments),
            result=tc.result,
        )

    def _deserialize_tool_call(self, doc: ToolCallRecordDocument) -> ToolCallRecord:
        return ToolCallRecord(
            tool_id=doc["tool_id"],
            tool_name=doc["tool_name"],
            arguments=doc["arguments"],
            result=doc["result"],
        )

    def _serialize_step_result(self, result: TestStepResult) -> TestStepResultDocument:
        return TestStepResultDocument(
            step_index=result.step_index,
            role=result.role,
            content=result.content,
            actual_response=result.actual_response,
            tool_calls=[self._serialize_tool_call(tc) for tc in result.tool_calls]
            if result.tool_calls
            else None,
            assertion=result.assertion,
            assertion_passed=result.assertion_passed,
            assertion_reasoning=result.assertion_reasoning,
            assertion_score=result.assertion_score,
        )

    def _deserialize_step_result(self, doc: TestStepResultDocument) -> TestStepResult:
        tool_calls_doc = doc.get("tool_calls")
        return TestStepResult(
            step_index=doc["step_index"],
            role=doc["role"],
            content=doc["content"],
            actual_response=doc.get("actual_response"),
            tool_calls=[self._deserialize_tool_call(tc) for tc in tool_calls_doc]
            if tool_calls_doc
            else None,
            assertion=doc.get("assertion"),
            assertion_passed=doc.get("assertion_passed"),
            assertion_reasoning=doc.get("assertion_reasoning"),
            assertion_score=doc.get("assertion_score"),
        )

    def _serialize_scenario_result(self, result: TestScenarioResult) -> TestScenarioResultDocument:
        return TestScenarioResultDocument(
            scenario_id=result.scenario_id,
            scenario_name=result.scenario_name,
            status=result.status.value,
            duration_ms=result.duration_ms,
            step_results=[self._serialize_step_result(sr) for sr in result.step_results],
            error=result.error,
            repetition=result.repetition,
        )

    def _deserialize_scenario_result(self, doc: TestScenarioResultDocument) -> TestScenarioResult:
        return TestScenarioResult(
            scenario_id=TestScenarioId(doc["scenario_id"]),
            scenario_name=doc["scenario_name"],
            status=TestStepStatus(doc["status"]),
            duration_ms=doc["duration_ms"],
            step_results=[self._deserialize_step_result(sr) for sr in doc["step_results"]],
            error=doc.get("error"),
            repetition=doc.get("repetition", 1),
        )

    def _serialize_suite(self, suite: TestSuite) -> TestSuiteDocument:
        return TestSuiteDocument(
            id=ObjectId(suite.id),
            version=self.VERSION.to_string(),
            creation_utc=suite.creation_utc.isoformat(),
            agent_id=suite.agent_id,
            name=suite.name,
            description=suite.description,
        )

    def _deserialize_suite(self, doc: TestSuiteDocument) -> TestSuite:
        return TestSuite(
            id=TestSuiteId(doc["id"]),
            agent_id=AgentId(doc["agent_id"]),
            name=doc["name"],
            description=doc["description"],
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
        )

    def _serialize_scenario(self, scenario: TestScenario) -> TestScenarioDocument:
        return TestScenarioDocument(
            id=ObjectId(scenario.id),
            version=self.VERSION.to_string(),
            creation_utc=scenario.creation_utc.isoformat(),
            suite_id=scenario.suite_id,
            name=scenario.name,
            description=scenario.description,
            steps=[self._serialize_step(s) for s in scenario.steps],
            customer_id=scenario.customer_id,
            repetitions=scenario.repetitions,
        )

    def _deserialize_scenario(self, doc: TestScenarioDocument) -> TestScenario:
        customer_id_str = doc.get("customer_id")
        return TestScenario(
            id=TestScenarioId(doc["id"]),
            suite_id=TestSuiteId(doc["suite_id"]),
            name=doc["name"],
            description=doc["description"],
            steps=[self._deserialize_step(s) for s in doc["steps"]],
            customer_id=CustomerId(customer_id_str) if customer_id_str else None,
            repetitions=doc.get("repetitions", 1),
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
        )

    def _serialize_run(self, run: TestRun) -> TestRunDocument:
        return TestRunDocument(
            id=ObjectId(run.id),
            version=self.VERSION.to_string(),
            creation_utc=run.creation_utc.isoformat(),
            completion_utc=run.completion_utc.isoformat() if run.completion_utc else None,
            suite_id=run.suite_id,
            agent_id=run.agent_id,
            status=run.status.value,
            total=run.total,
            passed=run.passed,
            failed=run.failed,
            errors=run.errors,
            duration_ms=run.duration_ms,
            scenario_results=[self._serialize_scenario_result(sr) for sr in run.scenario_results],
        )

    def _deserialize_run(self, doc: TestRunDocument) -> TestRun:
        completion_utc_str = doc.get("completion_utc")
        return TestRun(
            id=TestRunId(doc["id"]),
            suite_id=TestSuiteId(doc["suite_id"]),
            agent_id=AgentId(doc["agent_id"]),
            status=TestRunStatus(doc["status"]),
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
            completion_utc=(
                datetime.fromisoformat(completion_utc_str) if completion_utc_str else None
            ),
            total=doc.get("total", 0),
            passed=doc.get("passed", 0),
            failed=doc.get("failed", 0),
            errors=doc.get("errors", 0),
            duration_ms=doc.get("duration_ms", 0.0),
            scenario_results=[
                self._deserialize_scenario_result(sr) for sr in doc.get("scenario_results", [])
            ],
        )

    # TestSuite CRUD

    @override
    async def create_suite(
        self,
        agent_id: AgentId,
        name: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        id: Optional[TestSuiteId] = None,
    ) -> TestSuite:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)
            suite_checksum = md5_checksum(f"{agent_id}{name}{description}")
            suite_id = id or TestSuiteId(self._id_generator.generate(suite_checksum))

            suite = TestSuite(
                id=suite_id,
                agent_id=agent_id,
                name=name,
                description=description,
                creation_utc=creation_utc,
            )

            await self._suite_collection.insert_one(document=self._serialize_suite(suite))

        return suite

    @override
    async def read_suite(self, suite_id: TestSuiteId) -> TestSuite:
        async with self._lock.reader_lock:
            doc = await self._suite_collection.find_one(filters={"id": {"$eq": suite_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(suite_id))

        return self._deserialize_suite(doc)

    @override
    async def list_suites(
        self,
        agent_id: Optional[AgentId] = None,
    ) -> Sequence[TestSuite]:
        async with self._lock.reader_lock:
            filters: Where = {}
            if agent_id is not None:
                filters = {"agent_id": {"$eq": str(agent_id)}}

            docs = await self._suite_collection.find(filters=filters)

        return [self._deserialize_suite(doc) for doc in docs]

    @override
    async def update_suite(
        self,
        suite_id: TestSuiteId,
        params: TestSuiteUpdateParams,
    ) -> TestSuite:
        async with self._lock.writer_lock:
            update_doc = TestSuiteDocument(
                {
                    **({"name": params["name"]} if "name" in params else {}),
                    **({"description": params["description"]} if "description" in params else {}),
                }
            )

            result = await self._suite_collection.update_one(
                filters={"id": {"$eq": suite_id}},
                params=update_doc,
            )

        if not result.updated_document:
            raise ItemNotFoundError(item_id=UniqueId(suite_id))

        return self._deserialize_suite(result.updated_document)

    @override
    async def delete_suite(self, suite_id: TestSuiteId) -> None:
        async with self._lock.writer_lock:
            # Delete all scenarios in this suite
            scenarios = await self._scenario_collection.find(
                filters={"suite_id": {"$eq": suite_id}}
            )
            for scenario in scenarios:
                await self._scenario_collection.delete_one(filters={"id": {"$eq": scenario["id"]}})

            # Delete the suite
            result = await self._suite_collection.delete_one(filters={"id": {"$eq": suite_id}})

        if not result.deleted_document:
            raise ItemNotFoundError(item_id=UniqueId(suite_id))

    # TestScenario CRUD

    @override
    async def create_scenario(
        self,
        suite_id: TestSuiteId,
        name: str,
        description: str,
        steps: Sequence[TestStep],
        customer_id: Optional[CustomerId] = None,
        repetitions: int = 1,
        creation_utc: Optional[datetime] = None,
        id: Optional[TestScenarioId] = None,
    ) -> TestScenario:
        # Verify suite exists
        await self.read_suite(suite_id)

        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)
            steps_str = str([(s.role, s.content, s.should) for s in steps])
            scenario_checksum = md5_checksum(f"{suite_id}{name}{description}{steps_str}")
            scenario_id = id or TestScenarioId(self._id_generator.generate(scenario_checksum))

            scenario = TestScenario(
                id=scenario_id,
                suite_id=suite_id,
                name=name,
                description=description,
                steps=steps,
                customer_id=customer_id,
                repetitions=repetitions,
                creation_utc=creation_utc,
            )

            await self._scenario_collection.insert_one(document=self._serialize_scenario(scenario))

        return scenario

    @override
    async def read_scenario(self, scenario_id: TestScenarioId) -> TestScenario:
        async with self._lock.reader_lock:
            doc = await self._scenario_collection.find_one(filters={"id": {"$eq": scenario_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(scenario_id))

        return self._deserialize_scenario(doc)

    @override
    async def list_scenarios(self, suite_id: TestSuiteId) -> Sequence[TestScenario]:
        async with self._lock.reader_lock:
            docs = await self._scenario_collection.find(filters={"suite_id": {"$eq": suite_id}})

        return [self._deserialize_scenario(doc) for doc in docs]

    @override
    async def update_scenario(
        self,
        scenario_id: TestScenarioId,
        params: TestScenarioUpdateParams,
    ) -> TestScenario:
        async with self._lock.writer_lock:
            update_doc = TestScenarioDocument(
                {
                    **({"name": params["name"]} if "name" in params else {}),
                    **({"description": params["description"]} if "description" in params else {}),
                    **(
                        {"steps": [self._serialize_step(s) for s in params["steps"]]}
                        if "steps" in params
                        else {}
                    ),
                    **({"customer_id": params["customer_id"]} if "customer_id" in params else {}),
                    **({"repetitions": params["repetitions"]} if "repetitions" in params else {}),
                }
            )

            result = await self._scenario_collection.update_one(
                filters={"id": {"$eq": scenario_id}},
                params=update_doc,
            )

        if not result.updated_document:
            raise ItemNotFoundError(item_id=UniqueId(scenario_id))

        return self._deserialize_scenario(result.updated_document)

    @override
    async def delete_scenario(self, scenario_id: TestScenarioId) -> None:
        async with self._lock.writer_lock:
            result = await self._scenario_collection.delete_one(
                filters={"id": {"$eq": scenario_id}}
            )

        if not result.deleted_document:
            raise ItemNotFoundError(item_id=UniqueId(scenario_id))

    # TestRun operations

    @override
    async def create_run(
        self,
        suite_id: TestSuiteId,
        agent_id: AgentId,
        creation_utc: Optional[datetime] = None,
        id: Optional[TestRunId] = None,
    ) -> TestRun:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)
            run_checksum = md5_checksum(f"{suite_id}{agent_id}{creation_utc.isoformat()}")
            run_id = id or TestRunId(self._id_generator.generate(run_checksum))

            run = TestRun(
                id=run_id,
                suite_id=suite_id,
                agent_id=agent_id,
                status=TestRunStatus.PENDING,
                creation_utc=creation_utc,
                completion_utc=None,
                total=0,
                passed=0,
                failed=0,
                errors=0,
                duration_ms=0.0,
                scenario_results=[],
            )

            await self._run_collection.insert_one(document=self._serialize_run(run))

        return run

    @override
    async def read_run(self, run_id: TestRunId) -> TestRun:
        async with self._lock.reader_lock:
            doc = await self._run_collection.find_one(filters={"id": {"$eq": run_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(run_id))

        return self._deserialize_run(doc)

    @override
    async def list_runs(
        self,
        suite_id: Optional[TestSuiteId] = None,
        limit: int = 50,
    ) -> Sequence[TestRun]:
        async with self._lock.reader_lock:
            filters: Where = {}
            if suite_id is not None:
                filters = {"suite_id": {"$eq": str(suite_id)}}

            docs = await self._run_collection.find(filters=filters)

        # Sort by creation_utc descending and apply limit
        runs = [self._deserialize_run(doc) for doc in docs]
        runs.sort(key=lambda r: r.creation_utc, reverse=True)
        return runs[:limit]

    @override
    async def update_run(
        self,
        run_id: TestRunId,
        params: TestRunUpdateParams,
    ) -> TestRun:
        async with self._lock.writer_lock:
            update_doc = TestRunDocument(
                {
                    **({"status": params["status"].value} if "status" in params else {}),
                    **(
                        {"completion_utc": params["completion_utc"].isoformat()}
                        if "completion_utc" in params
                        else {}
                    ),
                    **({"total": params["total"]} if "total" in params else {}),
                    **({"passed": params["passed"]} if "passed" in params else {}),
                    **({"failed": params["failed"]} if "failed" in params else {}),
                    **({"errors": params["errors"]} if "errors" in params else {}),
                    **({"duration_ms": params["duration_ms"]} if "duration_ms" in params else {}),
                    **(
                        {
                            "scenario_results": [
                                self._serialize_scenario_result(sr)
                                for sr in params["scenario_results"]
                            ]
                        }
                        if "scenario_results" in params
                        else {}
                    ),
                }
            )

            result = await self._run_collection.update_one(
                filters={"id": {"$eq": run_id}},
                params=update_doc,
            )

        if not result.updated_document:
            raise ItemNotFoundError(item_id=UniqueId(run_id))

        return self._deserialize_run(result.updated_document)

    @override
    async def delete_run(self, run_id: TestRunId) -> None:
        async with self._lock.writer_lock:
            result = await self._run_collection.delete_one(
                filters={"id": {"$eq": run_id}},
            )

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(run_id))

    @override
    async def delete_runs(self, suite_id: Optional[TestSuiteId] = None) -> int:
        # First get all runs to delete
        filters: dict[str, Any] = {}
        if suite_id:
            filters["suite_id"] = {"$eq": suite_id}

        docs = await self._run_collection.find(filters=filters)

        # Delete each one
        deleted_count = 0
        for doc in docs:
            run_id = TestRunId(doc["id"])
            result = await self._run_collection.delete_one(
                filters={"id": {"$eq": run_id}},
            )
            deleted_count += result.deleted_count

        return deleted_count
