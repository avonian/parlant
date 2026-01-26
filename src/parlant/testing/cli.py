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

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import click
from rich.console import Console

from parlant.testing.discovery import DiscoveryError, discover_suites, list_tests
from parlant.testing.reporter import (
    RichReporter,
    generate_json_report,
    print_summary,
)
from parlant.testing.runner import TestReport, TestRunner


DEFAULT_HOME_DIR = Path.home() / ".parlant"


@click.command()
@click.argument("paths", nargs=-1, required=False, type=click.Path(exists=True))
@click.option(
    "--pattern",
    "-p",
    default=None,
    help="Regex pattern to filter scenario names",
)
@click.option(
    "--parallel",
    "-n",
    default=1,
    type=int,
    help="Number of tests to run concurrently",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Write JSON report to file",
)
@click.option(
    "--fail-fast",
    "-x",
    is_flag=True,
    default=False,
    help="Stop on first failure",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    default=False,
    help="Show discovered tests without running",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level (-v, -vv, -vvv)",
)
@click.option(
    "--server-url",
    "-s",
    default=None,
    help="URL of Parlant server (required for --suite-id/--scenario-id)",
)
@click.option(
    "--suite-id",
    default=None,
    help="ID of test suite to run from database (requires --server-url)",
)
@click.option(
    "--scenario-id",
    default=None,
    help="ID of single scenario to run from database (requires --server-url)",
)
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=True),
    help="Parlant data directory (default: $PARLANT_HOME or ~/.parlant)",
)
def main(
    paths: Tuple[str, ...],
    pattern: Optional[str],
    parallel: int,
    output: Optional[str],
    fail_fast: bool,
    list_only: bool,
    verbose: int,
    server_url: Optional[str],
    suite_id: Optional[str],
    scenario_id: Optional[str],
    data_dir: Optional[str],
) -> None:
    """Run Parlant agent tests.

    PATHS: One or more files or directories containing test files.

    Alternatively, use --server-url with --suite-id or --scenario-id to run
    tests stored in the database.

    Examples:

        parlant-test tests/

        parlant-test tests/test_greeting.py --pattern "greet"

        parlant-test tests/ --parallel 4 --output results.json

        parlant-test --server-url http://localhost:8800 --suite-id ts_abc123
    """
    console = Console()

    # Validate options
    database_mode = suite_id is not None or scenario_id is not None
    if database_mode:
        if not server_url:
            console.print(
                "[bold red]Error:[/bold red] --suite-id and --scenario-id require --server-url"
            )
            sys.exit(2)
        if paths:
            console.print("[bold red]Error:[/bold red] Cannot combine PATHS with --suite-id")
            sys.exit(2)
    else:
        if not paths:
            console.print(
                "[bold red]Error:[/bold red] Either PATHS or --suite-id/--scenario-id is required"
            )
            sys.exit(2)

    exit_code = 0
    try:
        if database_mode:
            # Run from database
            assert server_url is not None
            exit_code = asyncio.run(
                _run_from_database(
                    console=console,
                    server_url=server_url,
                    suite_id=suite_id,
                    scenario_id=scenario_id,
                    data_dir=data_dir,
                    pattern=pattern,
                    parallel=parallel,
                    output=output,
                    fail_fast=fail_fast,
                    list_only=list_only,
                    verbose=verbose,
                )
            )
        else:
            # Run from Python files (existing behavior)
            exit_code = _run_from_files(
                console=console,
                paths=paths,
                pattern=pattern,
                parallel=parallel,
                output=output,
                fail_fast=fail_fast,
                list_only=list_only,
                verbose=verbose,
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        exit_code = 130
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose >= 2:
            import traceback

            console.print(traceback.format_exc())
        exit_code = 1

    sys.exit(exit_code)


def _run_from_files(
    console: Console,
    paths: Tuple[str, ...],
    pattern: Optional[str],
    parallel: int,
    output: Optional[str],
    fail_fast: bool,
    list_only: bool,
    verbose: int,
) -> int:
    """Run tests from Python files (existing behavior)."""
    try:
        # Discover test suites
        console.print(f"[dim]Discovering tests in: {', '.join(paths)}[/dim]")
        suites = discover_suites(list(paths))

        if not suites:
            console.print("[yellow]No test suites found.[/yellow]")
            return 0

        # Count scenarios
        total_scenarios = sum(len(s.get_scenarios()) for s in suites)
        console.print(f"[dim]Found {len(suites)} suite(s) with {total_scenarios} scenario(s)[/dim]")

        # List mode
        if list_only:
            tests = list_tests(suites)
            if pattern:
                import re

                compiled = re.compile(pattern)
                tests = [t for t in tests if compiled.search(t)]

            console.print()
            console.print("[bold]Discovered tests:[/bold]")
            for test in tests:
                # Escape square brackets for Rich
                escaped = test.replace("[", "\\[")
                console.print(f"  {escaped}")
            console.print()
            console.print(f"Total: {len(tests)} test(s)")
            return 0

        # Create reporter and runner
        reporter = RichReporter(
            console=console,
            panel_count=parallel,
            panel_height=12 if parallel > 2 else 15,
        )

        runner = TestRunner(listener=reporter)

        # Run tests
        console.print()
        reporter.start_live()

        async def run_and_cleanup() -> TestReport:
            try:
                return await runner.run(
                    suites=suites,
                    parallel=parallel,
                    pattern=pattern,
                    fail_fast=fail_fast,
                )
            finally:
                # Delete test sessions immediately
                for suite in suites:
                    await suite.delete_queued_sessions()

        try:
            report = asyncio.run(run_and_cleanup())
        finally:
            reporter.stop_live()

        # Print summary with final score
        final_score = reporter.get_final_score()
        print_summary(console, report, final_score)

        # Write JSON output
        if output:
            json_report = generate_json_report(report)
            Path(output).write_text(json_report)
            console.print(f"[dim]JSON report written to: {output}[/dim]")

        return 0 if report.failed == 0 else 1

    except DiscoveryError as e:
        console.print(f"[bold red]Discovery error:[/bold red] {e}")
        return 2


async def _run_from_database(
    console: Console,
    server_url: str,
    suite_id: Optional[str],
    scenario_id: Optional[str],
    data_dir: Optional[str],
    pattern: Optional[str],
    parallel: int,
    output: Optional[str],
    fail_fast: bool,
    list_only: bool,
    verbose: int,
) -> int:
    """Run tests from local database."""
    from parlant.adapters.db.json_file import JSONFileDocumentDatabase
    from parlant.core.common import IdGenerator
    from parlant.core.loggers import LogLevel, Logger
    from parlant.core.test_suites import (
        TestScenario,
        TestSuite,
        TestSuiteDocumentStore,
        TestSuiteId,
        TestScenarioId,
    )

    # Simple silent logger for database operations
    class SilentLogger(Logger):
        def set_level(self, log_level: LogLevel) -> None:
            pass

        def trace(self, message: str) -> None:
            pass

        def debug(self, message: str) -> None:
            pass

        def info(self, message: str) -> None:
            pass

        def warning(self, message: str) -> None:
            pass

        def error(self, message: str) -> None:
            pass

        def critical(self, message: str) -> None:
            pass

        @contextmanager
        def scope(self, scope_id: str) -> Iterator[None]:
            yield

    # Determine data directory
    parlant_home = (
        Path(data_dir) if data_dir else Path(os.environ.get("PARLANT_HOME", DEFAULT_HOME_DIR))
    )
    db_path = parlant_home / "test_suites.json"

    if not db_path.exists():
        console.print(f"[bold red]Error:[/bold red] Database not found: {db_path}")
        console.print("[dim]Make sure PARLANT_HOME is set or use --data-dir[/dim]")
        return 2

    console.print(f"[dim]Loading tests from: {db_path}[/dim]")

    # Initialize database and store
    logger = SilentLogger()
    id_generator = IdGenerator()
    async with JSONFileDocumentDatabase(logger, db_path) as database:
        store = TestSuiteDocumentStore(
            id_generator=id_generator,
            database=database,
        )
        async with store:
            # Load suite and scenarios
            suites_to_run: List[Tuple[TestSuite, Sequence[TestScenario]]] = []

            try:
                if suite_id:
                    suite = await store.read_suite(TestSuiteId(suite_id))
                    scenarios = await store.list_scenarios(TestSuiteId(suite_id))
                    suites_to_run.append((suite, scenarios))
                    console.print(
                        f"[dim]Found suite '{suite.name}' with {len(scenarios)} scenario(s)[/dim]"
                    )
                elif scenario_id:
                    scenario = await store.read_scenario(TestScenarioId(scenario_id))
                    suite = await store.read_suite(scenario.suite_id)
                    suites_to_run.append((suite, [scenario]))
                    console.print(f"[dim]Found scenario '{scenario.name}'[/dim]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                return 2

            if not suites_to_run:
                console.print("[yellow]No test suites found.[/yellow]")
                return 0

            # List mode
            if list_only:
                console.print()
                console.print("[bold]Discovered tests:[/bold]")
                for _, scenarios in suites_to_run:
                    for scenario in scenarios:
                        name = scenario.name
                        if pattern:
                            import re

                            if not re.search(pattern, name):
                                continue
                        if scenario.repetitions > 1:
                            for rep in range(1, scenario.repetitions + 1):
                                escaped = f"{name}[rep_{rep}/{scenario.repetitions}]".replace(
                                    "[", "\\["
                                )
                                console.print(f"  {escaped}")
                        else:
                            console.print(f"  {name}")
                return 0

            # Create DatabaseSuite adapters
            from parlant.testing.database_suite import DatabaseSuite

            cli_suites: List[DatabaseSuite] = []
            for suite, scenarios in suites_to_run:
                db_suite = DatabaseSuite(
                    server_url=server_url,
                    agent_id=str(suite.agent_id),
                    scenarios=scenarios,
                )
                cli_suites.append(db_suite)

            # Create reporter and runner
            reporter = RichReporter(
                console=console,
                panel_count=parallel,
                panel_height=12 if parallel > 2 else 15,
            )

            runner = TestRunner(listener=reporter)

            # Run tests
            console.print()
            reporter.start_live()

            try:
                report = await runner.run(
                    suites=cli_suites,  # type: ignore[arg-type]
                    parallel=parallel,
                    pattern=pattern,
                    fail_fast=fail_fast,
                )
            finally:
                reporter.stop_live()

                # Delete test sessions immediately
                for db_suite in cli_suites:
                    await db_suite.delete_queued_sessions()

            # Print summary with final score
            final_score = reporter.get_final_score()
            print_summary(console, report, final_score)

            # Write JSON output
            if output:
                json_report = generate_json_report(report)
                Path(output).write_text(json_report)
                console.print(f"[dim]JSON report written to: {output}[/dim]")

            return 0 if report.failed == 0 else 1

    # Should not reach here, but satisfy mypy
    return 1


if __name__ == "__main__":
    main()
