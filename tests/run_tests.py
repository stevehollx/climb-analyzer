#!/usr/bin/env python3
"""
Main test orchestrator for climb analyzer test suite.

Runs 19 test scenarios covering CLI, interactive, batch, and address-based modes.
Supports resuming from any test number and validates output against Strava data.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --start 5          # Start from test 5
    python tests/run_tests.py --test 3,7,12      # Run specific tests
    python tests/run_tests.py --skip 8,9,10      # Skip specific tests
    python tests/run_tests.py --early-exit-only  # Only run early exit tests
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pexpect
    HAS_PEXPECT = True
except ImportError:
    HAS_PEXPECT = False
    print("Warning: pexpect not installed. Interactive tests will be skipped.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Output validation will be limited.")

from test_colors import (
    Colors, print_pass, print_fail, print_skip, print_info,
    print_warn, print_error, print_running, box_header, light_box,
    divider, format_duration, green, red, yellow, cyan, bold
)
from interactive_sequences import (
    get_sequence, get_cli_command, is_interactive_test,
    is_early_exit_test, INTERACTIVE_SEQUENCES, CLI_COMMANDS
)
from result_validator import ResultValidator, ClimbMatch
from strava_scraper import validate_against_strava, StravaScraper


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_id: int
    name: str
    status: str  # 'pass', 'fail', 'skip', 'error'
    duration: float = 0.0
    climbs_found: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    strava_validation: Optional[Dict[str, Any]] = None
    output_file: Optional[Path] = None


class TestRunner:
    """Main test runner for climb analyzer test suite."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: List[TestResult] = []
        self.start_time = time.time()

        # Load test definitions
        test_file = Path(__file__).parent / 'validation_climbs.json'
        with open(test_file) as f:
            self.test_definitions = json.load(f)

        # Setup validator
        self.validator = ResultValidator(output_dir='output')

        # Progress update interval (seconds)
        self.progress_interval = 300  # 5 minutes

        # Disable colors if requested
        if args.no_color or not sys.stdout.isatty():
            Colors.disable()

    def get_tests_to_run(self) -> List[int]:
        """Determine which tests to run based on arguments."""
        all_tests = list(range(1, 20))  # Tests 1-19

        # Filter by early-exit-only
        if self.args.early_exit_only:
            all_tests = [t for t in all_tests if is_early_exit_test(t)]

        # Filter by --test (specific tests)
        if self.args.test:
            specified = [int(t.strip()) for t in self.args.test.split(',')]
            all_tests = [t for t in specified if t in all_tests]

        # Filter by --start (start from test N)
        if self.args.start:
            all_tests = [t for t in all_tests if t >= self.args.start]

        # Filter by --skip (skip specific tests)
        if self.args.skip:
            skip_tests = [int(t.strip()) for t in self.args.skip.split(',')]
            all_tests = [t for t in all_tests if t not in skip_tests]

        return all_tests

    def run_all(self) -> int:
        """Run all selected tests and return exit code."""
        tests_to_run = self.get_tests_to_run()

        if not tests_to_run:
            print_error("No tests to run with current filters")
            return 1

        # Print header
        print(box_header("CLIMB ANALYZER TEST SUITE"))
        print()
        print_info(f"Running {len(tests_to_run)} tests: {tests_to_run}")
        print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(divider())
        print()

        # Run each test
        for test_id in tests_to_run:
            result = self._run_test(test_id)
            self.results.append(result)

            # Print result
            self._print_test_result(result)
            print()

            # Early termination on critical error
            if result.status == 'error' and not self.args.continue_on_error:
                print_error(f"Critical error in test {test_id}, stopping")
                break

        # Print summary
        self._print_summary()

        # Return exit code
        failed = sum(1 for r in self.results if r.status in ['fail', 'error'])
        return 1 if failed > 0 else 0

    def _run_test(self, test_id: int) -> TestResult:
        """Run a single test by ID."""
        test_def = self.test_definitions.get(f'test_{test_id}', {})
        test_name = test_def.get('test_name', f'Test {test_id}')

        print_running(f"Test {test_id}: {test_name}")

        start_time = time.time()

        # Determine test type
        if is_interactive_test(test_id):
            result = self._run_interactive_test(test_id, test_def)
        else:
            result = self._run_cli_test(test_id, test_def)

        result.duration = time.time() - start_time
        return result

    def _run_cli_test(self, test_id: int, test_def: Dict) -> TestResult:
        """Run a CLI-based test."""
        test_name = test_def.get('test_name', f'Test {test_id}')
        result = TestResult(test_id=test_id, name=test_name, status='pass')

        cmd = get_cli_command(test_id)
        if not cmd:
            result.status = 'error'
            result.errors.append(f"No CLI command defined for test {test_id}")
            return result

        # Check if early exit test
        if is_early_exit_test(test_id):
            return self._run_early_exit_cli(test_id, test_def, cmd)

        # Full analysis run
        return self._run_full_cli(test_id, test_def, cmd)

    def _run_full_cli(self, test_id: int, test_def: Dict, cmd: List[str]) -> TestResult:
        """Run a full CLI analysis (no timeout)."""
        test_name = test_def.get('test_name', f'Test {test_id}')
        result = TestResult(test_id=test_id, name=test_name, status='pass')

        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Monitor output with progress updates
            output_lines = []
            last_progress = time.time()

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                output_lines.append(line)

                # Progress update
                if time.time() - last_progress >= self.progress_interval:
                    elapsed = time.time() - self.start_time
                    print_info(f"  Still running... ({format_duration(elapsed)} elapsed)")
                    last_progress = time.time()

                # Check for error patterns
                if 'error' in line.lower() or 'exception' in line.lower():
                    if self.args.verbose:
                        print(f"  {Colors.YELLOW}{line.strip()}{Colors.END}")

            # Check exit code
            if process.returncode != 0:
                result.status = 'fail'
                result.errors.append(f"Process exited with code {process.returncode}")

            # Validate output
            if result.status == 'pass':
                self._validate_output(test_id, test_def, result)

        except Exception as e:
            result.status = 'error'
            result.errors.append(str(e))

        return result

    def _run_early_exit_cli(
        self, test_id: int, test_def: Dict, cmd: List[str]
    ) -> TestResult:
        """Run a CLI test that exits early after region confirmation."""
        test_name = test_def.get('test_name', f'Test {test_id}')
        result = TestResult(test_id=test_id, name=test_name, status='pass')

        # Patterns to look for
        region_patterns = [
            r'(?i)selected.*region',
            r'(?i)using.*region',
            r'(?i)processing.*region',
            r'(?i)downloading.*osm',
            r'(?i)region:',
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            region_confirmed = False
            timeout = 120  # 2 minute timeout for early exit tests

            start = time.time()
            while time.time() - start < timeout:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue

                output_lines.append(line)

                # Check for region confirmation
                for pattern in region_patterns:
                    if re.search(pattern, line):
                        region_confirmed = True
                        if self.args.verbose:
                            print_info(f"  Region confirmed: {line.strip()}")

                        # Check expected selection if defined
                        expected = test_def.get('expected_selection')
                        if expected:
                            if isinstance(expected, list):
                                # Multiple regions expected
                                for exp in expected:
                                    if exp.lower() not in line.lower():
                                        result.warnings.append(
                                            f"Expected region '{exp}' not in: {line.strip()}"
                                        )
                            else:
                                if expected.lower() not in line.lower():
                                    result.warnings.append(
                                        f"Expected '{expected}' not in: {line.strip()}"
                                    )

                        # Send SIGINT to stop
                        process.send_signal(signal.SIGINT)
                        time.sleep(0.5)
                        break

                if region_confirmed:
                    break

            # Cleanup
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

            if not region_confirmed:
                result.status = 'fail'
                result.errors.append("Region confirmation not detected within timeout")

        except Exception as e:
            result.status = 'error'
            result.errors.append(str(e))

        return result

    def _run_interactive_test(self, test_id: int, test_def: Dict) -> TestResult:
        """Run an interactive (pexpect) test."""
        test_name = test_def.get('test_name', f'Test {test_id}')
        result = TestResult(test_id=test_id, name=test_name, status='pass')

        if not HAS_PEXPECT:
            result.status = 'skip'
            result.warnings.append("pexpect not installed")
            return result

        sequence = get_sequence(test_id)
        if not sequence:
            result.status = 'error'
            result.errors.append(f"No interactive sequence for test {test_id}")
            return result

        try:
            # Start interactive session
            child = pexpect.spawn('./climb-analyzer', encoding='utf-8', timeout=30)

            # Execute each step
            for step in sequence.steps:
                try:
                    child.expect(step.expect_pattern, timeout=step.timeout)
                    if self.args.verbose:
                        print_info(f"  {step.description}")
                    child.sendline(step.send_response)
                except pexpect.TIMEOUT:
                    result.status = 'fail'
                    result.errors.append(f"Timeout waiting for: {step.description}")
                    child.terminate()
                    return result
                except pexpect.EOF:
                    result.status = 'fail'
                    result.errors.append(f"Unexpected EOF at: {step.description}")
                    return result

            # Handle early exit vs full run
            if sequence.early_exit:
                # Wait for region confirmation then exit
                time.sleep(2)
                child.sendcontrol('c')
                child.close()

                # Check for expected pattern
                if sequence.expected_region_pattern:
                    output = child.before + child.after if child.after else child.before
                    if not re.search(sequence.expected_region_pattern, output or ''):
                        result.warnings.append("Expected region pattern not found")
            else:
                # Wait for full completion (no timeout)
                last_progress = time.time()
                while child.isalive():
                    try:
                        child.expect(pexpect.TIMEOUT, timeout=self.progress_interval)
                    except pexpect.TIMEOUT:
                        elapsed = time.time() - self.start_time
                        print_info(f"  Still running... ({format_duration(elapsed)} elapsed)")
                    except pexpect.EOF:
                        break

                # Validate output
                if result.status == 'pass':
                    self._validate_output(test_id, test_def, result)

        except Exception as e:
            result.status = 'error'
            result.errors.append(str(e))

        return result

    def _validate_output(self, test_id: int, test_def: Dict, result: TestResult):
        """Validate output file and specific climbs."""
        if not HAS_PANDAS:
            result.warnings.append("pandas not installed, skipping output validation")
            return

        region = test_def.get('region', '')
        if region in ['address', 'custom_bbox']:
            # Special handling for non-region tests
            region = test_def.get('test_name', '').split('-')[0].strip()

        # Find output file
        output_file = self.validator.find_output_file(region)
        if not output_file:
            result.status = 'fail'
            result.errors.append(f"No output file found for region: {region}")
            return

        result.output_file = output_file

        # Validate file structure
        validation = self.validator.validate_file(output_file)
        result.climbs_found = validation.total_climbs

        if not validation.valid:
            result.status = 'fail'
            result.errors.extend([e.message for e in validation.errors])
        result.warnings.extend([w.message for w in validation.warnings])

        # Validate specific climbs
        climbs_to_check = test_def.get('climbs', [])
        df = pd.read_excel(output_file)

        for climb_def in climbs_to_check:
            climb_name = climb_def.get('name', '')
            alt_names = climb_def.get('alt_names', [])

            # Check for cross-region climbs
            if climb_def.get('cross_region_check'):
                expected_regions = climb_def.get('expected_regions', [])
                match, regions_ok = self.validator.find_cross_region_climb(
                    df, climb_name, expected_regions, alt_names
                )
                if not match.found:
                    result.status = 'fail'
                    result.errors.append(f"Cross-region climb not found: {climb_name}")
                elif not regions_ok:
                    result.warnings.append(
                        f"Climb {climb_name} found but regions not verified"
                    )
            else:
                match = self.validator.find_climb(df, climb_name, alt_names)
                if not match.found:
                    result.status = 'fail'
                    result.errors.append(f"Expected climb not found: {climb_name}")

            # Strava validation if segment ID provided
            strava_id = climb_def.get('strava_segment_id')
            if strava_id and match.found:
                self._validate_strava(match, strava_id, result)

    def _validate_strava(
        self, match: ClimbMatch, strava_id: str, result: TestResult
    ):
        """Validate a climb against Strava segment data."""
        try:
            metrics = self.validator.get_climb_metrics(match)
            tolerance = self.test_definitions.get('_metadata', {}).get(
                'tolerance_percent', 15
            )

            strava_result = validate_against_strava(
                metrics, strava_id, tolerance_percent=tolerance
            )

            result.strava_validation = strava_result

            if strava_result.get('error'):
                result.warnings.append(f"Strava validation failed: {strava_result['error']}")
            elif not strava_result.get('valid'):
                result.warnings.append(
                    f"Strava comparison outside {tolerance}% tolerance"
                )
                for metric, data in strava_result.get('comparisons', {}).items():
                    if not data.get('within_tolerance'):
                        result.warnings.append(
                            f"  {metric}: {data['diff_percent']:.1f}% diff"
                        )

        except Exception as e:
            result.warnings.append(f"Strava validation error: {e}")

    def _print_test_result(self, result: TestResult):
        """Print the result of a single test."""
        duration_str = format_duration(result.duration)

        if result.status == 'pass':
            print_pass(f"Test {result.test_id}: {result.name} ({duration_str})")
            if result.climbs_found > 0:
                print(f"       {result.climbs_found} climbs found")
        elif result.status == 'fail':
            print_fail(f"Test {result.test_id}: {result.name} ({duration_str})")
            for error in result.errors[:5]:
                print(f"       {Colors.RED}{error}{Colors.END}")
        elif result.status == 'skip':
            print_skip(f"Test {result.test_id}: {result.name}")
            for warn in result.warnings:
                print(f"       {warn}")
        else:  # error
            print_error(f"Test {result.test_id}: {result.name}")
            for error in result.errors:
                print(f"       {Colors.RED}{error}{Colors.END}")

        # Print warnings
        if result.warnings and result.status == 'pass':
            for warn in result.warnings[:3]:
                print(f"       {Colors.YELLOW}{warn}{Colors.END}")

    def _print_summary(self):
        """Print test suite summary."""
        total_duration = time.time() - self.start_time

        passed = sum(1 for r in self.results if r.status == 'pass')
        failed = sum(1 for r in self.results if r.status == 'fail')
        skipped = sum(1 for r in self.results if r.status == 'skip')
        errors = sum(1 for r in self.results if r.status == 'error')

        print()
        print(divider('='))
        print(box_header("TEST SUMMARY"))
        print()

        summary_lines = [
            f"  Total tests: {len(self.results)}",
            f"  Duration: {format_duration(total_duration)}",
            "---",
            f"  {Colors.GREEN}Passed:  {passed}{Colors.END}",
            f"  {Colors.RED}Failed:  {failed}{Colors.END}",
            f"  {Colors.YELLOW}Skipped: {skipped}{Colors.END}",
            f"  {Colors.RED}Errors:  {errors}{Colors.END}",
        ]

        print(light_box(summary_lines))
        print()

        # List failed tests
        if failed > 0 or errors > 0:
            print(f"{Colors.RED}Failed/Error tests:{Colors.END}")
            for r in self.results:
                if r.status in ['fail', 'error']:
                    print(f"  - Test {r.test_id}: {r.name}")
                    for err in r.errors[:2]:
                        print(f"      {err}")
            print()

        # Overall status
        if failed == 0 and errors == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED{Colors.END}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}TESTS FAILED{Colors.END}")


def main():
    parser = argparse.ArgumentParser(
        description='Run climb analyzer test suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py                    # Run all tests
  python tests/run_tests.py --start 5          # Start from test 5
  python tests/run_tests.py --test 3,7,12      # Run specific tests
  python tests/run_tests.py --skip 8,9,10      # Skip specific tests
  python tests/run_tests.py --early-exit-only  # Only early exit tests
        """
    )

    parser.add_argument(
        '--start', type=int, metavar='N',
        help='Start from test number N'
    )
    parser.add_argument(
        '--test', type=str, metavar='IDS',
        help='Run specific tests (comma-separated, e.g., "3,7,12")'
    )
    parser.add_argument(
        '--skip', type=str, metavar='IDS',
        help='Skip specific tests (comma-separated, e.g., "8,9,10")'
    )
    parser.add_argument(
        '--early-exit-only', action='store_true',
        help='Only run early exit tests (14-19)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--no-color', action='store_true',
        help='Disable colored output'
    )
    parser.add_argument(
        '--continue-on-error', action='store_true',
        help='Continue running tests after critical errors'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all tests without running them'
    )

    args = parser.parse_args()

    # List tests if requested
    if args.list:
        test_file = Path(__file__).parent / 'validation_climbs.json'
        with open(test_file) as f:
            test_defs = json.load(f)

        print("Available tests:")
        print()
        for i in range(1, 20):
            test_def = test_defs.get(f'test_{i}', {})
            name = test_def.get('test_name', f'Test {i}')
            mode = 'interactive' if is_interactive_test(i) else 'CLI'
            early = ' (early exit)' if is_early_exit_test(i) else ''
            print(f"  {i:2d}. [{mode:11s}] {name}{early}")
        return 0

    # Run tests
    runner = TestRunner(args)
    return runner.run_all()


if __name__ == '__main__':
    sys.exit(main())
