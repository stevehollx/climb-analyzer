#!/usr/bin/env python3
"""
Interactive menu navigation sequences for pexpect automation.

Defines the expected prompts and responses for navigating the
climb analyzer interactive menu for each test case.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class MenuStep:
    """A single step in menu navigation."""
    expect_pattern: str  # Regex pattern to expect
    send_response: str   # Response to send
    timeout: int = 10    # Timeout in seconds
    description: str = ""  # Human-readable description


@dataclass
class InteractiveSequence:
    """Complete sequence for an interactive test."""
    test_id: int
    description: str
    steps: List[MenuStep]
    early_exit: bool = False  # Exit after region confirmation
    expected_region_pattern: Optional[str] = None  # Pattern to verify region selection


# Common patterns for matching menu prompts
PATTERNS = {
    'checkpoint_resume': r'(?i)select.*analysis.*resume|resume.*analysis',
    'select_mode': r'(?i)(select|choose|enter).*(mode|option|what|how|choice)',
    'select_continent': r'(?i)(select|enter).*continent',
    'select_country': r'(?i)(select|enter).*(country|number.*cancel)',
    'select_state': r'(?i)(select|enter).*(state|province|region|number.*cancel)',
    'select_subregion': r'(?i)(select|enter).*(subregion|area|city|number.*cancel)',
    'surface_filter': r'(?i)(surface|filter).*(type|option)',
    'confirm': r'(?i)(confirm|proceed|continue|start|\[y/n\]|\(y/n\))',
    'cycling_filter': r'(?i)(cycling|accessible).*filter|\(y/N\)',
    'enter_address': r'(?i)(enter|type).*(address|location)',
    'enter_radius': r'(?i)(enter|specify).*(radius|distance)',
    'downloading': r'(?i)(downloading|fetching|processing)',
    'analysis_started': r'(?i)(starting|beginning|analyzing)',
    'region_selected': r'(?i)(selected|using|region:)',
}


# Interactive sequences for each test
INTERACTIVE_SEQUENCES = {
    # Test 2: Luxembourg - Interactive
    2: InteractiveSequence(
        test_id=2,
        description="Luxembourg via interactive menu",
        steps=[
            MenuStep(
                expect_pattern=PATTERNS['checkpoint_resume'],
                send_response="0",  # Start new analysis (skip checkpoint resume)
                timeout=30,
                description="Skip checkpoint resume, start new analysis",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_mode'],
                send_response="1",  # Region-based analysis
                description="Select region analysis mode",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_continent'],
                send_response="2",  # Europe (assuming order)
                timeout=15,
                description="Select Europe continent",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_country'],
                send_response="Luxembourg",
                timeout=15,
                description="Select Luxembourg",
            ),
            MenuStep(
                expect_pattern=PATTERNS['confirm'],
                send_response="y",
                description="Confirm selection",
            ),
        ],
    ),

    # Test 4: New York - Interactive + Paved filter
    4: InteractiveSequence(
        test_id=4,
        description="New York via interactive with paved filter",
        steps=[
            MenuStep(
                expect_pattern=PATTERNS['checkpoint_resume'],
                send_response="0",  # Start new analysis (skip checkpoint resume)
                timeout=30,
                description="Skip checkpoint resume, start new analysis",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_mode'],
                send_response="1",  # Region-based analysis
                description="Select region analysis mode",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_continent'],
                send_response="4",  # North America
                timeout=15,
                description="Select North America",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_country'],
                send_response="United States",
                timeout=15,
                description="Select United States",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_state'],
                send_response="New York",
                timeout=15,
                description="Select New York state",
            ),
            MenuStep(
                expect_pattern=PATTERNS['surface_filter'],
                send_response="paved",
                timeout=10,
                description="Select paved surfaces",
            ),
            MenuStep(
                expect_pattern=PATTERNS['confirm'],
                send_response="y",
                description="Confirm selection",
            ),
        ],
    ),

    # Test 5: Bristol (UK) - Interactive
    5: InteractiveSequence(
        test_id=5,
        description="Bristol UK via interactive menu",
        steps=[
            MenuStep(
                expect_pattern=PATTERNS['checkpoint_resume'],
                send_response="0",  # Start new analysis (skip checkpoint resume)
                timeout=30,
                description="Skip checkpoint resume, start new analysis",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_mode'],
                send_response="1",  # Region-based analysis
                description="Select region analysis mode",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_continent'],
                send_response="2",  # Europe
                timeout=15,
                description="Select Europe continent",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_country'],
                send_response="United Kingdom",
                timeout=15,
                description="Select United Kingdom",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_subregion'],
                send_response="England",
                timeout=15,
                description="Select England",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_subregion'],
                send_response="Bristol",
                timeout=15,
                description="Select Bristol",
            ),
            MenuStep(
                expect_pattern=PATTERNS['confirm'],
                send_response="y",
                description="Confirm selection",
            ),
        ],
    ),

    # Test 13: Cleveland SC 30mi - Interactive Address
    13: InteractiveSequence(
        test_id=13,
        description="30mi radius from Cleveland SC via interactive",
        steps=[
            MenuStep(
                expect_pattern=PATTERNS['checkpoint_resume'],
                send_response="0",  # Start new analysis (skip checkpoint resume)
                timeout=30,
                description="Skip checkpoint resume, start new analysis",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_mode'],
                send_response="2",  # Address-based analysis
                description="Select address analysis mode",
            ),
            MenuStep(
                expect_pattern=PATTERNS['enter_address'],
                send_response="8155 Geer Hwy, Cleveland, SC 29635",
                timeout=15,
                description="Enter address",
            ),
            MenuStep(
                expect_pattern=PATTERNS['enter_radius'],
                send_response="30",
                timeout=10,
                description="Enter 30 mile radius",
            ),
            MenuStep(
                expect_pattern=PATTERNS['confirm'],
                send_response="y",
                description="Confirm selection",
            ),
        ],
    ),

    # Test 14: Georgia (Europe) - Interactive Early Exit
    14: InteractiveSequence(
        test_id=14,
        description="Georgia (Europe) interactive - verify European selection",
        early_exit=True,
        expected_region_pattern=r'(?i)(europe|georgia).*selected|georgia.*europe',
        steps=[
            MenuStep(
                expect_pattern=PATTERNS['checkpoint_resume'],
                send_response="0",  # Start new analysis (skip checkpoint resume)
                timeout=30,
                description="Skip checkpoint resume, start new analysis",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_mode'],
                send_response="2",  # Select country/region (not address)
                description="Select region analysis mode",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_continent'],
                send_response="6",  # Europe (was 2, now at position 6)
                timeout=15,
                description="Select Europe continent",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_country'],
                send_response="20",  # Georgia is at position 20 in Europe list
                timeout=15,
                description="Select Georgia country",
            ),
            # Early exit test - stop here and verify region selection
            # The test will Ctrl+C after this and check expected_region_pattern
        ],
    ),

    # Test 17: Georgia (US) - Interactive Early Exit
    17: InteractiveSequence(
        test_id=17,
        description="Georgia (US) interactive - verify US state selection",
        early_exit=True,
        expected_region_pattern=r'(?i)(united\s*states|us|north\s*america).*georgia|georgia.*state',
        steps=[
            MenuStep(
                expect_pattern=PATTERNS['checkpoint_resume'],
                send_response="0",  # Start new analysis (skip checkpoint resume)
                timeout=30,
                description="Skip checkpoint resume, start new analysis",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_mode'],
                send_response="2",  # Select country/region (not address)
                description="Select region analysis mode",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_continent'],
                send_response="7",  # North America
                timeout=15,
                description="Select North America",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_country'],
                send_response="4",  # US is at position 4 in North America list
                timeout=15,
                description="Select United States",
            ),
            MenuStep(
                expect_pattern=PATTERNS['select_state'],
                send_response="11",  # Georgia is at position 11 in US state list
                timeout=15,
                description="Select Georgia state",
            ),
            # Early exit test - stop here and verify region selection
            # The test will Ctrl+C after this and check expected_region_pattern
        ],
    ),
}


def get_sequence(test_id: int) -> Optional[InteractiveSequence]:
    """Get the interactive sequence for a test ID."""
    return INTERACTIVE_SEQUENCES.get(test_id)


def get_all_interactive_test_ids() -> List[int]:
    """Get all test IDs that use interactive mode."""
    return list(INTERACTIVE_SEQUENCES.keys())


# CLI commands for non-interactive tests
CLI_COMMANDS = {
    1: ["./climb-analyzer", "-r", "Georgia", "--allow-cross-country-merge"],
    3: ["./climb-analyzer", "-r", "Hawaii"],
    6: ["./climb-analyzer", "-r", "Worcestershire", "-s", "paved"],
    7: ["./climb-analyzer", "-r", "Isle of Wight", "--cycling-filter"],
    8: ["./climb-analyzer", "-r", "Rutland,Leicestershire"],
    9: ["./climb-analyzer", "-r", "Luxembourg,Belgium", "--allow-cross-country-merge"],
    10: ["./climb-analyzer", "-r", "Oregon,Idaho"],
    11: ["./climb-analyzer", "-g", "27.1,100.1,27.3,100.3"],  # Lijiang bbox
    12: ["./climb-analyzer", "-a", "8155 Geer Hwy, Cleveland, SC 29635", "--radius", "30"],
    # Early exit CLI tests
    15: ["./climb-analyzer", "-r", "Georgia"],  # Should pick European Georgia
    16: ["./climb-analyzer", "-r", "Georgia"],  # Test disambiguation
    18: ["./climb-analyzer", "-r", "Georgia,Luxembourg"],  # Batch European
    19: ["./climb-analyzer", "-r", "us/Georgia,Luxembourg"],  # Mixed batch
}


def get_cli_command(test_id: int) -> Optional[List[str]]:
    """Get the CLI command for a test ID."""
    return CLI_COMMANDS.get(test_id)


def is_interactive_test(test_id: int) -> bool:
    """Check if a test uses interactive mode."""
    return test_id in INTERACTIVE_SEQUENCES


def is_early_exit_test(test_id: int) -> bool:
    """Check if a test should exit early after region confirmation."""
    if test_id in INTERACTIVE_SEQUENCES:
        return INTERACTIVE_SEQUENCES[test_id].early_exit
    # CLI early exit tests
    return test_id in [15, 16, 18, 19]
