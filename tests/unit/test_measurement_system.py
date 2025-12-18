#!/usr/bin/env python3
"""
Unit tests for measurement system detection (metric vs imperial).

Tests the address pattern matching logic that determines whether to use
imperial units (US addresses) or metric units (non-US addresses).

The actual function `_detect_us_from_address_pattern` is nested inside
`process_address_based_climb_search` in engine.py. These tests verify
the pattern matching logic independently.
"""

import pytest
import re
from typing import Optional


# Reimplementation of the pattern matching logic for testing
# This mirrors the logic in engine.py:18521-18544
def detect_us_from_address_pattern(addr: str) -> Optional[bool]:
    """
    Detect if address is in the US based on common patterns.

    Returns:
        True: US address detected (use imperial)
        None: Unknown, needs API call (default to metric)
    """
    addr_lower = addr.lower()

    # US state abbreviations
    us_states = [
        'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
        'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
        'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
        'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
        'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy', 'dc'
    ]

    # Check for ", XX" pattern (city, state abbreviation)
    state_match = re.search(r',\s*([a-z]{2})\b', addr_lower)
    if state_match and state_match.group(1) in us_states:
        return True

    # Explicit country mentions
    if 'united states' in addr_lower or ', usa' in addr_lower:
        return True

    return None  # Unknown - need API call


class TestUSStateAbbreviationDetection:
    """Tests for US state abbreviation pattern matching."""

    def test_city_state_format(self):
        """Standard 'City, ST' format should detect US."""
        assert detect_us_from_address_pattern("Denver, CO") is True
        assert detect_us_from_address_pattern("Los Angeles, CA") is True
        assert detect_us_from_address_pattern("New York, NY") is True

    def test_full_address_with_state(self):
        """Full address with state abbreviation should detect US."""
        assert detect_us_from_address_pattern("123 Main St, Denver, CO") is True
        assert detect_us_from_address_pattern("456 Broadway, New York, NY 10001") is True

    def test_all_state_abbreviations(self):
        """Test all 50 states + DC are recognized."""
        states = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
        ]
        for state in states:
            result = detect_us_from_address_pattern(f"City, {state}")
            assert result is True, f"State {state} should be detected as US"

    def test_case_insensitive(self):
        """Detection should be case insensitive."""
        assert detect_us_from_address_pattern("DENVER, CO") is True
        assert detect_us_from_address_pattern("denver, co") is True
        assert detect_us_from_address_pattern("Denver, co") is True


class TestExplicitUSMentions:
    """Tests for explicit 'United States' or 'USA' mentions."""

    def test_united_states_suffix(self):
        """Addresses with 'United States' should detect US."""
        assert detect_us_from_address_pattern("Denver, Colorado, United States") is True
        assert detect_us_from_address_pattern("New York, United States of America") is True

    def test_usa_suffix(self):
        """Addresses with ', USA' should detect US."""
        assert detect_us_from_address_pattern("Denver, USA") is True
        assert detect_us_from_address_pattern("Los Angeles, California, USA") is True

    def test_usa_case_variations(self):
        """USA detection should be case insensitive."""
        assert detect_us_from_address_pattern("Denver, usa") is True
        assert detect_us_from_address_pattern("Denver, USA") is True


class TestNonUSAddresses:
    """Tests for non-US addresses (should return None, not False)."""

    def test_european_cities(self):
        """European addresses should return None (unknown)."""
        assert detect_us_from_address_pattern("Paris, France") is None
        assert detect_us_from_address_pattern("London, UK") is None
        assert detect_us_from_address_pattern("Berlin, Germany") is None
        assert detect_us_from_address_pattern("Rome, Italy") is None

    def test_asian_cities(self):
        """Asian addresses should return None (unknown)."""
        assert detect_us_from_address_pattern("Tokyo, Japan") is None
        assert detect_us_from_address_pattern("Beijing, China") is None
        assert detect_us_from_address_pattern("Mumbai, India") is None

    def test_south_american_cities(self):
        """South American addresses should return None (unknown)."""
        assert detect_us_from_address_pattern("São Paulo, Brazil") is None
        assert detect_us_from_address_pattern("Buenos Aires, Argentina") is None

    def test_australian_cities(self):
        """Australian addresses should return None (unknown)."""
        assert detect_us_from_address_pattern("Sydney, Australia") is None
        assert detect_us_from_address_pattern("Melbourne, AU") is None


class TestEdgeCases:
    """Tests for edge cases and potential false positives."""

    def test_two_letter_country_codes_not_us_states(self):
        """Two-letter country codes that aren't US states should return None."""
        # UK, FR, DE, etc. are not US state abbreviations
        assert detect_us_from_address_pattern("London, UK") is None
        assert detect_us_from_address_pattern("Paris, FR") is None
        assert detect_us_from_address_pattern("Berlin, DE") is None  # DE is Delaware!

    def test_delaware_vs_germany_de(self):
        """DE is Delaware - German addresses need different format."""
        # ", DE" matches Delaware
        assert detect_us_from_address_pattern("Wilmington, DE") is True
        # Germany uses "Germany" not "DE" in typical addresses
        assert detect_us_from_address_pattern("Berlin, Germany") is None

    def test_georgia_country_vs_state(self):
        """Georgia country vs Georgia US state - both use GA abbreviation."""
        # ", GA" matches Georgia state
        assert detect_us_from_address_pattern("Atlanta, GA") is True
        # Georgia country typically written differently
        assert detect_us_from_address_pattern("Tbilisi, Georgia") is None

    def test_empty_and_malformed_addresses(self):
        """Empty or malformed addresses should return None."""
        assert detect_us_from_address_pattern("") is None
        assert detect_us_from_address_pattern("Unknown") is None
        assert detect_us_from_address_pattern("123") is None

    def test_just_city_name(self):
        """Just a city name without state should return None."""
        assert detect_us_from_address_pattern("Denver") is None
        assert detect_us_from_address_pattern("Paris") is None

    def test_coordinates_only(self):
        """Coordinate strings should return None."""
        assert detect_us_from_address_pattern("39.7392, -104.9903") is None


class TestMeasurementUnitImplication:
    """Tests documenting the measurement unit implications."""

    def test_us_implies_imperial(self):
        """US detection (True) means imperial units should be used."""
        us_addresses = [
            "Denver, CO",
            "Los Angeles, CA",
            "New York, USA",
            "Chicago, Illinois, United States"
        ]
        for addr in us_addresses:
            result = detect_us_from_address_pattern(addr)
            assert result is True, f"{addr} should use imperial units"

    def test_unknown_implies_metric_default(self):
        """Unknown detection (None) means metric units as default."""
        non_us_addresses = [
            "Paris, France",
            "London, UK",
            "Tokyo, Japan",
            "Sydney, Australia"
        ]
        for addr in non_us_addresses:
            result = detect_us_from_address_pattern(addr)
            assert result is None, f"{addr} should default to metric units"
