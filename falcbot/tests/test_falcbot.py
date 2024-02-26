"""
Unit and regression test for the falcbot package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import falcbot


def test_falcbot_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "falcbot" in sys.modules
