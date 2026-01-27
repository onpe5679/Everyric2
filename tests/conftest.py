"""Pytest configuration and fixtures."""

import pytest

from everyric2.config.settings import reset_settings


@pytest.fixture(autouse=True)
def reset_settings_before_test():
    """Reset settings before each test."""
    reset_settings()
    yield
    reset_settings()
