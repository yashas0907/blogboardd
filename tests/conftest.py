"""Shared fixtures for the BlogBoard test suite."""
import os

# Ensure every test runs against local storage with a temp dir and no
# external services — set BEFORE blogboard modules are imported.
os.environ.setdefault("STORAGE_BACKEND", "local")
