"""
conftest.py — repo-root pytest configuration.

Adds the repo root to sys.path so that `import src.data.ffpp_splits` works
without requiring `pip install -e .` first.  This is the most portable
approach — works with any pytest version >= 6, on Linux and Windows, with or
without a virtual environment.
"""
import sys
from pathlib import Path

# Ensure the repo root (the directory containing this file) is on sys.path.
# pathlib.resolve() gives us an absolute path regardless of where pytest is
# invoked from, so imports like `from src.data.manifest import ...` always work.
sys.path.insert(0, str(Path(__file__).parent.resolve()))
