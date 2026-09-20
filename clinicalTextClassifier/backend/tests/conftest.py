"""
Pytest configuration and fixtures for Agent 1 tests.

This file adjusts the Python path so that the app module can be imported.
"""

import sys
from pathlib import Path

# Add the backend directory to sys.path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
