"""Pytest configuration for orchestrator test suite."""

import os
import sys
from pathlib import Path

# Add project root and orchestrator root to sys.path
ORCHESTRATOR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ORCHESTRATOR_ROOT.parent

for path in [str(ORCHESTRATOR_ROOT), str(PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)
