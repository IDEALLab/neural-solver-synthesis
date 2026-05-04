"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add deps to path for submodule imports
deps_path = project_root / "deps"
if str(deps_path) not in sys.path:
    sys.path.insert(0, str(deps_path))
