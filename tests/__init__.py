"""
Test package configuration.

We insert the project `src/` directory into sys.path so tests can import the
library without requiring an editable install. This keeps the workflow light
while the packaging metadata is still under construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))
