"""Entry point for Streamlit Cloud — delegates to dashboard/app.py."""

import runpy
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

runpy.run_path("dashboard/app.py", run_name="__main__")
