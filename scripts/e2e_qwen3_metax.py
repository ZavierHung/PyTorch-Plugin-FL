#!/usr/bin/env python3
"""MetaX v2 e2e runner (repo-root entry avoids tests/perf sys.path quirk on muxi)."""
import runpy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
target = _REPO / "tests" / "perf" / "e2e_qwen3_infer_cuda.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
