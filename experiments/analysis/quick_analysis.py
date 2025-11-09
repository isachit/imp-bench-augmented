#!/usr/bin/env python3
"""Compatibility wrapper for the unified SPAR analysis CLI (quick mode).

This script simply forwards to `analyze_spar_results.py` with `--quick-only`
so existing tooling continues to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze_spar_results import main as analyze_main


def main() -> None:
    args = ["--quick-only"]
    if "--no-save" not in sys.argv[1:]:
        args.append("--no-save")
    analyze_args = args + sys.argv[1:]
    analyze_main(analyze_args)


if __name__ == "__main__":
    main()
