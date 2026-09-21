#!/usr/bin/env python3
"""Compatibility wrapper around the unified SPAR analysis CLI for LLM judge."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze_spar_results import main as analyze_main


def main() -> None:
    extra_args = ["--llm-judge"]
    analyze_args = extra_args + sys.argv[1:]
    analyze_main(analyze_args)


if __name__ == "__main__":
    main()
