#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Inspect DeepSeek V4 routed-expert safetensors layout without reading payloads.

Run: python3 bench/layer_contig.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from layer_contig_model import LayoutError, analyze, scan_layout
from layer_contig_report import print_report, write_artifact


def parse_args() -> argparse.Namespace:
    """Parse header-only tool locations."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=root / "models/deepseek-v4-flash")
    parser.add_argument("--output", type=Path, default=root / "artifacts/layer_contig.json")
    return parser.parse_args()


def main() -> int:
    """Run header scan and materialize complete real-offset artifact."""
    args = parse_args()
    try:
        analysis = analyze(scan_layout(args.model_dir))
        write_artifact(args.output, analysis.records)
        print_report(analysis, args.output)
    except (LayoutError, OSError) as error:
        print(f"layer_contig: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
