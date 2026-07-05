#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k_npu_bench.metrics import read_csv_rows, write_markdown_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Markdown summary from benchmark CSV.")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Benchmark Summary")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv_rows(args.csv)
    write_markdown_summary(
        args.out,
        args.title,
        rows,
        ["latency_ms", "ttft_ms", "tpot_ms", "tokens_per_s", "fps", "avg_power_w", "peak_power_w", "energy_j", "tokens_per_j", "fps_per_w", "peak_memory_mb"],
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

