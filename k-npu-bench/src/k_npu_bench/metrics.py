from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable


COMMON_FIELDS = [
    "timestamp",
    "benchmark",
    "vendor",
    "device",
    "runtime",
    "model",
    "task",
    "batch_size",
    "concurrency",
    "input_units",
    "output_units",
    "latency_ms",
    "ttft_ms",
    "tpot_ms",
    "tokens_per_s",
    "fps",
    "avg_power_w",
    "peak_power_w",
    "energy_j",
    "tokens_per_j",
    "fps_per_w",
    "peak_memory_mb",
    "status",
    "notes",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


def estimate_tokens(text: str) -> int:
    """Small dependency-free token estimate for early benchmarking.

    Prefer server-provided usage.total_tokens or tokenizer-specific counting
    when publishing final benchmark numbers.
    """
    if not text:
        return 0
    words = len(text.split())
    chars = max(1, len(text))
    return max(words, round(chars / 4))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize_numeric(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in fields:
        vals: list[float] = []
        for row in rows:
            try:
                raw = row.get(field, "")
                if raw not in ("", None):
                    vals.append(float(raw))
            except (TypeError, ValueError):
                continue
        if not vals:
            continue
        summary[field] = {
            "count": float(len(vals)),
            "mean": statistics.fmean(vals),
            "p50": percentile(vals, 50),
            "p90": percentile(vals, 90),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
            "min": min(vals),
            "max": max(vals),
        }
    return summary


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                items.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return items


def write_rows(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or sorted({key for row in rows for key in row.keys()})
    exists = out.exists() and out.stat().st_size > 0
    with out.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_markdown_summary(
    path: str | Path,
    title: str,
    rows: list[dict[str, Any]],
    numeric_fields: list[str],
) -> None:
    summary = summarize_numeric(rows, numeric_fields)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"Rows: {len(rows)}", ""]
    if summary:
        lines.extend(["| Metric | Count | Mean | P50 | P90 | P95 | P99 | Min | Max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for metric, stats in summary.items():
            lines.append(
                "| {metric} | {count:.0f} | {mean:.3f} | {p50:.3f} | {p90:.3f} | {p95:.3f} | {p99:.3f} | {min:.3f} | {max:.3f} |".format(
                    metric=metric,
                    **stats,
                )
            )
    else:
        lines.append("No numeric fields found.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

