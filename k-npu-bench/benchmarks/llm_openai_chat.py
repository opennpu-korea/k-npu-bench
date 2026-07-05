#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k_npu_bench.metrics import COMMON_FIELDS, now_iso, read_jsonl, write_rows
from k_npu_bench.openai_llm import run_streaming_chat
from k_npu_bench.power import make_sampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark OpenAI-compatible streaming chat APIs.")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", default="configs/prompts.jsonl")
    p.add_argument("--output", default="results/llm_openai_chat.csv")
    p.add_argument("--vendor", default="unknown")
    p.add_argument("--device", default="unknown")
    p.add_argument("--runtime", default="openai-compatible")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--power-sampler", choices=["none", "nvidia-smi"], default="none")
    p.add_argument("--gpu-index", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompts = read_jsonl(args.prompts)
    rows = []

    for idx in range(args.warmup):
        item = prompts[idx % len(prompts)]
        run_streaming_chat(
            args.base_url,
            args.api_key,
            args.model,
            item["messages"],
            args.max_tokens,
            args.temperature,
        )

    for run_idx in range(args.runs):
        for item in prompts:
            sampler = make_sampler(args.power_sampler, index=args.gpu_index).start()
            result = run_streaming_chat(
                args.base_url,
                args.api_key,
                args.model,
                item["messages"],
                args.max_tokens,
                args.temperature,
            )
            stats = sampler.stop()
            tokens_per_j = result.output_tokens / stats.energy_j if stats.energy_j > 0 else 0.0
            rows.append(
                {
                    "timestamp": now_iso(),
                    "benchmark": "llm_openai_chat",
                    "vendor": args.vendor,
                    "device": args.device,
                    "runtime": args.runtime,
                    "model": args.model,
                    "task": item.get("id", "chat"),
                    "batch_size": 1,
                    "concurrency": args.concurrency,
                    "input_units": len(str(item["messages"])),
                    "output_units": result.output_tokens,
                    "latency_ms": f"{result.latency_ms:.3f}",
                    "ttft_ms": f"{result.ttft_ms:.3f}",
                    "tpot_ms": f"{result.tpot_ms:.3f}",
                    "tokens_per_s": f"{result.tokens_per_s:.3f}",
                    "fps": "",
                    "avg_power_w": f"{stats.avg_power_w:.3f}",
                    "peak_power_w": f"{stats.peak_power_w:.3f}",
                    "energy_j": f"{stats.energy_j:.3f}",
                    "tokens_per_j": f"{tokens_per_j:.6f}",
                    "fps_per_w": "",
                    "peak_memory_mb": f"{stats.peak_memory_mb:.3f}",
                    "status": result.status,
                    "notes": result.error or f"run={run_idx}",
                }
            )

    write_rows(args.output, rows, COMMON_FIELDS)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

