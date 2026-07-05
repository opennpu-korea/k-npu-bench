#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k_npu_bench.metrics import COMMON_FIELDS, now_iso, read_jsonl, write_rows
from k_npu_bench.openai_llm import run_streaming_chat
from k_npu_bench.power import make_sampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Professional LLM load sweep for OpenAI-compatible APIs.")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", default="configs/prompts.jsonl")
    p.add_argument("--output", default="results/llm_load_sweep.csv")
    p.add_argument("--vendor", default="unknown")
    p.add_argument("--device", default="unknown")
    p.add_argument("--runtime", default="openai-compatible")
    p.add_argument("--concurrency-levels", default="1,2,4,8")
    p.add_argument("--requests-per-level", type=int, default=16)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--power-sampler", choices=["none", "nvidia-smi"], default="none")
    p.add_argument("--gpu-index", type=int, default=0)
    return p.parse_args()


def one_request(args: argparse.Namespace, messages: list[dict]) -> dict:
    result = run_streaming_chat(
        args.base_url,
        args.api_key,
        args.model,
        messages,
        args.max_tokens,
        args.temperature,
    )
    return {
        "latency_ms": result.latency_ms,
        "ttft_ms": result.ttft_ms,
        "tpot_ms": result.tpot_ms,
        "tokens_per_s": result.tokens_per_s,
        "output_units": result.output_tokens,
        "status": result.status,
        "notes": result.error,
    }


def main() -> None:
    args = parse_args()
    prompts = read_jsonl(args.prompts)
    levels = [int(x.strip()) for x in args.concurrency_levels.split(",") if x.strip()]

    for i in range(args.warmup):
        one_request(args, prompts[i % len(prompts)]["messages"])

    rows = []
    for level in levels:
        sampler = make_sampler(args.power_sampler, index=args.gpu_index).start()
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = []
            for i in range(args.requests_per_level):
                item = prompts[i % len(prompts)]
                futures.append(pool.submit(one_request, args, item["messages"]))
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                rows.append(
                    {
                        "timestamp": now_iso(),
                        "benchmark": "llm_load_sweep",
                        "vendor": args.vendor,
                        "device": args.device,
                        "runtime": args.runtime,
                        "model": args.model,
                        "task": "chat_load_sweep",
                        "batch_size": 1,
                        "concurrency": level,
                        "input_units": 1,
                        "output_units": result["output_units"],
                        "latency_ms": f"{result['latency_ms']:.3f}",
                        "ttft_ms": f"{result['ttft_ms']:.3f}",
                        "tpot_ms": f"{result['tpot_ms']:.3f}",
                        "tokens_per_s": f"{result['tokens_per_s']:.3f}",
                        "fps": "",
                        "avg_power_w": "",
                        "peak_power_w": "",
                        "energy_j": "",
                        "tokens_per_j": "",
                        "fps_per_w": "",
                        "peak_memory_mb": "",
                        "status": result["status"],
                        "notes": result["notes"] or f"request={i}",
                    }
                )
        stats = sampler.stop()
        if rows and stats.energy_j > 0:
            level_rows = [r for r in rows if int(r["concurrency"]) == level]
            total_tokens = sum(float(r["output_units"] or 0) for r in level_rows)
            tokens_per_j = total_tokens / stats.energy_j
            for row in level_rows:
                row["avg_power_w"] = f"{stats.avg_power_w:.3f}"
                row["peak_power_w"] = f"{stats.peak_power_w:.3f}"
                row["energy_j"] = f"{stats.energy_j:.3f}"
                row["tokens_per_j"] = f"{tokens_per_j:.6f}"
                row["peak_memory_mb"] = f"{stats.peak_memory_mb:.3f}"

    write_rows(args.output, rows, COMMON_FIELDS)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

