#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k_npu_bench.metrics import COMMON_FIELDS, now_iso, write_rows
from k_npu_bench.openai_llm import run_streaming_chat
from k_npu_bench.power import make_sampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark OpenAI-compatible vision-language chat APIs.")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Describe this image in one sentence.")
    p.add_argument("--output", default="results/multimodal_openai_chat.csv")
    p.add_argument("--vendor", default="unknown")
    p.add_argument("--device", default="unknown")
    p.add_argument("--runtime", default="openai-compatible")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--power-sampler", choices=["none", "nvidia-smi"], default="none")
    p.add_argument("--gpu-index", type=int, default=0)
    return p.parse_args()


def data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main() -> None:
    args = parse_args()
    image_url = data_url(args.image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    rows = []
    for run_idx in range(args.runs):
        sampler = make_sampler(args.power_sampler, index=args.gpu_index).start()
        result = run_streaming_chat(
            args.base_url,
            args.api_key,
            args.model,
            messages,
            args.max_tokens,
            args.temperature,
        )
        stats = sampler.stop()
        tokens_per_j = result.output_tokens / stats.energy_j if stats.energy_j > 0 else 0.0
        rows.append(
            {
                "timestamp": now_iso(),
                "benchmark": "multimodal_openai_chat",
                "vendor": args.vendor,
                "device": args.device,
                "runtime": args.runtime,
                "model": args.model,
                "task": "vision-language",
                "batch_size": 1,
                "concurrency": 1,
                "input_units": 1,
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
                "notes": result.error or f"run={run_idx},image={Path(args.image).name}",
            }
        )
    write_rows(args.output, rows, COMMON_FIELDS)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

