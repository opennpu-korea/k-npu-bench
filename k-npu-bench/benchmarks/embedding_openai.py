#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k_npu_bench.metrics import COMMON_FIELDS, monotonic_ms, now_iso, read_jsonl, write_rows
from k_npu_bench.power import make_sampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark OpenAI-compatible embeddings APIs.")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True)
    p.add_argument("--input", default="configs/prompts.jsonl")
    p.add_argument("--output", default="results/embedding_openai.csv")
    p.add_argument("--vendor", default="unknown")
    p.add_argument("--device", default="unknown")
    p.add_argument("--runtime", default="openai-compatible")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--power-sampler", choices=["none", "nvidia-smi"], default="none")
    p.add_argument("--gpu-index", type=int, default=0)
    return p.parse_args()


def item_text(item: dict) -> str:
    if "text" in item:
        return str(item["text"])
    if "messages" in item:
        return "\n".join(m.get("content", "") for m in item["messages"])
    return str(item)


def main() -> None:
    args = parse_args()
    items = read_jsonl(args.input)
    texts = [item_text(x) for x in items]
    rows = []
    url = args.base_url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}

    batches = [texts[i : i + args.batch_size] for i in range(0, len(texts), args.batch_size)]
    for run_idx in range(args.runs):
        for batch in batches:
            sampler = make_sampler(args.power_sampler, index=args.gpu_index).start()
            start = monotonic_ms()
            status = "ok"
            notes = ""
            vectors = 0
            try:
                resp = requests.post(url, headers=headers, json={"model": args.model, "input": batch}, timeout=300)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                vectors = len(data)
            except Exception as exc:
                status = "error"
                notes = str(exc)
            latency = monotonic_ms() - start
            stats = sampler.stop()
            per_s = vectors / max(0.001, latency / 1000.0)
            rows.append(
                {
                    "timestamp": now_iso(),
                    "benchmark": "embedding_openai",
                    "vendor": args.vendor,
                    "device": args.device,
                    "runtime": args.runtime,
                    "model": args.model,
                    "task": "embedding",
                    "batch_size": len(batch),
                    "concurrency": 1,
                    "input_units": len(batch),
                    "output_units": vectors,
                    "latency_ms": f"{latency:.3f}",
                    "ttft_ms": "",
                    "tpot_ms": "",
                    "tokens_per_s": f"{per_s:.3f}",
                    "fps": "",
                    "avg_power_w": f"{stats.avg_power_w:.3f}",
                    "peak_power_w": f"{stats.peak_power_w:.3f}",
                    "energy_j": f"{stats.energy_j:.3f}",
                    "tokens_per_j": "",
                    "fps_per_w": "",
                    "peak_memory_mb": f"{stats.peak_memory_mb:.3f}",
                    "status": status,
                    "notes": notes or f"run={run_idx}",
                }
            )
    write_rows(args.output, rows, COMMON_FIELDS)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

