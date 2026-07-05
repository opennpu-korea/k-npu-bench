#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k_npu_bench.metrics import COMMON_FIELDS, monotonic_ms, now_iso, write_rows
from k_npu_bench.power import make_sampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark image inference HTTP APIs.")
    p.add_argument("--url", required=True)
    p.add_argument("--targets", default="configs/vision_http_targets.csv")
    p.add_argument("--output", default="results/vision_http_fps.csv")
    p.add_argument("--vendor", default="unknown")
    p.add_argument("--device", default="unknown")
    p.add_argument("--runtime", default="http")
    p.add_argument("--model", default="unknown")
    p.add_argument("--task", default="vision")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--mode", choices=["multipart", "base64-json"], default="multipart")
    p.add_argument("--field", default="file")
    p.add_argument("--power-sampler", choices=["none", "nvidia-smi"], default="none")
    p.add_argument("--gpu-index", type=int, default=0)
    return p.parse_args()


def load_targets(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    rows = []
    targets = load_targets(args.targets)
    for run_idx in range(args.runs):
        for item in targets:
            image_path = Path(item["path"])
            sampler = make_sampler(args.power_sampler, index=args.gpu_index).start()
            start = monotonic_ms()
            status = "ok"
            notes = ""
            try:
                if args.mode == "multipart":
                    with image_path.open("rb") as f:
                        resp = requests.post(args.url, files={args.field: f}, timeout=300)
                else:
                    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                    resp = requests.post(args.url, json={"image": encoded, "id": item.get("id")}, timeout=300)
                resp.raise_for_status()
            except Exception as exc:
                status = "error"
                notes = str(exc)
            latency = monotonic_ms() - start
            stats = sampler.stop()
            fps = 1000.0 / latency if latency > 0 else 0.0
            fps_per_w = fps / stats.avg_power_w if stats.avg_power_w > 0 else 0.0
            rows.append(
                {
                    "timestamp": now_iso(),
                    "benchmark": "vision_http_fps",
                    "vendor": args.vendor,
                    "device": args.device,
                    "runtime": args.runtime,
                    "model": args.model,
                    "task": args.task,
                    "batch_size": 1,
                    "concurrency": 1,
                    "input_units": 1,
                    "output_units": 1,
                    "latency_ms": f"{latency:.3f}",
                    "ttft_ms": "",
                    "tpot_ms": "",
                    "tokens_per_s": "",
                    "fps": f"{fps:.3f}",
                    "avg_power_w": f"{stats.avg_power_w:.3f}",
                    "peak_power_w": f"{stats.peak_power_w:.3f}",
                    "energy_j": f"{stats.energy_j:.3f}",
                    "tokens_per_j": "",
                    "fps_per_w": f"{fps_per_w:.6f}",
                    "peak_memory_mb": f"{stats.peak_memory_mb:.3f}",
                    "status": status,
                    "notes": notes or f"run={run_idx},id={item.get('id','')}",
                }
            )
    write_rows(args.output, rows, COMMON_FIELDS)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

