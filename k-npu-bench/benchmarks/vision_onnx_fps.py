#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k_npu_bench.metrics import COMMON_FIELDS, monotonic_ms, now_iso, write_rows
from k_npu_bench.power import make_sampler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ONNX Runtime image inference FPS.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--images", required=True, help="Glob pattern, e.g. './samples/*.jpg'")
    p.add_argument("--output", default="results/vision_onnx_fps.csv")
    p.add_argument("--vendor", default="unknown")
    p.add_argument("--device", default="unknown")
    p.add_argument("--runtime", default="onnxruntime")
    p.add_argument("--model", default="onnx-model")
    p.add_argument("--task", default="vision-onnx")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--provider", default="CPUExecutionProvider")
    p.add_argument("--power-sampler", choices=["none", "nvidia-smi"], default="none")
    p.add_argument("--gpu-index", type=int, default=0)
    return p.parse_args()


def preprocess(path: str, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((width, height))
    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def main() -> None:
    import onnxruntime as ort

    args = parse_args()
    images = sorted(glob.glob(args.images))
    if not images:
        raise SystemExit(f"No images matched: {args.images}")
    session = ort.InferenceSession(args.model_path, providers=[args.provider])
    input_name = session.get_inputs()[0].name

    for i in range(args.warmup):
        x = preprocess(images[i % len(images)], args.width, args.height)
        session.run(None, {input_name: x})

    rows = []
    for run_idx in range(args.runs):
        for image_path in images:
            x = preprocess(image_path, args.width, args.height)
            sampler = make_sampler(args.power_sampler, index=args.gpu_index).start()
            start = monotonic_ms()
            status = "ok"
            notes = ""
            try:
                session.run(None, {input_name: x})
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
                    "benchmark": "vision_onnx_fps",
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
                    "notes": notes or f"run={run_idx},image={Path(image_path).name}",
                }
            )
    write_rows(args.output, rows, COMMON_FIELDS)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

