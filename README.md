# k-npu-bench

GPU 사용자 관점에서 국산 NPU와 SoC 추론 성능을 비교·공개하기 위한 벤치마크 스크립트 저장소다.

목표는 FuriosaAI, Rebellions/RBLN, DEEPX 등 국산 NPU 환경에서도 GPU 사용자에게 익숙한 지표를 같은 CSV/Markdown 형식으로 기록하는 것이다.

## Metrics

| Metric | Meaning | Main Target |
|---|---|---|
| TTFT | Time To First Token | Streaming LLM |
| TPOT | Time Per Output Token | Streaming LLM |
| tokens/s | Decode throughput | LLM, embedding, reranking |
| latency | End-to-end response time | All workloads |
| FPS | Frames or images per second | Vision, video, ONNX |
| power | Average and peak power | GPU/NPU efficiency |
| memory | Peak device memory | Capacity and stability |
| tokens/J | Energy efficiency | LLM |
| FPS/W | Vision energy efficiency | Vision |

## Repository Layout

```text
k-npu-bench/
  benchmarks/
    llm_openai_chat.py
    embedding_openai.py
    rerank_http.py
    multimodal_openai_chat.py
    vision_http_fps.py
    vision_onnx_fps.py
  methodologies/
    llm_load_sweep.py
    power_efficiency_sweep.py
  scripts/
    summarize_results.py
    merge_csv.py
  configs/
    prompts.jsonl
    rerank_cases.jsonl
    vision_http_targets.csv
  docs/
    methodology.md
    result-schema.md
  results/
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ONNX Runtime이 필요 없는 LLM/API 벤치마크만 실행한다면 `requests`만 있어도 된다.

## Quick Start

### 1. OpenAI-Compatible LLM Chat

Furiosa-LLM, vLLM-RBLN, 일반 vLLM, SGLang, llama.cpp server 등 OpenAI 호환 API 서버에 사용할 수 있다.

```bash
python benchmarks/llm_openai_chat.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model furiosa-ai/Qwen3-4B-FP8 \
  --vendor FuriosaAI \
  --device RNGD \
  --runtime Furiosa-LLM \
  --prompts configs/prompts.jsonl \
  --runs 5 \
  --output results/llm_chat_rngd.csv
```

NVIDIA GPU 서버와 비교할 때는 전력/메모리 샘플링을 켠다.

```bash
python benchmarks/llm_openai_chat.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model Qwen/Qwen3-8B \
  --vendor NVIDIA \
  --device RTX4090 \
  --runtime vLLM \
  --power-sampler nvidia-smi \
  --gpu-index 0 \
  --output results/llm_chat_rtx4090.csv
```

### 2. LLM Load Sweep

동시성별 TTFT, TPOT, latency, tokens/s 분포를 본다.

```bash
python methodologies/llm_load_sweep.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model Qwen/Qwen3-8B \
  --concurrency-levels 1,2,4,8,16 \
  --requests-per-level 32 \
  --output results/llm_load_sweep.csv
```

### 3. Embedding

```bash
python benchmarks/embedding_openai.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model Qwen3-Embedding \
  --input configs/prompts.jsonl \
  --batch-size 4 \
  --output results/embedding.csv
```

### 4. Reranking

```bash
python benchmarks/rerank_http.py \
  --url http://localhost:8000/v1/rerank \
  --api-key EMPTY \
  --model Qwen3-Reranker \
  --cases configs/rerank_cases.jsonl \
  --output results/rerank.csv
```

### 5. Vision HTTP FPS

```bash
python benchmarks/vision_http_fps.py \
  --url http://localhost:8080/infer \
  --targets configs/vision_http_targets.csv \
  --vendor DEEPX \
  --device DX-M1 \
  --runtime DX-RT \
  --model YOLO \
  --task object-detection \
  --output results/vision_http_dx_m1.csv
```

### 6. ONNX Runtime FPS

```bash
python benchmarks/vision_onnx_fps.py \
  --model-path model.onnx \
  --images "./samples/*.jpg" \
  --provider CPUExecutionProvider \
  --output results/vision_onnx.csv
```

### 7. Markdown Summary

```bash
python scripts/summarize_results.py \
  --csv results/llm_load_sweep.csv \
  --out results/llm_load_sweep.md \
  --title "LLM Load Sweep"
```

## Result Files

Benchmark outputs are CSV by default. Markdown summaries can be generated for GitHub publication.

Recommended public result layout:

```text
results/
  2026-07/
    furiosa-rngd-qwen3/
      raw.csv
      summary.md
      env.md
    rbln-atom-qwen3/
      raw.csv
      summary.md
      env.md
    deepx-dxm1-yolo/
      raw.csv
      summary.md
      env.md
```

## Benchmark Rules

- Report hardware, SDK/runtime version, model, quantization, batch size, concurrency, prompt set, and sampling method.
- Exclude warmup requests from published latency/FPS summary.
- Publish p50, p90, p95, p99, mean, min, max.
- Do not compare GPU and NPU without noting precision, quantization, context length, batch size, and request concurrency.
- Use `Needs Check` or `Experimental` labels for unofficial vendor support.

## License

MIT

