# Benchmark Methodology

이 문서는 `k-npu-bench` 결과를 공개할 때 적용할 기본 방법론이다.

## 1. Measurement Phases

| Phase | Purpose | Publish? |
|---|---|---|
| Cold start | Server/model load behavior 확인 | 별도 공개 |
| Warmup | JIT, cache, first-run overhead 제거 | 요약 통계에서 제외 |
| Steady state | 안정적인 처리량과 지연시간 측정 | 기본 공개 |
| Load sweep | 동시성 증가에 따른 처리량·지연시간 변화 | 권장 |
| Long run | thermal throttling, memory leak, stability 확인 | 운영 검증 시 권장 |

## 2. LLM Metrics

| Metric | Definition |
|---|---|
| TTFT | request 전송 시작부터 첫 output token 수신까지의 시간 |
| TPOT | 첫 token 이후 output token 1개를 생성하는 평균 시간 |
| tokens/s | decode 구간 output token 처리량 |
| latency | request 시작부터 streaming 종료까지의 end-to-end 시간 |
| tokens/J | output token 수를 측정 구간 에너지로 나눈 값 |

주의사항:

- TTFT/TPOT는 streaming API에서 측정해야 한다.
- 서버가 usage token을 반환하지 않으면 tokenizer 기반 계수가 필요하다.
- 현재 스크립트는 의존성 없는 기본 추정치를 사용하므로, 정식 공개 시 모델별 tokenizer 계수로 보정하는 것이 좋다.

## 3. Vision Metrics

| Metric | Definition |
|---|---|
| latency | 이미지 1장 또는 batch 1회 추론 시간 |
| FPS | `1000 / latency_ms` 또는 전체 프레임 수 / 전체 처리 시간 |
| FPS/W | FPS를 평균 전력으로 나눈 값 |
| peak memory | 측정 중 최대 device memory |

주의사항:

- 카메라 입력, 디코딩, 전처리, 후처리를 포함했는지 명시해야 한다.
- NPU raw inference와 application pipeline FPS는 분리해 공개하는 것이 좋다.

## 4. Power and Memory

기본 스크립트는 `nvidia-smi` 기반 GPU 전력/메모리 샘플링을 지원한다.

국산 NPU는 다음 방식으로 확장하는 것을 권장한다.

| Vendor | Suggested Source |
|---|---|
| FuriosaAI | Furiosa SMI / metrics endpoint / Prometheus exporter |
| Rebellions / RBLN | `rbln-smi`, RBLN system management tools |
| DEEPX | DX-RT / board-level telemetry / external power meter |

전력 공개 시 권장 수준:

- Level 1: device-reported power
- Level 2: wall power meter
- Level 3: both device and wall power

## 5. Load Sweep

권장 동시성:

```text
1, 2, 4, 8, 16, 32
```

각 동시성별 최소 요청 수:

```text
small model: 32+
large model: 16+
production report: 100+
```

공개 지표:

- latency p50/p95/p99
- TTFT p50/p95/p99
- TPOT p50/p95/p99
- aggregate tokens/s
- error rate
- avg/peak power
- peak memory

## 6. Fair Comparison Rules

GPU와 NPU 비교 시 반드시 고정해야 하는 조건:

- model and checkpoint
- quantization / precision
- max tokens
- context length
- batch size
- concurrency
- input prompt set
- tokenizer counting method
- runtime version
- driver version
- temperature and sampling options

서로 다른 precision이나 quantization이면 성능 비교가 아니라 deployment profile 비교로 표시한다.

