# Result Schema

모든 벤치마크 스크립트는 가능한 한 같은 CSV 필드를 사용한다.

| Field | Description |
|---|---|
| timestamp | Measurement time |
| benchmark | Script or benchmark name |
| vendor | Vendor name, e.g. FuriosaAI, RBLN, DEEPX, NVIDIA |
| device | Device name, e.g. RNGD, RBLN-CA12, DX-M1, RTX4090 |
| runtime | Runtime or serving stack |
| model | Model name |
| task | Workload task |
| batch_size | Batch size |
| concurrency | Concurrent request level |
| input_units | Input count, prompt count, document count, image count |
| output_units | Output token/vector/document/image count |
| latency_ms | End-to-end latency in milliseconds |
| ttft_ms | Time to first token |
| tpot_ms | Time per output token |
| tokens_per_s | Output token throughput or vector/document throughput |
| fps | Frames/images per second |
| avg_power_w | Average device power |
| peak_power_w | Peak device power |
| energy_j | Energy during measurement window |
| tokens_per_j | Output tokens per joule |
| fps_per_w | FPS per watt |
| peak_memory_mb | Peak device memory |
| status | ok/error |
| notes | Free-form run notes |

## Minimal CSV Example

```csv
timestamp,benchmark,vendor,device,runtime,model,task,batch_size,concurrency,input_units,output_units,latency_ms,ttft_ms,tpot_ms,tokens_per_s,fps,avg_power_w,peak_power_w,energy_j,tokens_per_j,fps_per_w,peak_memory_mb,status,notes
2026-07-06T12:00:00+0900,llm_openai_chat,FuriosaAI,RNGD,Furiosa-LLM,Qwen3,chat,1,1,1,128,3000,250,21.5,46.5,,180,190,540,0.237,,24576,ok,run=0
```

