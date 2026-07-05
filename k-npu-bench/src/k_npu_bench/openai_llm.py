from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

from .metrics import estimate_tokens, monotonic_ms


@dataclass
class ChatResult:
    text: str
    latency_ms: float
    ttft_ms: float
    tpot_ms: float
    output_tokens: int
    tokens_per_s: float
    status: str
    error: str = ""


def chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def run_streaming_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_s: float = 300.0,
) -> ChatResult:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    start = monotonic_ms()
    first_token_at: float | None = None
    chunks: list[str] = []
    try:
        with requests.post(
            chat_completions_url(base_url),
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout_s,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                item = json.loads(data)
                delta = item.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    if first_token_at is None:
                        first_token_at = monotonic_ms()
                    chunks.append(content)
        end = monotonic_ms()
        text = "".join(chunks)
        output_tokens = estimate_tokens(text)
        latency = end - start
        ttft = (first_token_at - start) if first_token_at else latency
        decode_ms = max(0.0, end - (first_token_at or end))
        tpot = decode_ms / max(1, output_tokens - 1) if output_tokens > 1 else 0.0
        tokens_per_s = output_tokens / max(0.001, decode_ms / 1000.0) if output_tokens else 0.0
        return ChatResult(
            text=text,
            latency_ms=latency,
            ttft_ms=ttft,
            tpot_ms=tpot,
            output_tokens=output_tokens,
            tokens_per_s=tokens_per_s,
            status="ok",
        )
    except Exception as exc:
        end = monotonic_ms()
        return ChatResult(
            text="",
            latency_ms=end - start,
            ttft_ms=0.0,
            tpot_ms=0.0,
            output_tokens=0,
            tokens_per_s=0.0,
            status="error",
            error=str(exc),
        )

