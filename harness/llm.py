from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig, api_key: str | None = None):
        api_key = api_key or os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"缺少 API Key：请在 .env 中设置 {config.api_key_env}"
                "（DeepSeek 申请地址 platform.deepseek.com/api_keys）"
            )
        self.config = config
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
        stream: bool = False,
        meta: dict | None = None,
        **kwargs,
    ):
        # thinking 模式不支持采样参数，强制忽略
        if thinking:
            temperature = max_tokens = None

        params = dict(
            model=model or self.config.model,
            messages=messages,
            stream=stream,
        )
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if thinking:
            params["extra_body"] = {"thinking": {"type": "enabled"}}
        if stream and self.config.usage_log:
            params["stream_options"] = {"include_usage": True}
        params.update(kwargs)

        start = time.monotonic()
        response = self._client.chat.completions.create(**params)
        if stream:
            return self._log_stream(response, start, params, meta)
        self._write_log(response.usage, params, start, meta)
        return response

    def _log_stream(self, stream, start, params, meta):
        usage = None
        try:
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                yield chunk
        finally:
            self._write_log(usage, params, start, meta)

    def _write_log(self, usage, params, start, meta):
        if not self.config.usage_log:
            return
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "base_url": self.config.base_url,
            "model": params["model"],
            "thinking": "extra_body" in params,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "duration_ms": round((time.monotonic() - start) * 1000),
            "meta": meta or {},
        }
        path = Path(self.config.usage_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
