from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import OpenAI

from .config import EmbeddingConfig


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig, api_key: str | None = None):
        api_key = api_key or os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"缺少 API Key：请在 .env 中设置 {config.api_key_env}")
        self.config = config
        self._client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout)

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量向量化，返回 (n, dim) float32 矩阵。"""
        if not texts:
            raise ValueError("texts 不能为空")
        for text in texts:
            if not text.strip():
                raise ValueError("存在空文本，无法向量化")

        vectors: list[list[float]] = []
        total_tokens = 0
        start = time.monotonic()
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            response = self._client.embeddings.create(model=self.config.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
            total_tokens += getattr(response.usage, "total_tokens", 0) or 0

        self._write_log(len(texts), total_tokens, start)
        return np.asarray(vectors, dtype=np.float32)

    def _write_log(self, n_inputs: int, total_tokens: int, start: float) -> None:
        if not self.config.usage_log:
            return
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "base_url": self.config.base_url,
            "model": self.config.model,
            "type": "embedding",
            "inputs": n_inputs,
            "total_tokens": total_tokens,
            "duration_ms": round((time.monotonic() - start) * 1000),
        }
        path = Path(self.config.usage_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
