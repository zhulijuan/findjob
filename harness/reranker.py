from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import httpx

from .config import RerankConfig


class RerankerClient:
    """Cross-encoder 精排客户端（SiliconFlow /rerank，Cohere 风格协议）。"""

    def __init__(self, config: RerankConfig, api_key: str | None = None):
        import os

        api_key = api_key or os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"缺少 API Key：请在 .env 中设置 {config.api_key_env}")
        self.config = config
        self._api_key = api_key

    def rerank(self, query: str, chunks: list[dict], usage_log: str | None = None) -> list[tuple[float, dict]]:
        if not chunks:
            return []
        start = time.monotonic()
        response = httpx.post(
            f"{self.config.base_url.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.config.model,
                "query": query,
                "documents": [c["text"] for c in chunks],
                "top_n": len(chunks),
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        self._write_log(len(chunks), data.get("meta", {}), start, usage_log)
        ranked = sorted(data["results"], key=lambda r: -r["relevance_score"])
        return [(float(r["relevance_score"]), chunks[r["index"]]) for r in ranked]

    def _write_log(self, n_docs: int, meta: dict, start: float, usage_log: str | None) -> None:
        if not usage_log:
            return
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "base_url": self.config.base_url,
            "model": self.config.model,
            "type": "rerank",
            "inputs": n_docs,
            "duration_ms": round((time.monotonic() - start) * 1000),
            "meta": meta or {},
        }
        path = Path(usage_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
