from __future__ import annotations

from .config import AppConfig
from .embedding import EmbeddingClient
from .vectorstore import FaissStore


class Retriever:
    def __init__(
        self,
        config: AppConfig,
        embed_client: EmbeddingClient | None = None,
        store: FaissStore | None = None,
    ):
        if config.embedding is None or config.vectorstore is None or config.rag is None:
            raise ValueError("配置缺少 embedding/vectorstore/rag 段，请检查 config.yaml")
        self.config = config
        self.embed_client = embed_client or EmbeddingClient(config.embedding)
        self.store = store or FaissStore.load(config.vectorstore.index_path)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[float, dict]]:
        k = top_k or self.config.rag.top_k
        vector = self.embed_client.embed([query])[0]
        return self.store.search(vector, k)
