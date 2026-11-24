from __future__ import annotations

from .bm25 import BM25Index
from .config import AppConfig
from .embedding import EmbeddingClient
from .fusion import rrf_fuse
from .reranker import RerankerClient
from .rewrite import maybe_rewrite_query
from .vectorstore import FaissStore


class Retriever:
    """检索编排：查询改写 → 双路召回（BM25/向量）→ RRF 融合 → 可选精排。"""

    def __init__(
        self,
        config: AppConfig,
        embed_client: EmbeddingClient | None = None,
        store: FaissStore | None = None,
        bm25: BM25Index | None = None,
    ):
        if config.embedding is None or config.vectorstore is None or config.rag is None:
            raise ValueError("配置缺少 embedding/vectorstore/rag 段，请检查 config.yaml")
        self.config = config
        self.retrieval = config.retrieval
        self.mode = self.retrieval.mode if self.retrieval else "vector"
        self.embed_client = embed_client
        self._store = store
        self._bm25 = bm25
        self._reranker: RerankerClient | None = None

    def _get_embed(self) -> EmbeddingClient:
        if self.embed_client is None:
            self.embed_client = EmbeddingClient(self.config.embedding)
        return self.embed_client

    def _get_store(self) -> FaissStore:
        if self._store is None:
            self._store = FaissStore.load(self.config.vectorstore.index_path)
        return self._store

    def _get_bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.load(self.config.vectorstore.index_path)
        return self._bm25

    def _get_reranker(self) -> RerankerClient:
        if self._reranker is None:
            self._reranker = RerankerClient(self.retrieval.rerank)
        return self._reranker

    def search(self, query: str, top_k: int | None = None) -> list[tuple[float, dict]]:
        k = top_k or self.config.rag.top_k
        query = maybe_rewrite_query(self.config, query)
        rerank_cfg = self.retrieval.rerank if self.retrieval else None
        fetch_k = max(k, rerank_cfg.top_m) if (rerank_cfg and rerank_cfg.enabled) else k

        if self.mode == "bm25":
            hits = self._get_bm25().search(query, fetch_k)
        elif self.mode == "hybrid":
            v_hits = self._get_store().search(
                self._get_embed().embed([query])[0], self.retrieval.vector_candidates
            )
            b_hits = self._get_bm25().search(query, self.retrieval.bm25_candidates)
            hits = rrf_fuse([v_hits, b_hits], rrf_k=self.retrieval.rrf_k, top_n=fetch_k)
        else:
            hits = self._get_store().search(self._get_embed().embed([query])[0], fetch_k)

        if rerank_cfg and rerank_cfg.enabled and hits:
            hits = self._get_reranker().rerank(
                query, [c for _, c in hits], usage_log=self.config.llm.usage_log
            )
        return hits[:k]
