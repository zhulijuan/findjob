from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from .chunker import Chunk


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-10, None)


class VectorStore:
    """向量库抽象，后续实现（如 Milvus）遵循同一接口。"""

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[float, dict]]:
        raise NotImplementedError

    def save(self, index_path: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class FaissStore(VectorStore):
    """FAISS IndexFlatIP + 向量归一化，内积等价余弦相似度。"""

    def __init__(self, dim: int, index=None, chunks: list[dict] | None = None):
        if faiss is None:
            raise ImportError("未安装 faiss-cpu：pip install faiss-cpu")
        self.dim = dim
        self.index = index if index is not None else faiss.IndexFlatIP(dim)
        self.chunks: list[dict] = chunks if chunks is not None else []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"向量数({vectors.shape[0]})与块数({len(chunks)})不一致")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度({vectors.shape[1]})与索引维度({self.dim})不一致")
        self.index.add(normalize(vectors))
        self.chunks.extend(c.__dict__ if isinstance(c, Chunk) else dict(c) for c in chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[float, dict]]:
        if not self.chunks:
            return []
        k = min(top_k, len(self.chunks))
        query = normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        scores, ids = self.index.search(query, k)
        return [(float(s), self.chunks[i]) for s, i in zip(scores[0], ids[0]) if i != -1]

    def save(self, index_path: str) -> None:
        path = Path(index_path + ".faiss")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        Path(index_path + ".meta.json").write_text(
            json.dumps({"dim": self.dim, "chunks": self.chunks}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: str) -> FaissStore:
        path = Path(index_path + ".faiss")
        if not path.exists():
            raise FileNotFoundError(f"索引不存在: {path}（请先运行 --build-index）")
        index = faiss.read_index(str(path))
        meta = json.loads(Path(index_path + ".meta.json").read_text(encoding="utf-8"))
        return cls(dim=meta["dim"], index=index, chunks=meta["chunks"])

    def __len__(self) -> int:
        return len(self.chunks)
