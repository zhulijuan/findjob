from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunker import Chunk


def tokenize(text: str, userdict: list[str] | None = None) -> list[str]:
    import jieba

    jieba.setLogLevel(60)
    for word in userdict or []:
        jieba.add_word(word)
    return [t for t in jieba.lcut(text) if t.strip()]


def load_userdict(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


class BM25Index:
    def __init__(self, chunks: list[dict], corpus_tokens: list[list[str]], userdict: list[str]):
        self.chunks = chunks
        self.userdict = userdict
        self._corpus_tokens = corpus_tokens
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    @classmethod
    def build(cls, chunks: list[Chunk], userdict_path: str | None = None) -> BM25Index:
        userdict = load_userdict(userdict_path)
        chunk_dicts = [c.__dict__ if isinstance(c, Chunk) else dict(c) for c in chunks]
        corpus_tokens = [tokenize(c["text"], userdict) for c in chunk_dicts]
        return cls(chunks=chunk_dicts, corpus_tokens=corpus_tokens, userdict=userdict)

    def search(self, query: str, top_k: int) -> list[tuple[float, dict]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query, self.userdict))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(float(scores[i]), self.chunks[i]) for i in order if scores[i] > 0]

    def save(self, index_path: str) -> None:
        path = Path(index_path + ".bm25.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"chunks": self.chunks, "corpus_tokens": self._corpus_tokens, "userdict": self.userdict},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: str) -> BM25Index:
        path = Path(index_path + ".bm25.json")
        if not path.exists():
            raise FileNotFoundError(f"BM25 索引不存在: {path}（请重新运行 --build-index）")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(chunks=data["chunks"], corpus_tokens=data["corpus_tokens"], userdict=data["userdict"])
