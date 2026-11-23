from __future__ import annotations

from pathlib import Path

from .chunker import split_document
from .config import AppConfig
from .embedding import EmbeddingClient
from .llm import LLMClient
from .retriever import Retriever
from .runner import build_messages
from .vectorstore import FaissStore


def _check_rag_config(config: AppConfig) -> None:
    if config.embedding is None or config.vectorstore is None or config.rag is None:
        raise ValueError("配置缺少 embedding/vectorstore/rag 段，请检查 config.yaml")


def build_index(config: AppConfig) -> int:
    """语料目录 → 分块 → 向量化 → 持久化索引，返回块数。"""
    _check_rag_config(config)
    corpus_dir = Path(config.rag.corpus_dir)
    files = sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"语料目录为空: {corpus_dir}")

    all_chunks = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        all_chunks.extend(split_document(text, file.name, config.rag.chunk_size, config.rag.chunk_overlap))
    if not all_chunks:
        raise ValueError("语料分块结果为空")

    vectors = EmbeddingClient(config.embedding).embed([c.text for c in all_chunks])
    store = FaissStore(dim=vectors.shape[1])
    store.add(vectors, all_chunks)
    store.save(config.vectorstore.index_path)
    return len(all_chunks)


def build_context(hits: list[tuple[float, dict]]) -> str:
    blocks = []
    for rank, (_score, chunk) in enumerate(hits, 1):
        heading = chunk["heading"] or "正文"
        blocks.append(f"【来源{rank}｜{chunk['source_file']}｜{heading}】\n{chunk['text']}")
    return "\n\n".join(blocks)


def answer_question(config: AppConfig, question: str, stream: bool = False):
    _check_rag_config(config)
    retriever = Retriever(config)
    hits = retriever.search(question)
    if not hits:
        raise ValueError("检索结果为空：请先运行 --build-index 建立索引")

    task = config.tasks.get("rag")
    if task is None:
        raise ValueError("配置缺少 rag 任务定义，请检查 config.yaml tasks 段")

    context = build_context(hits)
    messages = build_messages(task, {"context": context, "question": question})
    client = LLMClient(config.llm)
    response = client.chat(
        messages,
        model=task.model,
        temperature=task.temperature if task.temperature is not None else config.llm.temperature,
        max_tokens=task.max_tokens if task.max_tokens is not None else config.llm.max_tokens,
        thinking=task.thinking,
        stream=stream,
        meta={"task": "rag", "question": question},
    )
    return response, hits
