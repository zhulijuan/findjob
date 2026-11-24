from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .bm25 import BM25Index
from .config import AppConfig
from .embedding import EmbeddingClient
from .retriever import Retriever
from .vectorstore import FaissStore

KS = (1, 3, 5, 10)
MRR_K = 10
CONFIGS = ["vector", "bm25", "hybrid", "hybrid+rerank"]


def load_testset(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"测试集不存在: {p}")
    items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        raise ValueError(f"测试集为空: {p}")
    return items


def _config_for(config: AppConfig, name: str) -> AppConfig:
    cfg = config.model_copy(deep=True)
    if name == "vector":
        cfg.retrieval.mode = "vector"
    elif name == "bm25":
        cfg.retrieval.mode = "bm25"
    elif name == "hybrid":
        cfg.retrieval.mode = "hybrid"
    elif name == "hybrid+rerank":
        cfg.retrieval.mode = "hybrid"
        cfg.retrieval.rerank.enabled = True
    return cfg


def evaluate(config: AppConfig, testset_path: str) -> dict:
    if config.retrieval is None or config.vectorstore is None:
        raise ValueError("配置缺少 retrieval/vectorstore 段，请检查 config.yaml")
    testset = load_testset(testset_path)

    store = FaissStore.load(config.vectorstore.index_path)
    bm25 = BM25Index.load(config.vectorstore.index_path)
    embed_client = EmbeddingClient(config.embedding)

    all_records: list[dict] = []
    summary: dict[str, dict] = {}
    for name in CONFIGS:
        retriever = Retriever(_config_for(config, name), embed_client=embed_client, store=store, bm25=bm25)
        recall_hits = {k: 0 for k in KS}
        rr_sum = 0.0
        n_scored = 0
        for item in testset:
            hits = retriever.search(item["question"], top_k=MRR_K)
            hit_ids = [c["chunk_id"] for _, c in hits]
            gold = set(item.get("gold_chunk_ids") or [])
            record = {
                "config": name,
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "hit_ids": hit_ids,
            }
            if gold:
                n_scored += 1
                for k in KS:
                    if gold & set(hit_ids[:k]):
                        recall_hits[k] += 1
                first_rank = next((r for r, cid in enumerate(hit_ids, 1) if cid in gold), None)
                rr_sum += 1.0 / first_rank if first_rank else 0.0
                record["first_gold_rank"] = first_rank
            else:
                record["top1_score"] = hits[0][0] if hits else None
                record["top1_chunk"] = hits[0][1]["chunk_id"] if hits else None
            all_records.append(record)

        summary[name] = {f"recall@{k}": recall_hits[k] / n_scored for k in KS}
        summary[name]["mrr@10"] = rr_sum / n_scored
        summary[name]["n_scored"] = n_scored

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("logs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"eval_{ts}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md_path = out_dir / f"eval_{ts}.md"
    md_path.write_text(_render_markdown(summary, ts, testset_path), encoding="utf-8")

    return {"summary": summary, "jsonl": str(jsonl_path), "markdown": str(md_path)}


def _render_markdown(summary: dict, ts: str, testset_path: str) -> str:
    lines = [
        f"# 检索评测对比（{ts}）",
        "",
        f"- 测试集：{testset_path}",
        "- 指标仅统计带 gold 标注的题目；语料外题目用于定性检查拒答，明细见 jsonl",
        "",
        "| 配置 | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        lines.append(
            f"| {name} | {m['recall@1']:.1%} | {m['recall@3']:.1%} | {m['recall@5']:.1%} | {m['recall@10']:.1%} | {m['mrr@10']:.3f} |"
        )
    return "\n".join(lines) + "\n"
