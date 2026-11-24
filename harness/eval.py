from __future__ import annotations

import hashlib
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
METRIC_KEYS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"]
LAST_STATE_PATH = Path("logs/eval/last_state.json")

METRIC_DOC = """## 指标口径说明

| 指标 | 含义 | 风控类比 |
|---|---|---|
| recall@1 | 正确答案块（gold）恰好排在第 1 位的题目占比，最严格的"一次命中"指标 | Top1 捕获率 |
| recall@3 / @5 / @10 | gold 出现在前 3 / 5 / 10 名的题目占比，放宽排名门槛只看"找没找到" | 前 N 名捕获率（类似 lift 表逐层捕获） |
| MRR@10 | 每题第一个 gold 的排名取倒数再平均：第 1 名记 1.0、第 2 名记 0.5、第 3 名记 0.33，前 10 名都没中记 0——同时衡量"找没找到"和"排得靠不靠前" | 命中质量加权得分 |
| 语料外题 | 无 gold，不计入上述指标；用于定性检查"资料里没有的内容会不会瞎答"，明细见 jsonl | 反编造校验 |

判读方法：recall@10 高但 recall@1 低 = "找得到但排不对"，是 reranker 的目标场景；recall@10 也低 = "根本没找到"，要改语料、分块或查询改写。"""


def _file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _snapshot(config: AppConfig) -> dict:
    """知识库状态快照：语料文件 hash、BM25 词典、索引文件 hash、关键配置。"""
    corpus = {
        f.name: _file_sha(f)
        for f in sorted(Path(config.rag.corpus_dir).glob("*"))
        if f.suffix in (".md", ".txt")
    }
    userdict_words: list[str] = []
    if config.retrieval and config.retrieval.jieba_userdict:
        p = Path(config.retrieval.jieba_userdict)
        if p.exists():
            userdict_words = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "corpus": corpus,
        "userdict": userdict_words,
        "index": {
            "faiss_sha": _file_sha(Path(config.vectorstore.index_path + ".faiss")),
            "bm25_sha": _file_sha(Path(config.vectorstore.index_path + ".bm25.json")),
        },
        "config": {
            "mode": config.retrieval.mode if config.retrieval else None,
            "chunk_size": config.rag.chunk_size,
            "chunk_overlap": config.rag.chunk_overlap,
            "top_k": config.rag.top_k,
            "rrf_k": config.retrieval.rrf_k if config.retrieval else None,
            "rerank_enabled": config.retrieval.rerank.enabled if config.retrieval else None,
            "rewrite_enabled": config.retrieval.rewrite.enabled if config.retrieval else None,
        },
    }


CONFIG_LABELS = [
    ("mode", "检索模式"),
    ("chunk_size", "分块大小"),
    ("chunk_overlap", "分块重叠"),
    ("top_k", "top_k"),
    ("rrf_k", "RRF 常数"),
    ("rerank_enabled", "rerank 开关"),
    ("rewrite_enabled", "改写开关"),
]


def _diff_changes(last: dict | None, cur: dict) -> list[str]:
    if not last:
        return ["首次评测（无上次记录），本报告即为基线"]
    changes: list[str] = []

    last_corpus = last.get("corpus", {})
    cur_corpus = cur["corpus"]
    added = sorted(set(cur_corpus) - set(last_corpus))
    removed = sorted(set(last_corpus) - set(cur_corpus))
    modified = sorted(n for n in set(last_corpus) & set(cur_corpus) if last_corpus[n] != cur_corpus[n])
    if added:
        changes.append(f"语料新增 {len(added)} 个文件: {', '.join(added)}")
    if removed:
        changes.append(f"语料删除 {len(removed)} 个文件: {', '.join(removed)}")
    if modified:
        changes.append(f"语料修改 {len(modified)} 个文件: {', '.join(modified)}")

    last_words = set(last.get("userdict", []))
    cur_words = set(cur["userdict"])
    w_added = sorted(cur_words - last_words)
    w_removed = sorted(last_words - cur_words)
    if w_added:
        preview = "、".join(w_added[:10]) + ("…" if len(w_added) > 10 else "")
        changes.append(f"BM25 词典新增 {len(w_added)} 词: {preview}")
    if w_removed:
        preview = "、".join(w_removed[:10]) + ("…" if len(w_removed) > 10 else "")
        changes.append(f"BM25 词典移除 {len(w_removed)} 词: {preview}")

    if last.get("index", {}).get("faiss_sha") != cur["index"]["faiss_sha"]:
        changes.append("向量索引（FAISS）变更/重建")
    if last.get("index", {}).get("bm25_sha") != cur["index"]["bm25_sha"]:
        changes.append("BM25 索引变更/重建")

    last_cfg = last.get("config", {})
    for key, label in CONFIG_LABELS:
        if last_cfg.get(key) != cur["config"].get(key):
            changes.append(f"配置变更: {label} {last_cfg.get(key)} → {cur['config'].get(key)}")

    if not changes:
        changes.append("知识库与配置无变更（基线复测）")
    return changes


def _load_last() -> dict | None:
    if LAST_STATE_PATH.exists():
        try:
            return json.loads(LAST_STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _save_last(ts: str, state: dict, summary: dict) -> None:
    LAST_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_STATE_PATH.write_text(
        json.dumps({"ts": ts, "state": state, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


def _fmt(metric: str, value: float) -> str:
    return f"{value:.1%}" if metric.startswith("recall") else f"{value:.3f}"


def evaluate(config: AppConfig, testset_path: str, note: str | None = None) -> dict:
    if config.retrieval is None or config.vectorstore is None:
        raise ValueError("配置缺少 retrieval/vectorstore 段，请检查 config.yaml")
    testset = load_testset(testset_path)

    state = _snapshot(config)
    last = _load_last()
    changes = _diff_changes(last.get("state") if last else None, state)

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
    md_path.write_text(
        _render_markdown(summary, ts, testset_path, changes, note, last),
        encoding="utf-8",
    )
    meta_path = out_dir / f"eval_{ts}.meta.json"
    meta_path.write_text(
        json.dumps({"ts": ts, "testset": testset_path, "note": note, "changes": changes, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_last(ts, state, summary)

    return {"summary": summary, "changes": changes, "jsonl": str(jsonl_path), "markdown": str(md_path)}


def _render_markdown(
    summary: dict,
    ts: str,
    testset_path: str,
    changes: list[str],
    note: str | None,
    last: dict | None,
) -> str:
    lines = [
        f"# 检索评测对比（{ts}）",
        "",
        f"- 测试集：{testset_path}",
        "",
        "## 本次变更",
        "",
    ]
    lines += [f"- {c}" for c in changes]
    if note:
        lines += ["", f"> 备注：{note}"]

    lines += ["", "| 配置 | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |", "|---|---|---|---|---|---|"]
    for name, m in summary.items():
        lines.append("| " + " | ".join([name] + [_fmt(k, m[k]) for k in METRIC_KEYS]) + " |")

    if last and last.get("summary"):
        lines += [
            "",
            f"### 与上次评测对比（{last.get('ts', '?')}）",
            "",
            "仅列出发生变化的指标，其余持平：",
            "",
            "| 配置 | 指标 | 上次 | 本次 | 变化 |",
            "|---|---|---|---|---|",
        ]
        has_delta = False
        for name, m in summary.items():
            lm = last["summary"].get(name, {})
            for k in METRIC_KEYS:
                if k in lm and abs(m[k] - lm[k]) > 1e-9:
                    delta = m[k] - lm[k]
                    sign = "+" if delta >= 0 else ""
                    delta_str = f"{sign}{delta * 100:.1f}pp" if k.startswith("recall") else f"{sign}{delta:.3f}"
                    lines.append(f"| {name} | {k} | {_fmt(k, lm[k])} | {_fmt(k, m[k])} | {delta_str} |")
                    has_delta = True
        if not has_delta:
            lines.append("| — | — | — | — | 全部持平 |")

    lines += ["", METRIC_DOC, ""]
    return "\n".join(lines) + "\n"
