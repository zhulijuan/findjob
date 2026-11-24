from __future__ import annotations


def rrf_fuse(
    result_lists: list[list[tuple[float, dict]]],
    rrf_k: int = 60,
    weights: list[float] | None = None,
    top_n: int = 10,
) -> list[tuple[float, dict]]:
    """倒数排名融合：score = Σ w_i / (rrf_k + rank_i)，只依赖排名不依赖原始分数。"""
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    for list_idx, hits in enumerate(result_lists):
        weight = weights[list_idx] if weights else 1.0
        for rank, (_score, chunk) in enumerate(hits):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + weight / (rrf_k + rank + 1)
            chunk_map[cid] = chunk
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [(score, chunk_map[cid]) for cid, score in ranked]
