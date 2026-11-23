from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source_file: str
    heading: str
    chunk_id: str


def chunker_config_validator(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正数，收到: {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap 不能为负，收到: {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})")


# 章/条/标题/序号开头的行，视为结构切分点
HEADING_PAT = re.compile(
    r"^(第[一二三四五六七八九十百零]+[章节条].*"
    r"|[一二三四五六七八九十]+、.*"
    r"|\d+(\.\d+)*[、\.].*"
    r"|#+\s*.*)$"
)

SENTENCE_SPLIT_PAT = re.compile(r"(?<=[。；！？])")


def split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """按句子切分超长文本，块间携带 overlap 尾部。"""
    sentences = [s for s in SENTENCE_SPLIT_PAT.split(text) if s]
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sentence in sentences:
        if cur and cur_len + len(sentence) > chunk_size:
            pieces.append("".join(cur))
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(cur):
                if tail_len + len(prev) > chunk_overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            cur = tail
            cur_len = tail_len
        cur.append(sentence)
        cur_len += len(sentence)
    if cur:
        pieces.append("".join(cur))
    return pieces


def split_document(
    text: str,
    source_file: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int = 40,
) -> list[Chunk]:
    """中文条款感知分块：按章/条/标题行切分结构单元，超长单元按句切分。"""
    chunker_config_validator(chunk_size, chunk_overlap)

    sections: list[tuple[str, list[str]]] = []
    cur_heading = ""
    cur_lines: list[str] = []
    for line in text.splitlines():
        if HEADING_PAT.match(line.strip()):
            if cur_lines:
                sections.append((cur_heading, cur_lines))
                cur_lines = []
            cur_heading = line.strip()
        cur_lines.append(line)
    if cur_lines:
        sections.append((cur_heading, cur_lines))

    raw_chunks: list[Chunk] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        if len(body) <= chunk_size:
            raw_chunks.append(Chunk(text=body, source_file=source_file, heading=heading, chunk_id=""))
        else:
            for piece in split_long_text(body, chunk_size, chunk_overlap):
                raw_chunks.append(Chunk(text=piece, source_file=source_file, heading=heading, chunk_id=""))

    # 过短块并入同 section 前一块，避免碎片化
    merged: list[Chunk] = []
    for chunk in raw_chunks:
        if merged and len(chunk.text) < min_chunk_size and merged[-1].heading == chunk.heading:
            merged[-1] = Chunk(
                text=merged[-1].text + "\n" + chunk.text,
                source_file=chunk.source_file,
                heading=chunk.heading,
                chunk_id="",
            )
        else:
            merged.append(chunk)

    for i, chunk in enumerate(merged):
        chunk.chunk_id = f"{source_file}#{i}"
    return merged
