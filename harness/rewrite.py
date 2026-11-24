from __future__ import annotations

from .config import AppConfig
from .llm import LLMClient
from .runner import build_messages


def maybe_rewrite_query(config: AppConfig, question: str) -> str:
    """按配置决定是否用 LLM 改写查询；开关关闭时原样返回。"""
    retrieval = config.retrieval
    if retrieval is None or not retrieval.rewrite.enabled:
        return question

    task = config.tasks.get("rewrite")
    if task is None:
        raise ValueError("开启 retrieval.rewrite 需要 config.yaml tasks.rewrite 任务定义")

    messages = build_messages(task, {"question": question})
    client = LLMClient(config.llm)
    response = client.chat(
        messages,
        model=task.model,
        temperature=task.temperature if task.temperature is not None else config.llm.temperature,
        max_tokens=task.max_tokens if task.max_tokens is not None else config.llm.max_tokens,
        thinking=False,
        meta={"task": "rewrite"},
    )
    return response.choices[0].message.content.strip()
