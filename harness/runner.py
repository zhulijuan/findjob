from jinja2 import Template

from .config import AppConfig, TaskConfig
from .llm import LLMClient


def render(template: str, variables: dict | None = None) -> str:
    return Template(template).render(**(variables or {}))


def build_messages(task: TaskConfig, variables: dict | None = None) -> list[dict]:
    messages = []
    if task.system_prompt:
        messages.append({"role": "system", "content": render(task.system_prompt, variables)})
    messages.append({"role": "user", "content": render(task.user_prompt, variables)})
    return messages


def run_task(
    config: AppConfig,
    task_name: str,
    variables: dict | None = None,
    stream: bool = False,
):
    if task_name not in config.tasks:
        raise KeyError(f"任务不存在: {task_name}，可用任务: {', '.join(config.tasks)}")

    task = config.tasks[task_name]
    messages = build_messages(task, variables)
    client = LLMClient(config.llm)

    return client.chat(
        messages,
        model=task.model,
        temperature=task.temperature if task.temperature is not None else config.llm.temperature,
        max_tokens=task.max_tokens if task.max_tokens is not None else config.llm.max_tokens,
        thinking=task.thinking,
        stream=stream,
    )
