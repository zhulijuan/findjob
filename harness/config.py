import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0)
    timeout: float = Field(gt=0)


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    system_prompt: str = ""
    user_prompt: str
    model: str | None = None
    thinking: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class AppConfig(BaseModel):
    llm: LLMConfig
    tasks: dict[str, TaskConfig]

    def list_tasks(self) -> str:
        lines = []
        for name, task in self.tasks.items():
            desc = f" — {task.description}" if task.description else ""
            lines.append(f"{name}{desc}")
        return "\n".join(lines)


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    load_dotenv()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # 环境变量优先于 YAML，便于不改配置文件临时切换
    llm = data.setdefault("llm", {})
    if os.environ.get("DEEPSEEK_BASE_URL"):
        llm["base_url"] = os.environ["DEEPSEEK_BASE_URL"]
    if os.environ.get("DEEPSEEK_MODEL"):
        llm["model"] = os.environ["DEEPSEEK_MODEL"]

    return AppConfig.model_validate(data)
