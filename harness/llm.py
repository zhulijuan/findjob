import os

from openai import OpenAI

from .config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig, api_key: str | None = None):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "缺少 API Key：请在 .env 中设置 DEEPSEEK_API_KEY "
                "（申请地址 platform.deepseek.com/api_keys）"
            )
        self.config = config
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool = False,
        stream: bool = False,
        **kwargs,
    ):
        # thinking 模式不支持采样参数，强制忽略
        if thinking:
            temperature = max_tokens = None

        params = dict(
            model=model or self.config.model,
            messages=messages,
            stream=stream,
        )
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if thinking:
            params["extra_body"] = {"thinking": {"type": "enabled"}}
        params.update(kwargs)
        return self._client.chat.completions.create(**params)
