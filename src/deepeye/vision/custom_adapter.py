"""自定义 OpenAI 兼容视觉适配器。

复用 OpenAI Chat Completions 格式，从 ``custom_*`` 配置读取连接信息，
用于任意 OpenAI 兼容多模态服务（vLLM / Ollama / 通义 / 智谱等）。
"""

from __future__ import annotations

import httpx

from deepeye.config import settings
from deepeye.vision.base import VisionAdapter


class CustomVisionAdapter(VisionAdapter):
    """基于自定义 OpenAI 兼容 Chat Completions 的视觉适配器。

    Args:
        model: 模型名称，未指定时使用 ``settings.custom_model``。
        api_key: API Key，未指定时使用 ``settings.custom_api_key``。
        base_url: 接口地址，未指定时使用 ``settings.custom_base_url``，
            仍为空则抛出 :class:`ValueError`。

    Raises:
        ValueError: ``base_url`` 为空时创建适配器。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or settings.custom_model
        self.api_key = api_key if api_key is not None else settings.custom_api_key
        base = base_url if base_url is not None else settings.custom_base_url
        base = base.strip() if base and base.strip() else ""
        if not base:
            raise ValueError(
                "custom_base_url 未配置：使用 custom 视觉后端必须设置 CUSTOM_BASE_URL"
            )
        self.base_url = base

    async def describe(
        self,
        image_b64: str,
        mime_type: str,
        prompt: str,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        response_format: dict | None = None,
    ) -> str:
        """调用自定义 OpenAI 兼容 Chat Completions 返回图像描述文本。

        Args:
            max_tokens: 输出 token 上限覆盖；None 时使用 ``settings.max_tokens``。
            reasoning_effort: 推理深度覆盖；None 时用 ``settings.reasoning_effort``。
            response_format: 输出格式约束（如 ``{"type": "json_object"}``）。

        Raises:
            httpx.HTTPStatusError: API 返回非 2xx 状态码时由
                ``raise_for_status()`` 抛出。
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        data_uri = f"data:{mime_type};base64,{image_b64}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "max_tokens": max_tokens or settings.max_tokens,
            "reasoning_effort": reasoning_effort or settings.reasoning_effort,
        }
        if response_format:
            payload["response_format"] = response_format
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = settings.request_timeout
        last_exc: Exception | None = None
        for attempt in range(settings.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < settings.max_retries:
                    continue
                raise
        else:
            if last_exc:
                raise last_exc

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
