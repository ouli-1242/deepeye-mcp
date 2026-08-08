"""DeepEye 配置加载。

基于 pydantic-settings 从环境变量与 ``.env`` 文件加载配置，支持
OpenAI / Gemini / 自定义 OpenAI 兼容服务三类视觉后端。
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """DeepEye 全局配置。

    字段命名采用 ``snake_case``，对应环境变量为全大写形式
    （例如 ``vision_provider`` ↔ ``VISION_PROVIDER``）。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 视觉后端提供者：openai / gemini / custom
    vision_provider: str = "openai"

    # ---------- OpenAI 兼容后端 ----------
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-luna"
    openai_base_url: str = ""

    # ---------- Gemini 后端（预留） ----------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-pro"

    # ---------- 自定义 OpenAI 兼容后端（预留） ----------
    custom_api_key: str = ""
    custom_base_url: str = ""
    custom_model: str = "qwen-vl-max"

    # ---------- OCR 后端 ----------
    # extract_text 工具实际使用的视觉后端：openai / gemini / custom
    ocr_backend: str = "openai"

    # ---------- 性能优化 ----------
    # 图片预处理：最大边长，超过则等比缩放后转 JPEG；0 表示禁用预处理
    image_max_dim: int = 2048
    # 结果缓存开关
    cache_enabled: bool = True
    # 缓存最大条目数（LRU 淘汰）
    cache_max_size: int = 128
    # 缓存存活秒数（TTL）
    cache_ttl: int = 3600

    # ---------- 网络请求 ----------
    # 视觉后端 HTTP 请求超时（秒）
    request_timeout: float = 120.0
    # 失败重试次数（仅对网络/超时错误重试，不对 4xx 重试）
    max_retries: int = 3
    # 视觉模型返回的最大 token 数（推理模型需更大预算，否则 content 被截断为空）
    max_tokens: int = 4096
    # 推理深度：low / medium / high；留空则不发送该参数（部分后端不支持）
    reasoning_effort: str = ""

    # ---------- 图片 URL 下载 ----------
    # 下载图片的最大字节数，超过则拒绝（防止内存耗尽）
    max_image_bytes: int = 20 * 1024 * 1024
    # 是否允许访问内网/保留地址（默认禁止，防止 SSRF；仅本地调试时设为 true）
    allow_private_urls: bool = False


settings = Settings()
