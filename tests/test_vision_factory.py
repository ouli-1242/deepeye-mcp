"""``deepeye_mcp.vision`` 工厂函数 ``create_vision_adapter`` 单元测试。"""

from __future__ import annotations

import pytest

from deepeye_mcp.config import settings
from deepeye_mcp.vision import create_vision_adapter
from deepeye_mcp.vision.custom_adapter import CustomVisionAdapter
from deepeye_mcp.vision.gemini_adapter import GeminiVisionAdapter
from deepeye_mcp.vision.openai_adapter import OpenAIVisionAdapter


def test_create_vision_adapter_openai_default(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "openai")
    adapter = create_vision_adapter()
    assert isinstance(adapter, OpenAIVisionAdapter)


def test_create_vision_adapter_openai_model_override(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "openai")
    adapter = create_vision_adapter(model_override="gpt-4o-mini")
    assert isinstance(adapter, OpenAIVisionAdapter)
    assert adapter.model == "gpt-4o-mini"


def test_create_vision_adapter_openai_model_none_uses_config(monkeypatch):
    """未指定 model_override 时，应回退到 ``settings.openai_model``。"""
    monkeypatch.setattr(settings, "vision_provider", "openai")
    monkeypatch.setattr(settings, "openai_model", "gpt-4o")
    adapter = create_vision_adapter()
    assert isinstance(adapter, OpenAIVisionAdapter)
    assert adapter.model == "gpt-4o"


def test_create_vision_adapter_gemini(monkeypatch):
    """vision_provider=gemini 应返回 GeminiVisionAdapter。"""
    monkeypatch.setattr(settings, "vision_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_model", "gemini-1.5-pro")
    adapter = create_vision_adapter()
    assert isinstance(adapter, GeminiVisionAdapter)
    assert adapter.model == "gemini-1.5-pro"


def test_create_vision_adapter_gemini_model_override(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "gemini")
    adapter = create_vision_adapter(model_override="gemini-2.0-flash")
    assert isinstance(adapter, GeminiVisionAdapter)
    assert adapter.model == "gemini-2.0-flash"


def test_create_vision_adapter_custom(monkeypatch):
    """vision_provider=custom 且配置 custom_base_url 时应返回 CustomVisionAdapter。"""
    monkeypatch.setattr(settings, "vision_provider", "custom")
    monkeypatch.setattr(settings, "custom_model", "qwen-vl-max")
    monkeypatch.setattr(settings, "custom_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    adapter = create_vision_adapter()
    assert isinstance(adapter, CustomVisionAdapter)
    assert adapter.model == "qwen-vl-max"
    assert adapter.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_create_vision_adapter_custom_model_override(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "custom")
    monkeypatch.setattr(settings, "custom_base_url", "https://example.com/v1")
    adapter = create_vision_adapter(model_override="ollama-llava")
    assert isinstance(adapter, CustomVisionAdapter)
    assert adapter.model == "ollama-llava"


def test_create_vision_adapter_unknown_provider_raises_value_error(monkeypatch):
    """未知 provider 应抛 ValueError（而非 NotImplementedError）。"""
    monkeypatch.setattr(settings, "vision_provider", "unknown")
    with pytest.raises(ValueError):
        create_vision_adapter()


def test_create_vision_adapter_provider_case_insensitive(monkeypatch):
    """provider 大小写应被规范化（lower()）。"""
    monkeypatch.setattr(settings, "vision_provider", "OpenAI")
    adapter = create_vision_adapter()
    assert isinstance(adapter, OpenAIVisionAdapter)


def test_create_vision_adapter_gemini_case_insensitive(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "GEMINI")
    adapter = create_vision_adapter()
    assert isinstance(adapter, GeminiVisionAdapter)


def test_create_vision_adapter_custom_case_insensitive(monkeypatch):
    monkeypatch.setattr(settings, "vision_provider", "Custom")
    monkeypatch.setattr(settings, "custom_base_url", "https://example.com/v1")
    adapter = create_vision_adapter()
    assert isinstance(adapter, CustomVisionAdapter)
