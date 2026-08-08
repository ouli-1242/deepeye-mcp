"""``GeminiVisionAdapter`` 与 ``CustomVisionAdapter`` 单元测试。

通过 ``unittest.mock.patch`` 替换适配器模块内的 ``httpx.AsyncClient``，
避免任何真实 API 调用；验证 payload 构造、URL、返回值解析与错误处理。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from deepeye_mcp.config import settings
from deepeye_mcp.vision.custom_adapter import CustomVisionAdapter
from deepeye_mcp.vision.gemini_adapter import GeminiVisionAdapter


# ---------------------------------------------------------------------------
# 辅助：构造 fake httpx.AsyncClient
# ---------------------------------------------------------------------------


def _make_fake_client(json_payload: dict, status_code: int = 200) -> AsyncMock:
    """构造一个 fake ``httpx.AsyncClient``，POST 返回指定 JSON。"""
    fake_response = MagicMock()
    fake_response.json.return_value = json_payload
    fake_response.status_code = status_code
    fake_response.raise_for_status = MagicMock()

    fake_client = AsyncMock()
    fake_client.post = AsyncMock(return_value=fake_response)
    fake_client.__aenter__.return_value = fake_client
    fake_client.__aexit__.return_value = None
    return fake_client


def _make_error_client(response: MagicMock) -> AsyncMock:
    """构造 fake client，其响应 ``raise_for_status`` 抛 HTTPStatusError。"""
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error",
        request=MagicMock(),
        response=response,
    )
    fake_client = AsyncMock()
    fake_client.post = AsyncMock(return_value=response)
    fake_client.__aenter__.return_value = fake_client
    fake_client.__aexit__.return_value = None
    return fake_client


# ---------------------------------------------------------------------------
# GeminiVisionAdapter
# ---------------------------------------------------------------------------


@patch("deepeye_mcp.vision.gemini_adapter.httpx.AsyncClient")
async def test_gemini_describe_returns_text(mock_client_cls):
    """describe 应返回 candidates[0].content.parts[0].text（去空白）。"""
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": "  一只橘猫坐在窗台上  "}]}}
        ]
    }
    mock_client_cls.return_value = _make_fake_client(payload)

    adapter = GeminiVisionAdapter(
        model="gemini-1.5-pro",
        api_key="fake-key",
    )
    text = await adapter.describe("iVBOR", "image/png", "描述图片")

    assert text == "一只橘猫坐在窗台上"


@patch("deepeye_mcp.vision.gemini_adapter.httpx.AsyncClient")
async def test_gemini_describe_payload_and_url(mock_client_cls):
    """验证 URL 拼接、query 参数与 payload 结构。"""
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": "ok"}]}}
        ]
    }
    fake_client = _make_fake_client(payload)
    mock_client_cls.return_value = fake_client

    adapter = GeminiVisionAdapter(
        model="gemini-2.0-flash",
        api_key="my-api-key",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    await adapter.describe("iVBOR==", "image/png", "描述这张图")

    fake_client.post.assert_awaited_once()
    call = fake_client.post.await_args

    url = call.args[0]
    assert url == (
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/gemini-2.0-flash:generateContent"
    )

    # query 参数 ?key=...
    params = call.kwargs.get("params")
    assert params == {"key": "my-api-key"}

    # payload 结构
    sent_payload = call.kwargs.get("json")
    assert sent_payload == {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": "iVBOR=="}},
                    {"text": "描述这张图"},
                ]
            }
        ]
    }


@patch("deepeye_mcp.vision.gemini_adapter.httpx.AsyncClient")
async def test_gemini_describe_raises_on_error_status(mock_client_cls):
    """非 2xx 响应应通过 raise_for_status 抛 HTTPStatusError。"""
    fake_response = MagicMock()
    fake_response.status_code = 500
    fake_client = _make_error_client(fake_response)
    mock_client_cls.return_value = fake_client

    adapter = GeminiVisionAdapter(model="gemini-1.5-pro", api_key="k")
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.describe("iVBOR", "image/png", "prompt")


def test_gemini_default_base_url():
    """未传 base_url 时应回退到官方端点。"""
    adapter = GeminiVisionAdapter(model="gemini-1.5-pro", api_key="k")
    assert (
        adapter.base_url
        == "https://generativelanguage.googleapis.com/v1beta"
    )


def test_gemini_reads_settings_defaults(monkeypatch):
    """未传参时应从 settings 读取 gemini_model / gemini_api_key。"""
    monkeypatch.setattr(settings, "gemini_model", "gemini-1.5-pro")
    monkeypatch.setattr(settings, "gemini_api_key", "from-settings-key")
    adapter = GeminiVisionAdapter()
    assert adapter.model == "gemini-1.5-pro"
    assert adapter.api_key == "from-settings-key"


# ---------------------------------------------------------------------------
# CustomVisionAdapter
# ---------------------------------------------------------------------------


def test_custom_missing_base_url_raises_value_error(monkeypatch):
    """custom_base_url 为空时应抛 ValueError。"""
    monkeypatch.setattr(settings, "custom_base_url", "")
    with pytest.raises(ValueError):
        CustomVisionAdapter()


def test_custom_missing_base_url_via_constructor_raises():
    """构造时显式传空 base_url 也应抛 ValueError。"""
    with pytest.raises(ValueError):
        CustomVisionAdapter(base_url="")


@patch("deepeye_mcp.vision.custom_adapter.httpx.AsyncClient")
async def test_custom_describe_returns_text(mock_client_cls):
    """describe 应返回 choices[0].message.content（去空白）。"""
    payload = {
        "choices": [
            {"message": {"content": "  hello world  "}}
        ]
    }
    mock_client_cls.return_value = _make_fake_client(payload)

    adapter = CustomVisionAdapter(
        model="qwen-vl-max",
        api_key="fake-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    text = await adapter.describe("iVBOR", "image/png", "describe")

    assert text == "hello world"


@patch("deepeye_mcp.vision.custom_adapter.httpx.AsyncClient")
async def test_custom_describe_payload_and_url(mock_client_cls):
    """验证 URL、Authorization header 与 OpenAI 兼容 payload 结构。"""
    payload = {
        "choices": [{"message": {"content": "ok"}}]
    }
    fake_client = _make_fake_client(payload)
    mock_client_cls.return_value = fake_client

    adapter = CustomVisionAdapter(
        model="qwen-vl-max",
        api_key="bearer-token",
        base_url="https://example.com/v1",
    )
    await adapter.describe("iVBOR", "image/png", "描述图片")

    fake_client.post.assert_awaited_once()
    call = fake_client.post.await_args

    assert call.args[0] == "https://example.com/v1/chat/completions"

    headers = call.kwargs.get("headers")
    assert headers["Authorization"] == "Bearer bearer-token"
    assert headers["Content-Type"] == "application/json"

    sent_payload = call.kwargs.get("json")
    assert sent_payload["model"] == "qwen-vl-max"
    assert sent_payload["max_tokens"] == 1024
    messages = sent_payload["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "描述图片"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBOR"},
    }


@patch("deepeye_mcp.vision.custom_adapter.httpx.AsyncClient")
async def test_custom_describe_no_auth_header_when_no_key(mock_client_cls):
    """api_key 为空时不应携带 Authorization 头。"""
    payload = {"choices": [{"message": {"content": "ok"}}]}
    fake_client = _make_fake_client(payload)
    mock_client_cls.return_value = fake_client

    adapter = CustomVisionAdapter(
        model="ollama-llava",
        api_key="",
        base_url="http://localhost:11434/v1",
    )
    await adapter.describe("iVBOR", "image/png", "prompt")

    call = fake_client.post.await_args
    headers = call.kwargs.get("headers")
    assert "Authorization" not in headers


@patch("deepeye_mcp.vision.custom_adapter.httpx.AsyncClient")
async def test_custom_describe_raises_on_error_status(mock_client_cls):
    """非 2xx 响应应通过 raise_for_status 抛 HTTPStatusError。"""
    fake_response = MagicMock()
    fake_response.status_code = 401
    fake_client = _make_error_client(fake_response)
    mock_client_cls.return_value = fake_client

    adapter = CustomVisionAdapter(
        model="qwen-vl-max",
        api_key="bad",
        base_url="https://example.com/v1",
    )
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.describe("iVBOR", "image/png", "prompt")


def test_custom_reads_settings_defaults(monkeypatch):
    """未传参时应从 settings 读取 custom_model / custom_api_key / custom_base_url。"""
    monkeypatch.setattr(settings, "custom_model", "qwen-vl-max")
    monkeypatch.setattr(settings, "custom_api_key", "cfg-key")
    monkeypatch.setattr(settings, "custom_base_url", "https://cfg.example.com/v1")
    adapter = CustomVisionAdapter()
    assert adapter.model == "qwen-vl-max"
    assert adapter.api_key == "cfg-key"
    assert adapter.base_url == "https://cfg.example.com/v1"
