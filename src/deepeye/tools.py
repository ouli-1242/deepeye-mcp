"""DeepEye MCP 工具实现。

四个工具均返回 ``list[TextContent]``：
- :func:`describe_image`：图片详细描述
- :func:`extract_text`：OCR 文字提取
- :func:`ask_about_image`：视觉问答
- :func:`analyze_layout`：UI 布局结构化分析（返回 JSON）
"""

from __future__ import annotations

import hashlib
import json
import re

from mcp.types import TextContent

from deepeye.cache import vision_cache
from deepeye.config import settings
from deepeye.image_utils import parse_image_source, preprocess_image
from deepeye.vision import create_vision_adapter

_DEFAULT_DESCRIBE_PROMPT = (
    "分析这张图片，按以下维度分点描述：\n"
    "1. 布局结构：各元素的位置关系和层级\n"
    "2. 主要元素：具体列出每个元素及其文字内容\n"
    "3. 颜色与样式：背景色、文字色、关键配色\n"
    "4. 潜在问题：如有遮挡、重叠、对齐异常、内容缺失等"
)
_OCR_PROMPT = "提取图片中所有文字，保持排版，不加描述。"

# 布局分析 basic 模式提示词：只要求类型 + 文本 + 位置
_LAYOUT_BASIC_PROMPT = """分析这张 UI 截图的布局结构，返回 JSON。只返回 JSON，不加任何说明文字。
JSON 格式：
{"layout_type": "布局类型", "summary": "一句话描述", "elements": [{"type": "元素类型", "text": "文本内容", "position": {"x": 0, "y": 0, "width": 0, "height": 0}, "children": []}]}
位置坐标用百分比 0-100。元素类型：nav/button/text/image/input/link/icon/card/container/list。"""

# detailed 模式：在 basic 基础上额外要求样式信息
_LAYOUT_DETAILED_EXTRA = """
额外为元素返回 styles 字段：{"background_color": "#hex", "text_color": "#hex", "font_size": "14px", "border_radius": "8px", "padding": "12px"}。"""

# 模型输出不稳定，偶发不返回 JSON，调用重试次数
_LAYOUT_RETRIES = 3


async def _run_vision(
    image_source: str,
    prompt: str,
    model: str | None = None,
    use_cache: bool = True,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
) -> str:
    """内部统一流程：解析图像源 → 预处理 → 查缓存 → 调用适配器 → 写缓存。

    Args:
        image_source: 图像来源（本地路径 / URL / data URI）。
        prompt: 提示词。
        model: 可选模型名称覆盖。
        use_cache: 是否读写缓存；False 时每次调用都请求视觉后端
            （用于输出不稳定、需重试的场景）。
        max_tokens: 输出 token 上限覆盖；None 时使用配置默认值。
        reasoning_effort: 推理深度覆盖；None 时用配置默认。
        response_format: 输出格式约束（如 ``{"type": "json_object"}``）。

    Returns:
        视觉模型返回的文本。
    """
    b64_data, mime_type = await parse_image_source(image_source)
    # 图片预处理（缩放 + 转 JPEG），失败时原样返回不阻断
    b64_data, mime_type = preprocess_image(b64_data, mime_type)

    # 用处理后的 b64 计算哈希，作为缓存 key 的一部分
    image_hash = hashlib.sha256(b64_data.encode()).hexdigest()
    effective_model = model if model is not None else ""

    # 开启缓存时先查缓存，命中则直接返回
    if settings.cache_enabled and use_cache:
        cached = vision_cache.get(image_hash, prompt, effective_model)
        if cached is not None:
            return cached

    adapter = create_vision_adapter(model)
    text = await adapter.describe(
        b64_data,
        mime_type,
        prompt,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        response_format=response_format,
    )

    # 写入缓存供下次复用；空结果不缓存，避免污染
    if settings.cache_enabled and use_cache and text:
        vision_cache.set(image_hash, prompt, effective_model, text)

    return text


async def describe_image(
    image_source: str,
    prompt: str = _DEFAULT_DESCRIBE_PROMPT,
    model: str | None = None,
) -> list[TextContent]:
    """描述图片内容。

    Args:
        image_source: 图像来源（本地路径 / 公网 URL / Base64 data URI）。
        prompt: 描述提示词，默认为简洁描述。
        model: 可选模型名称覆盖。

    Returns:
        包含图片描述文本的 ``list[TextContent]``。
    """
    try:
        description = await _run_vision(image_source, prompt, model)
        return [TextContent(type="text", text=f"图片分析结果：\n{description}")]
    except Exception as exc:
        return [TextContent(type="text", text=f"图片分析失败：{exc}")]


async def extract_text(
    image_source: str,
    language: str = "auto",
) -> list[TextContent]:
    """提取图片中的文字（OCR）。

    使用固定 OCR 提示词，强制仅返回文字、保持排版、不加描述。

    Args:
        image_source: 图像来源（本地路径 / 公网 URL / Base64 data URI）。
        language: 识别语言，``"auto"`` 自动识别；其他值会附加语言提示。

    Returns:
        包含提取文字的 ``list[TextContent]``。
    """
    prompt = _OCR_PROMPT
    if language != "auto":
        prompt += f" 优先识别语言：{language}"
    try:
        text = await _run_vision(image_source, prompt)
        return [TextContent(type="text", text=text)]
    except Exception as exc:
        return [TextContent(type="text", text=f"OCR 失败：{exc}")]


async def ask_about_image(
    image_source: str,
    question: str,
) -> list[TextContent]:
    """根据图片内容回答问题。

    Args:
        image_source: 图像来源（本地路径 / 公网 URL / Base64 data URI）。
        question: 要回答的问题。

    Returns:
        包含答案的 ``list[TextContent]``。
    """
    prompt = f"根据图片回答：{question}"
    try:
        answer = await _run_vision(image_source, prompt)
        return [TextContent(type="text", text=answer)]
    except Exception as exc:
        return [TextContent(type="text", text=f"视觉问答失败：{exc}")]


def _extract_json(text: str, require_key: str | None = None) -> str | None:
    """从模型输出中稳健提取 JSON 对象。

    模型可能返回"说明文字 + JSON"或把 JSON 包在代码块里。
    扫描所有合法 JSON 对象：
    - ``require_key`` 非空时，只返回含该键的对象（兼容不完整 JSON 里
      嵌套元素先被解析到的情况）；
    - 否则返回第一个合法对象。
    """
    # 去掉 markdown 代码围栏
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if require_key is None or (isinstance(obj, dict) and require_key in obj):
            return json.dumps(obj, ensure_ascii=False)
    return None


async def analyze_layout(
    image_source: str,
    detail: str = "basic",
    model: str | None = None,
) -> list[TextContent]:
    """UI 布局结构化分析：返回 JSON 字符串，包含元素类型、位置坐标、（可选）样式。

    Args:
        image_source: 图像来源（本地路径 / 公网 URL / Base64 data URI）。
        detail: 分析粒度，``"basic"`` 仅返回类型+文本+位置；
            ``"detailed"`` 额外返回颜色、字号、圆角等样式信息。
        model: 可选模型名称覆盖。

    Returns:
        包含 JSON 字符串的 ``list[TextContent]``；模型未返回有效 JSON 时
        返回错误提示文本。
    """
    # 根据粒度拼装提示词
    if detail == "detailed":
        prompt = _LAYOUT_BASIC_PROMPT + _LAYOUT_DETAILED_EXTRA
    else:
        prompt = _LAYOUT_BASIC_PROMPT

    try:
        for _ in range(_LAYOUT_RETRIES + 1):
            # 禁用缓存：模型输出不稳定，且散文结果可能污染缓存导致重试失效
            # 布局 JSON 较大，用 8192 防截断；json_object 保证输出合法 JSON
            text = await _run_vision(
                image_source,
                prompt,
                model,
                use_cache=False,
                max_tokens=8192,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            # 模型可能返回 "说明文字 + JSON"，用稳健提取；必须含 layout_type 顶层对象
            json_str = _extract_json(text, require_key="layout_type")
            if json_str is not None:
                return [TextContent(type="text", text=json_str)]
        return [TextContent(type="text", text="布局分析失败：模型多次未返回有效 JSON")]
    except Exception as exc:
        return [TextContent(type="text", text=f"布局分析失败：{exc}")]
