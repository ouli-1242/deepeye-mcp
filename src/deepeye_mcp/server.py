"""DeepEye MCP Server 组装。

通过 ``mcp`` 官方库的 ``stdio_server`` 启动 MCP Server，注册
``list_tools`` 与 ``call_tool`` 处理器，Server 名称为 ``deepeye``。

注意：本实现适配 ``mcp>=2.0.0`` 的 API：使用构造器注册
``on_list_tools`` / ``on_call_tool`` 处理器（而非旧版装饰器），handler
返回 ``ListToolsResult`` / ``CallToolResult``。
"""

from __future__ import annotations

import asyncio

from loguru import logger

from mcp.server import NotificationOptions, Server
from mcp.server.context import ServerRequestContext
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from deepeye_mcp import __version__
from deepeye_mcp.tools import analyze_layout, ask_about_image, describe_image, extract_text

_TOOLS: list[Tool] = [
    Tool(
        name="describe_image",
        description="描述图片内容，支持本地路径 / 公网 URL / Base64 data URI 三种来源。",
        inputSchema={
            "type": "object",
            "properties": {
                "image_source": {
                    "type": "string",
                    "description": "图像来源：本地路径、http(s) URL 或 data:image/...;base64,... 形式的 data URI。",
                },
                "prompt": {
                    "type": "string",
                    "description": "描述提示词，可选；不传则使用默认详细描述提示词。",
                },
                "model": {
                    "type": "string",
                    "description": "可选模型名称覆盖，不传则使用配置默认模型。",
                },
            },
            "required": ["image_source"],
        },
    ),
    Tool(
        name="extract_text",
        description="提取图片中的文字（OCR），保持原文排版，不加额外描述。",
        inputSchema={
            "type": "object",
            "properties": {
                "image_source": {
                    "type": "string",
                    "description": "图像来源：本地路径、http(s) URL 或 data:image/...;base64,... 形式的 data URI。",
                },
                "language": {
                    "type": "string",
                    "description": "识别语言，默认 auto 自动识别；其他值会附加语言提示。",
                },
            },
            "required": ["image_source"],
        },
    ),
    Tool(
        name="ask_about_image",
        description="根据图片内容回答问题（视觉问答）。",
        inputSchema={
            "type": "object",
            "properties": {
                "image_source": {
                    "type": "string",
                    "description": "图像来源：本地路径、http(s) URL 或 data:image/...;base64,... 形式的 data URI。",
                },
                "question": {
                    "type": "string",
                    "description": "要回答的问题。",
                },
            },
            "required": ["image_source", "question"],
        },
    ),
    Tool(
        name="analyze_layout",
        description="UI 布局结构化分析：返回 JSON，包含元素类型、位置坐标、样式（detailed 模式）。适合前端复刻场景。",
        inputSchema={
            "type": "object",
            "properties": {
                "image_source": {"type": "string", "description": "图片来源：本地绝对路径、公网URL或Base64数据"},
                "detail": {"type": "string", "description": "分析粒度：basic（类型+位置）或 detailed（含颜色/字号等样式）", "default": "basic", "enum": ["basic", "detailed"]},
                "model": {"type": "string", "description": "指定视觉模型，不填使用默认配置"}
            },
            "required": ["image_source"]
        }
    ),
]


async def list_tools(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListToolsResult:
    """返回 DeepEye 暴露的工具定义。"""
    return ListToolsResult(tools=_TOOLS)


async def call_tool(
    ctx: ServerRequestContext, params: CallToolRequestParams
) -> CallToolResult:
    """分发工具调用到对应实现。

    Raises:
        ValueError: 未知工具名时抛出。
    """
    name = params.name
    arguments: dict = params.arguments or {}
    logger.debug("call_tool name={} arguments={}", name, arguments)

    if name == "describe_image":
        describe_kwargs: dict = {"image_source": arguments["image_source"]}
        if arguments.get("prompt"):
            describe_kwargs["prompt"] = arguments["prompt"]
        if arguments.get("model"):
            describe_kwargs["model"] = arguments["model"]
        content: list[TextContent] = await describe_image(**describe_kwargs)
    elif name == "extract_text":
        content = await extract_text(
            image_source=arguments["image_source"],
            language=arguments.get("language", "auto"),
        )
    elif name == "ask_about_image":
        content = await ask_about_image(
            image_source=arguments["image_source"],
            question=arguments["question"],
        )
    elif name == "analyze_layout":
        content = await analyze_layout(**arguments)
    else:
        raise ValueError(f"未知工具: {name}")

    return CallToolResult(content=content)


server = Server(
    "deepeye",
    version=__version__,
    on_list_tools=list_tools,
    on_call_tool=call_tool,
)


async def main() -> None:
    """启动 stdio MCP Server。"""
    logger.info("启动 DeepEye MCP Server v{}", __version__)
    init_options = server.create_initialization_options(
        notification_options=NotificationOptions()
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


def run() -> None:
    """同步入口 wrapper（供 console_scripts 使用）。

    ``main`` 是 ``async def``，setuptools 生成的 exe 执行
    ``sys.exit(main())`` 只创建 coroutine 不会 await，进程秒退。
    本 wrapper 通过 ``asyncio.run`` 正常驱动事件循环。
    同时在 Windows 下强制 stdout/stderr 使用 UTF-8，避免中文乱码。
    """
    import sys

    if sys.platform == "win32":
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8")
                except Exception:
                    pass

    asyncio.run(main())


if __name__ == "__main__":
    run()
